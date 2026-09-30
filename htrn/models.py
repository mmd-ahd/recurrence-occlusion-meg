"""HTRN: a ResNet-50 with a 256-D additive top-down feedback loop, plus its readout stages.

The three variants compared in the paper are readout stages of one trained network:

* HTRN-FF: stage 0 - only the shortcut projections run (no residual blocks, no feedback).
* HTRN-LR: stage 6 - all residual blocks run (local recurrence), no top-down feedback.
* HTRN-TD: stage 7 - local recurrence plus accumulated top-down feedback.

Stage ``s`` of the schedule activates one more residual block per layer until every block is active
(``generate_stage_schedule``). A block that is not active is replaced by its shortcut only.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import Bottleneck, ResNet as TorchvisionResNet

from dfbmodels.equilibrium import EquilibriumModel
from dfbmodels.state import StateExpDecay

RESNET50_BLOCKS = (3, 4, 6, 3)


class HTRN(TorchvisionResNet, EquilibriumModel):
    """ResNet-50 whose layer4 output is compressed to a 256-D state and fed back additively.

    The state layout matches the 342 ``state_dict`` keys of the released checkpoints.
    """

    def __init__(self, fb_filters=256, n_classes=1000, shape=(224, 224), tau=1.0):
        super().__init__(block=Bottleneck, layers=list(RESNET50_BLOCKS), num_classes=n_classes)
        self.shape = shape
        self.block_expansion = Bottleneck.expansion
        self.fb_filters = fb_filters

        del self.layer1, self.layer2, self.layer3, self.layer4

        self.inplanes = 64
        # Feedback pathway: 256-D state -> 64-D, added to the frontend output
        self.feedback_gate = nn.Conv2d(fb_filters, 64, kernel_size=1, stride=1, bias=False)

        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        # 2048-D layer4 features -> 256-D state space
        self.compress = nn.Conv2d(512 * self.block_expansion, fb_filters, kernel_size=1)

        self.frontend = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        self.classification_head = nn.Sequential(self.avgpool, nn.Flatten(start_dim=1), self.fc)

        self.state = torch.zeros(())
        self.error_head = StateExpDecay(fb_filters, (shape[0] // 32, shape[1] // 32), tau=tau)

        self.layernorm_error = nn.GroupNorm(num_groups=8, num_channels=fb_filters)
        self.layernorm_input = nn.GroupNorm(num_groups=8, num_channels=fb_filters)
        for norm in (self.layernorm_error, self.layernorm_input):
            nn.init.constant_(norm.weight, 1)
            nn.init.constant_(norm.bias, 0)
            norm.weight.requires_grad = False
            norm.bias.requires_grad = False

        self.T = nn.Parameter(torch.zeros(()), requires_grad=False)

    def to(self, device="cuda:0", **kwargs):
        """Move the model and its feedback state to `device`."""
        self.state = self.state.to(device=device)
        return super().to(device=device, **kwargs)

    @property
    def states(self):
        """Feedback state(s) iterated by ``EquilibriumModel.to_equilibrium``."""
        return {"layer4": self.state}

    @property
    def inlayer1(self):
        """Softmax-normalised feedback state, upsampled to the layer1 input resolution."""
        upsampled = F.interpolate(
            self.states["layer4"],
            size=(self.shape[0] // 4, self.shape[1] // 4),
            mode='bilinear',
            align_corners=False,
        )
        return torch.softmax(self.layernorm_input(upsampled), dim=1)

    def _init_states(self, x):
        """Initialise the feedback state with small Gaussian noise (consumes the torch RNG)."""
        self.state = torch.ones(
            (x.shape[0], self.fb_filters, self.shape[0] // 32, self.shape[1] // 32),
            device=x.device,
        )
        nn.init.normal_(self.state, mean=0.0, std=1e-3)

    def step(self, x):
        """One full-depth feedback iteration: inject feedback, run all blocks, update the state."""
        mixed_signal = x + self.feedback_gate(self.inlayer1)

        out = self.layer1(mixed_signal)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        semantic_features = self.compress(out)
        self.state = self.state + self.error_head(
            self.layernorm_error(semantic_features), T=self.T
        )
        self.T.data.fill_(self.T.item() + 1)
        return out


def load_htrn(checkpoint_path, device='cuda'):
    """Load an HTRN checkpoint (strict) and return the model in eval mode on ``device``."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt

    model = HTRN(
        fb_filters=state_dict['compress.weight'].shape[0] if 'compress.weight' in state_dict else 256,
        n_classes=state_dict['fc.weight'].shape[0] if 'fc.weight' in state_dict else 1000,
        shape=(224, 224),
        tau=1.0,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def layer_blocks(model):
    """The residual blocks of layer1..layer4 as four lists."""
    return [list(getattr(model, f'layer{i}').children()) for i in (1, 2, 3, 4)]


def generate_stage_schedule(caps=RESNET50_BLOCKS, n_tail_stages=5):
    """Number of active blocks per layer at each readout stage.

    Stage 0 activates none; each following stage adds one block per layer until ``caps`` is
    reached (stage 6 for ResNet-50), then ``n_tail_stages`` further stages keep full depth.
    """
    caps = list(caps)
    schedule = {0: [0, 0, 0, 0]}
    current = [0, 0, 0, 0]

    stage = 1
    while current != caps:
        for i in range(len(caps)):
            if current[i] < caps[i]:
                current[i] += 1
        schedule[stage] = list(current)
        stage += 1

    for _ in range(n_tail_stages):
        schedule[stage] = list(caps)
        stage += 1

    return schedule


def _shortcut_only(x, block):
    """Apply a block's shortcut projection (or identity) and ReLU, skipping its residual path."""
    residual = block.downsample(x) if block.downsample is not None else x
    return torch.relu(residual)


def run_blocks(x, blocks, active_count):
    """Run the first ``active_count`` blocks fully and the remaining ones as shortcut-only."""
    out = x
    for i in range(active_count):
        out = blocks[i](out)
    for i in range(active_count, len(blocks)):
        out = _shortcut_only(out, blocks[i])
    return out


def _run_layers(x, layers, block_counts):
    """Run layer1..layer4 in sequence; returns the four layer outputs."""
    outs = []
    for layer, count in zip(layers, block_counts):
        x = run_blocks(x, layer, count)
        outs.append(x)
    return outs


def _feedback_input(model, frontend_out, last_out, stage):
    """Update the feedback state from ``last_out`` and return frontend output plus feedback."""
    semantic_features = model.compress(last_out)
    T_tensor = torch.tensor(float(stage), device=frontend_out.device)
    model.state = model.state + model.error_head(
        model.layernorm_error(semantic_features), T=T_tensor
    )
    return frontend_out + model.feedback_gate(model.inlayer1)


def stage_outputs(model, layers, images, schedule, target_stage, use_topdown):
    """Layer1..layer4 outputs at one readout stage.

    Without top-down feedback the stage is a single pass with ``schedule[target_stage]`` active
    blocks. With feedback, stages 0..``target_stage`` are run in sequence, each injecting the
    accumulated feedback state while one more block per layer becomes active.
    """
    frontend_out = model.frontend(images)

    if not use_topdown:
        return _run_layers(frontend_out, layers, schedule[target_stage])

    model._init_states(frontend_out)
    model.T.data.zero_()

    outs = _run_layers(frontend_out, layers, schedule[0])
    for stage in range(1, target_stage + 1):
        in_signal = _feedback_input(model, frontend_out, outs[-1], stage)
        outs = _run_layers(in_signal, layers, schedule[stage])
    return outs


def pooled_stage_sweep(model, layers, schedule, images, lr_max=6, td_max=7):
    """Globally pooled layer4 features of every LR and TD stage for one batch.

    Returns ``{'LR_0'..'LR_<lr_max>', 'TD_0'..'TD_<td_max>': (batch, 2048) array}``. The TD sweep
    reuses one running feedback state across stages, so its cost is linear in ``td_max``.
    """
    def pool(out4):
        return F.adaptive_avg_pool2d(out4, 1).flatten(1).detach().cpu().numpy()

    feats = {}
    frontend_out = model.frontend(images)

    for stage in range(lr_max + 1):
        feats[f'LR_{stage}'] = pool(_run_layers(frontend_out, layers, schedule[stage])[-1])

    model._init_states(frontend_out)
    model.T.data.zero_()

    last_out = _run_layers(frontend_out, layers, schedule[0])[-1]
    feats['TD_0'] = pool(last_out)
    for stage in range(1, td_max + 1):
        in_signal = _feedback_input(model, frontend_out, last_out, stage)
        last_out = _run_layers(in_signal, layers, schedule[stage])[-1]
        feats[f'TD_{stage}'] = pool(last_out)
    return feats


def sequential_topdown(model, images, n_steps=5):
    """Layer4 output after ``n_steps`` full-depth feedback iterations (5 in training).

    This runs the same ``n_steps`` iterations as ``dfbmodels.models.resnet.ResNet.forward(images,
    T=n_steps)`` but always completes them all (that forward stops early once the state stabilises)
    and starts from a noise-initialised state instead of zeros.
    """
    frontend_out = model.frontend(images)
    model._init_states(frontend_out)
    model.T.data.zero_()
    out = None
    for _ in range(n_steps):
        out = model.step(frontend_out)
    return out
