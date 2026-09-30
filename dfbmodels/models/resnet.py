import torch
import torch.nn as nn
import torch.nn.functional as F
from dfbmodels.state import StateExpDecay
from dfbmodels.equilibrium import EquilibriumModel
from torchvision.models.resnet import BasicBlock, Bottleneck
from torchvision.models import ResNet as _ResNet

class ResNet(_ResNet, EquilibriumModel):
    """
    Modified DFM ResNet supporting 256D semantic bottleneck states, parallel additive feedback gating,
    and backbone weight freezing for ImageNet co-adaptation. `fc` is never frozen by
    set_backbone_freeze, even when the rest of the backbone is (freeze_backbone=True) -- see
    set_backbone_freeze's docstring.
    """

    def __init__(
        self, 
        frontend='resnet', 
        fb_filters=256, 
        registers=0, 
        n_classes=1000, 
        shape=(224, 224), 
        tau=1e0, 
        block=BasicBlock, 
        layers=[3, 4, 6, 3],
        freeze_backbone=False
    ):
        super(ResNet, self).__init__(block, layers, num_classes=n_classes)

        self.shape = shape
        self.block_expansion = block.expansion
        self.fb_filters = fb_filters
        self.registers = registers
        self.classes = n_classes

        # Re-build backbone layers with pristine, native channel shapes
        del self.layer1, self.layer2, self.layer3, self.layer4

        self.fc = nn.Linear(512 * self.block_expansion, n_classes)
        self.inplanes = 64  # Native 64 channels

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if frontend == 'resnet':
            self.frontend = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool)
        else:
            raise NotImplementedError(f"Frontend '{frontend}' is not supported.")

        # Initialize feedforward weights
        self.initialize()

        # --- DFM BOTTLENECK & PARALLEL ADDITIVE GATE ---
        # 1. Compress 2048D feedforward output down to 256D semantic state
        self.compress = nn.Conv2d(512 * self.block_expansion, fb_filters, kernel_size=1)
        
        # 2. Project 256D top-down state down to 64D entry visual space
        self.feedback_gate = nn.Conv2d(fb_filters, 64, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.feedback_gate.weight, mode="fan_out", nonlinearity="linear")

        # Classification Head
        self.classification_head = nn.Sequential(*[self.avgpool, nn.Flatten(start_dim=1), self.fc])

        # State and Error Dynamics (Output resolution is H/32 x W/32)
        self.state = torch.zeros(())
        self.error_head = self.create_error_head(fb_filters, self.shape, tau=tau)

        # GroupNorm over 256 channels (16 groups = 16 channels per group)
        num_groups = 16 if fb_filters % 16 == 0 else 8
        self.layernorm_error = nn.GroupNorm(num_groups=num_groups, num_channels=fb_filters)
        self.layernorm_input = nn.GroupNorm(num_groups=num_groups, num_channels=fb_filters)

        nn.init.constant_(self.layernorm_error.weight, 1.0)
        nn.init.constant_(self.layernorm_error.bias, 0.0)
        nn.init.constant_(self.layernorm_input.weight, 1.0)
        nn.init.constant_(self.layernorm_input.bias, 0.0)

        # Freeze normalization scaling weights for numeric stability
        self.layernorm_error.weight.requires_grad = False
        self.layernorm_error.bias.requires_grad = False
        self.layernorm_input.weight.requires_grad = False
        self.layernorm_input.bias.requires_grad = False

        self.T = nn.Parameter(torch.zeros(()), requires_grad=False)

        # Optional Hard-Freezing of Backbone Weights
        if freeze_backbone:
            self.set_backbone_freeze(True)

    def create_error_head(self, fb_filters, shape, tau=1e0):
        """Creates the StateExpDecay module operating over H/32 x W/32 state spatial grid."""
        return StateExpDecay(fb_filters, (shape[0] // 32, shape[1] // 32), tau=tau)

    def initialize(self):
        """Kaiming initialization for Conv layers and FC head."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0.0)

    def set_backbone_freeze(self, freeze=True):
        """Freeze or unfreeze the conv backbone (frontend and layer1..layer4).

        `fc` always stays trainable so the pretrained readout can adapt to the
        feedback-augmented representation.
        """
        for module in [self.frontend, self.layer1, self.layer2, self.layer3, self.layer4]:
            for param in module.parameters():
                param.requires_grad = not freeze
        for param in self.fc.parameters():
            param.requires_grad = True

    def to(self, device="cuda:0", **kwargs):
        self.state = self.state.to(device=device)
        return super(ResNet, self).to(device=device, **kwargs)

    @property
    def states(self):
        return {"layer4": self.state}

    @property
    def inlayer1(self):
        """
        Upsamples the 256D state vector to Layer1 spatial resolution (H/4 x W/4)
        and applies GroupNorm + Softmax.
        """
        states = self.states
        upsampled = F.interpolate(
            states["layer4"],
            size=(self.shape[0] // 4, self.shape[1] // 4),
            mode='bilinear',
            align_corners=False
        )
        return torch.softmax(self.layernorm_input(upsampled), dim=1)

    def _init_states(self, x):
        """Initializes state tensor with normal distribution N(0, 0.001)."""
        self.state = torch.ones(
            (x.shape[0], self.fb_filters, self.shape[0] // 32, self.shape[1] // 32),
            device=x.device
        )
        nn.init.normal_(self.state, mean=0.0, std=1e-3)

    def step(self, x):
        """
        Executes a single top-down feedback iteration:
        1. Project 256D state to 64D entry space via feedback_gate.
        2. Combine entry feature x with feedback signal.
        3. Pass through feedforward visual hierarchy.
        4. Compress features to 256D and update internal state.
        """
        # Top-down signal projected to 64D and additively mixed
        fb_signal = self.feedback_gate(self.inlayer1)
        mixed_signal = x + fb_signal

        # Feedforward pass
        out = self.layer1(mixed_signal)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # Compress to 256D semantic state space
        semantic_features = self.compress(out)

        # Update state via StateExpDecay
        self.state = self.state + self.error_head(self.layernorm_error(semantic_features), T=self.T)

        # Safe counter update
        self.T.data.fill_(self.T.item() + 1)
        return out

    def prediction(self):
        """Decodes predictions from final layer4 features using classification head."""
        return self.classification_head(self.state_features if hasattr(self, 'state_features') else self.last_out)

    def classify_features(self, out):
        """Applies classification head directly over feedforward feature maps."""
        return self.classification_head(out)

    def forward(self, x, T=0, atol=1e-2):
        x = self.frontend(x)
        self.T.data.zero_()
        self._init_states(x)

        if T == 0:
            fb_signal = self.feedback_gate(self.inlayer1)
            out = self.layer1(x + fb_signal)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            return self.classify_features(out)

        # Fixed point iterations to equilibrium state
        self.to_equilibrium(x, None, None, T=T - 1, atol=atol)
        
        # Execute final pass with optimized state
        final_features = self.step(x)
        return self.classify_features(final_features)


# --- SUBCLASSES ---

class ResNet18(ResNet):
    def __init__(self, frontend='resnet', fb_filters=256, registers=0, n_classes=1000, shape=(224, 224), tau=1e0, freeze_backbone=False):
        super(ResNet18, self).__init__(
            frontend=frontend, fb_filters=fb_filters, registers=registers, 
            n_classes=n_classes, shape=shape, tau=tau, block=BasicBlock, layers=[2, 2, 2, 2],
            freeze_backbone=freeze_backbone
        )

class ResNet34(ResNet):
    def __init__(self, frontend='resnet', fb_filters=256, registers=0, n_classes=1000, shape=(224, 224), tau=1e0, freeze_backbone=False):
        super(ResNet34, self).__init__(
            frontend=frontend, fb_filters=fb_filters, registers=registers, 
            n_classes=n_classes, shape=shape, tau=tau, block=BasicBlock, layers=[3, 4, 6, 3],
            freeze_backbone=freeze_backbone
        )

class ResNet50(ResNet):
    def __init__(self, frontend='resnet', fb_filters=256, registers=0, n_classes=1000, shape=(224, 224), tau=1e0, freeze_backbone=False):
        super(ResNet50, self).__init__(
            frontend=frontend, fb_filters=fb_filters, registers=registers, 
            n_classes=n_classes, shape=shape, tau=tau, block=Bottleneck, layers=[3, 4, 6, 3],
            freeze_backbone=freeze_backbone
        )