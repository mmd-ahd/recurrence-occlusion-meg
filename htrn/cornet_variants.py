"""CORnet-Z / -RT / -S readout conditions.

None of the CORnet variants has top-down feedback, so each contributes at most a feedforward and
a recurrent condition:

* CORnet-Z: purely feedforward - a single ``Feedforward`` condition.
* CORnet-RT: ``Feedforward`` = each block's own recurrent state forced to zero, read at t=3;
  ``Recurrent`` = native self-recurrence, read at the final step t=4.
* CORnet-S: ``Feedforward`` = a single internal iteration per block; ``Recurrent`` = the native
  number of internal iterations.
"""
import types

import torch.nn.functional as F

import cornet

CORNET_CONDITIONS = {
    'CORNet-Z':  [{'name': 'Feedforward'}],
    'CORNet-RT': [{'name': 'Feedforward', 'force_zero_state': True},
                  {'name': 'Recurrent',   'force_zero_state': False}],
    'CORNet-S':  [{'name': 'Feedforward', 'feedforward': True},
                  {'name': 'Recurrent',   'feedforward': False}],
}


def extract_cornet_z(model, images):
    """Activations of V1, V2, V4 and IT for CORnet-Z."""
    m = model.module
    v1 = m.V1(images)
    v2 = m.V2(v1)
    v4 = m.V4(v2)
    it = m.IT(v4)
    return {'V1': v1, 'V2': v2, 'V4': v4, 'IT': it}


def extract_cornet_rt(model, images, force_zero_state):
    """Re-run ``CORnet_RT.forward`` but return all four areas.

    The image re-enters V1 at every step and V2/V4/IT read the previous step's output of the area
    below. With ``force_zero_state`` every block's own previous output is discarded, leaving only
    the cross-area feedforward pipeline (read at t=3); otherwise the native recurrence runs to
    t=4.
    """
    m = model.module
    blocks = ['V1', 'V2', 'V4', 'IT']
    outputs = {'inp': images}
    states = {}
    for b in blocks:
        this_inp = images if b == 'V1' else None
        out, st = getattr(m, b)(this_inp, batch_size=images.size(0))
        outputs[b] = out
        states[b] = None if force_zero_state else st

    n_times = 4 if force_zero_state else 5
    target_t = 3 if force_zero_state else 4
    for t in range(1, n_times):
        new_outputs = {'inp': images}
        for i, b in enumerate(blocks):
            prev_b = blocks[i - 1] if i > 0 else 'inp'
            out, st = getattr(m, b)(new_outputs.get(prev_b, outputs.get(prev_b)), states[b])
            new_outputs[b] = out
            states[b] = None if force_zero_state else st
        outputs = new_outputs
        if t == target_t:
            break

    return {b: outputs[b] for b in blocks}


def staged_corblock_s_forward(self, inp, max_t=None):
    """``CORblock_S.forward`` with an early exit after internal iteration ``max_t``.

    The native forward only exposes the final iteration's output, so reading an intermediate
    iteration requires re-implementing the loop.
    """
    x = self.conv_input(inp)
    n = self.times if max_t is None else max_t + 1
    output = None
    for t in range(n):
        if t == 0:
            skip = self.norm_skip(self.skip(x))
            self.conv2.stride = (2, 2)
        else:
            skip = x
            self.conv2.stride = (1, 1)

        x = self.conv1(x)
        x = getattr(self, f'norm1_{t}')(x)
        x = self.nonlin1(x)

        x = self.conv2(x)
        x = getattr(self, f'norm2_{t}')(x)
        x = self.nonlin2(x)

        x = self.conv3(x)
        x = getattr(self, f'norm3_{t}')(x)

        x = x + skip
        x = self.nonlin3(x)
        output = self.output(x)

    return output


def patch_cornet_s(model):
    """Replace V2/V4/IT ``forward`` with the early-exit version (V1 has no repetition)."""
    for name in ('V2', 'V4', 'IT'):
        block = getattr(model.module, name)
        block.forward = types.MethodType(staged_corblock_s_forward, block)


def extract_cornet_s(model, images, feedforward):
    """Activations of all areas of CORnet-S; with `feedforward`, each block stops after its first iteration."""
    m = model.module
    max_t = 0 if feedforward else None
    v1 = m.V1(images)
    v2 = m.V2(v1, max_t=max_t)
    v4 = m.V4(v2, max_t=max_t)
    it = m.IT(v4, max_t=max_t)
    return {'V1': v1, 'V2': v2, 'V4': v4, 'IT': it}


def extract_areas(variant, model, images, condition):
    """All four area activations of ``variant`` under one entry of ``CORNET_CONDITIONS``."""
    if variant == 'CORNet-Z':
        return extract_cornet_z(model, images)
    if variant == 'CORNet-RT':
        return extract_cornet_rt(model, images, condition['force_zero_state'])
    if variant == 'CORNet-S':
        return extract_cornet_s(model, images, condition['feedforward'])
    raise ValueError(f'Unknown CORNet variant: {variant}')


def pool_area(tensor):
    """Globally average-pooled activations as a NumPy array."""
    return F.adaptive_avg_pool2d(tensor, 1).flatten(1).detach().cpu().numpy()


def load_cornet_variant(variant, device, pretrained=True):
    """Load a pretrained CORnet variant (``'CORNet-Z'``, ``'CORNet-RT'`` or ``'CORNet-S'``)."""
    letter = variant.split('-')[1].lower()
    model = cornet.get_model(letter, pretrained=pretrained)
    if variant == 'CORNet-S':
        patch_cornet_s(model)
    model.to(device)
    model.eval()
    return model
