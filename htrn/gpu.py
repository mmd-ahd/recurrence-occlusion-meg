"""GPU helpers."""
import logging

import torch

log = logging.getLogger(__name__)


def find_max_batch_size(run_fn, device, candidates=(256, 192, 128, 96, 64, 48, 32, 16, 8, 4),
                        default=32, max_memory_fraction=0.8):
    """Largest candidate batch size whose forward pass stays under a fraction of GPU memory.

    Args:
        run_fn: Called with a random ``(bs, 3, 224, 224)`` tensor; should run the most expensive
            forward path that will be used.
        device: ``torch.device``; on CPU ``default`` is returned.
        candidates: Batch sizes to try, largest first.
        default: Batch size used on CPU.
        max_memory_fraction: Peak allocated memory must stay below this share of total VRAM.

    Returns:
        The chosen batch size (1 if none fit).
    """
    if device.type != 'cuda':
        return default

    total = torch.cuda.get_device_properties(device).total_memory
    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            dummy = torch.randn(bs, 3, 224, 224, device=device)
            with torch.no_grad():
                run_fn(dummy)
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device)
            del dummy
            torch.cuda.empty_cache()
            if peak < max_memory_fraction * total:
                log.info('batch size %d (peak %.2f / %.2f GB)', bs, peak / 1e9, total / 1e9)
                return bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
    log.info('batch size 1 (no candidate fit in memory)')
    return 1
