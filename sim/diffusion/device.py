"""Explicit accelerator operations. Formal training never falls back to CPU."""
import torch


def resolve_device(value):
    device = torch.device(value)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; CPU fallback is forbidden")
        if device.index not in (None, 0):
            raise ValueError("single-GPU training requires cuda:0")
        # Memory-stat APIs do not lazily initialize CUDA themselves.
        torch.cuda.set_device(0)
        return torch.device("cuda:0")
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS unavailable; CPU fallback is forbidden")
        return device
    raise ValueError("formal training requires CUDA or MPS; CPU fallback is forbidden")


def inference_device():
    """Use the accelerator on this host without falling back to CPU."""
    if torch.cuda.is_available():
        return resolve_device("cuda:0")
    if torch.backends.mps.is_available():
        return resolve_device("mps")
    raise RuntimeError("Diffusion inference requires CUDA or MPS; CPU fallback is forbidden")


def seed_all(seed, device):
    import random
    import numpy as np
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    elif device.type == "mps":
        torch.mps.manual_seed(seed)


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda": torch.cuda.synchronize(device)
    elif device.type == "mps": torch.mps.synchronize()


def empty_cache(device):
    device = torch.device(device)
    if device.type == "cuda": torch.cuda.empty_cache()
    elif device.type == "mps": torch.mps.empty_cache()


def memory_peaks(device):
    device = torch.device(device)
    if device.type != "cuda": return {}
    return dict(cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(device),
                cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(device))
