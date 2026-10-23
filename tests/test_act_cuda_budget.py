"""Keep a second ACT trainer inside its own CUDA allocator budget."""
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sim.act import train


def config(device="cuda", fraction=None):
    return SimpleNamespace(act={"device": device, "cuda_memory_fraction": fraction})


def test_optional_budget_preserves_existing_training_defaults():
    with patch("torch.cuda.set_per_process_memory_fraction") as apply:
        assert hasattr(train, "configure_cuda_memory_budget"), "training needs an opt-in CUDA allocator budget"
        train.configure_cuda_memory_budget(config())
        apply.assert_not_called()


def test_budget_applies_only_to_the_current_cuda_process():
    with patch("torch.cuda.set_per_process_memory_fraction") as apply:
        train.configure_cuda_memory_budget(config("cuda:0", 0.375))
        apply.assert_called_once_with(0.375, device="cuda:0")


def test_unindexed_cuda_uses_the_default_device_api_contract():
    with patch("torch.cuda.set_per_process_memory_fraction") as apply:
        train.configure_cuda_memory_budget(config("cuda", 0.375))
        apply.assert_called_once_with(0.375, device=None)


@pytest.mark.parametrize("fraction", [0, -0.1, 1.01, float("nan")])
def test_invalid_budget_fails_before_cuda_allocation(fraction):
    with patch("torch.cuda.set_per_process_memory_fraction") as apply:
        with pytest.raises(ValueError, match="fraction"):
            train.configure_cuda_memory_budget(config(fraction=fraction))
        apply.assert_not_called()


def test_cuda_budget_does_not_silently_apply_to_cpu():
    with pytest.raises(ValueError, match="CUDA"):
        train.configure_cuda_memory_budget(config("cpu", 0.375))
