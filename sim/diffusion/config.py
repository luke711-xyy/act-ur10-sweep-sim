"""Dependency-free workbench defaults; importing UI must not configure Torch."""
DEFAULT_DATASET = "/Users/huazhi_luke/.codex/worktrees/act-five-view/runs/act_dataset_five_view_rudra_300_smooth320_v1"
DEFAULT_OUT = "runs/diffusion_policy_mvp"


def read_launch_config(path):
    import json
    from pathlib import Path
    value=json.loads(Path(path).read_text())
    if not isinstance(value,dict) or set(value)!={"dataset","out","training"}:
        raise ValueError("launch JSON requires dataset, out and training")
    if not all(isinstance(value[k],str) and value[k] for k in ("dataset","out")) or not isinstance(value["training"],dict):
        raise ValueError("invalid launch paths or training settings")
    return value
