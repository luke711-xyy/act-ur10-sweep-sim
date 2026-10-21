"""Read-only schema-v5 adapter; never exposes simulator supervision to DP."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from sim.act.dataset import ActDataset, _record_image_paths

VIEWS = ("corner_xminus_yminus", "corner_xminus_yplus",
         "corner_xplus_yminus", "corner_xplus_yplus", "wrist")


def normalize(x, stats):
    return np.asarray(x, dtype=np.float32) * np.asarray(stats["scale"],dtype="f4") + np.asarray(stats["offset"],dtype="f4")


def denormalize(x, stats):
    return (np.asarray(x,dtype="f4")-np.asarray(stats["offset"],dtype="f4"))/np.asarray(stats["scale"],dtype="f4")


def fit_limits(x):
    """Official DP limits transform to [-1,1]; constant channels map to zero."""
    low, high = x.min(axis=0), x.max(axis=0)
    width = high-low
    scale = 2/np.where(width < 1e-4, 2., width)
    offset = -(high+low)*scale/2
    return {k:np.asarray(v).tolist() for k,v in
            dict(min=low,max=high,scale=scale,offset=offset).items()}


class DiffusionDataset(ActDataset):
    def __init__(self, root):
        super().__init__(str(root), camera_views=VIEWS)
        if not self.index:
            raise ValueError("no valid schema-v5 successful expert training samples")
        self.numeric = {}
        self.image_paths = {}
        digest = hashlib.sha256((self.root/"manifest.jsonl").read_bytes())
        for record in self.records:
            path = self.root/record["arrays"]
            digest.update(path.read_bytes())
            with np.load(path,allow_pickle=False) as z:
                data = {k:np.asarray(z[k]).copy() for k in
                        ("state","environment_state","action","action_valid","t")}
            n = len(data["action"])
            if (data["environment_state"].shape != (n,1)
                    or data["action_valid"].shape != (n,)
                    or data["t"].shape != (n,)
                    or not np.allclose(np.diff(data["t"])[data["action_valid"][1:]],
                                       .04,atol=1e-5,rtol=0)):
                raise ValueError("DP requires target count1 and consecutive 40ms frames")
            for key in ("state","environment_state","action"):
                if not np.isfinite(data[key]).all():
                    raise ValueError(f"nonfinite {key} in {record['episode_id']}")
            eid = record["episode_id"]
            self.numeric[eid] = data
            self.image_paths[eid] = _record_image_paths(record)
        self.identity = digest.hexdigest()
        self.normalization = {}
        for key in ("state","action"):
            values = np.concatenate([d[key][d["action_valid"]]
                                     for d in self.numeric.values()])
            self.normalization[key] = fit_limits(values)

    def __getitem__(self, index):
        _, rec, i = self.index[index]
        eid = rec["episode_id"]
        d = self.numeric[eid]
        obs_ids = [max(0,i-1),i]
        state = d["state"][obs_ids].astype("f4")
        obs = {"state":state,
               "environment_state":d["environment_state"][obs_ids].astype("f4")}
        for view in self.camera_views:
            images = []
            for j in obs_ids:
                with Image.open(self.root/self.image_paths[eid][view][j]) as image:
                    if image.size != (320,320) or image.mode != "RGB":
                        raise ValueError("DP requires original 320x320 RGB, no resizing")
                    images.append(np.asarray(image,dtype="f4").transpose(2,0,1)/255)
            obs[view] = np.stack(images)
        actions = np.zeros((28,4),dtype="f4")
        mask = np.zeros(28,dtype=bool)
        # 0 historical, 1..25 executable, 26..27 structural padding.
        for slot in range(26):
            j = i-1+slot
            if 0 <= j < len(d["action"]) and d["action_valid"][j]:
                actions[slot] = d["action"][j]
                mask[slot] = True
        return {"obs":obs,"state":state,"action":actions,"valid_mask":mask,
                "timestamps":d["t"][obs_ids].copy()}

    def audit(self):
        return dict(identity=self.identity,episodes=len(self.records),
                    frames=sum(len(d["t"]) for d in self.numeric.values()),
                    valid_windows=len(self.index),camera_views=list(self.camera_views),
                    split="train",validation=None,normalization=self.normalization)
