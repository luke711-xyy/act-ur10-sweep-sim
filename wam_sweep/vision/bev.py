"""Calibrated table-plane inverse projection and masked multiscale fusion."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np
import torch
from torch import nn

from ..contracts import CAMERA_VIEWS
from .features import ActFeaturePacket, LAYERS, validate_capture_metadata


@dataclass(frozen=True)
class BEVSpec:
    x_bounds: tuple[float, float] = (-.52, .52)
    y_bounds: tuple[float, float] = (-.42, .42)
    resolution: float = .005
    plane_z: float = 0.

    def __post_init__(self):
        if (not all(math.isfinite(x) for x in (*self.x_bounds, *self.y_bounds, self.resolution, self.plane_z))
                or self.resolution <= 0 or any(b <= a for a, b in (self.x_bounds, self.y_bounds))):
            raise ValueError('invalid BEV geometry')
        for a, b in (self.x_bounds, self.y_bounds):
            if not math.isclose((b-a)/self.resolution, round((b-a)/self.resolution), abs_tol=1e-7):
                raise ValueError('BEV bounds must divide into whole cells')

    @property
    def shape(self):
        return (round((self.y_bounds[1]-self.y_bounds[0])/self.resolution),
                round((self.x_bounds[1]-self.x_bounds[0])/self.resolution))


@dataclass(frozen=True)
class BEVOutput:
    features: torch.Tensor  # B,128,Y=168,X=208 at default 5mm
    valid: torch.Tensor  # B,1,Y,X bool, union of valid camera frusta
    view_valid: torch.Tensor  # B,5,Y,X bool
    timestamp: float
    spec: BEVSpec


def validate_calibration(packet, calibration):
    validate_capture_metadata(calibration,packet.timestamp,packet.input_hash)


class BEVEncoder(nn.Module):
    """Use known table z only, never object height/segmentation/GT occupancy.

    Convolutions project each frozen scale to width128; bilinear sampling transfers
    its spatial raster to world cells. Equal valid-camera averaging and learned
    scale softmax preserve zero in unobserved cells. Occlusion by other objects
    is inferred by the detector, not secretly supplied as ground-truth masks.
    """
    def __init__(self, spec: BEVSpec | None = None):
        super().__init__()
        self.spec = spec or BEVSpec()
        self.projections = nn.ModuleDict({layer: nn.Conv2d(ch, 128, 1, bias=False)
                                         for layer, (ch, _) in LAYERS.items()})
        self.scale_logits = nn.Parameter(torch.zeros(3))

    def world_grid(self, *, device=None):
        h, w = self.spec.shape
        xs = self.spec.x_bounds[0] + (torch.arange(w, device=device, dtype=torch.float32)+.5)*self.spec.resolution
        ys = self.spec.y_bounds[0] + (torch.arange(h, device=device, dtype=torch.float32)+.5)*self.spec.resolution
        y, x = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack((x, y, torch.full_like(x, self.spec.plane_z)), -1)

    @staticmethod
    def project_points(points, camera):
        if camera.convention != 'mujoco_x_right_y_up_minus_z_forward' or tuple(camera.size) != (320, 320):
            raise ValueError('unsupported camera convention/size')
        k, c = np.asarray(camera.intrinsics), np.asarray(camera.camera_to_world)
        if (k.shape != (3,3) or c.shape != (4,4) or not np.isfinite(k).all() or not np.isfinite(c).all()
                or k[0,0] <= 0 or k[1,1] <= 0 or not np.allclose(k[2], [0,0,1])
                or not np.allclose(c[3], [0,0,0,1])
                or not np.allclose(c[:3,:3].T @ c[:3,:3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(c[:3,:3]), 1., atol=1e-6)):
            raise ValueError('invalid calibrated camera matrices')
        if points.dtype != torch.float32 or points.shape[-1] != 3:
            raise ValueError('projection points must be FP32 xyz')
        rotation = torch.as_tensor(c[:3,:3].copy(), device=points.device, dtype=torch.float32)
        origin = torch.as_tensor(c[:3,3].copy(), device=points.device, dtype=torch.float32)
        local = (points-origin) @ rotation
        depth = -local[...,2]
        safe_depth = depth.clamp_min(1e-6)
        u = float(k[0,0])*local[...,0]/safe_depth + float(k[0,2])
        v = float(k[1,2])-float(k[1,1])*local[...,1]/safe_depth
        h, w = camera.size
        valid = (depth > 1e-6) & (u >= 0) & (u <= w-1) & (v >= 0) & (v <= h-1)
        grid = torch.stack((2*(u+.5)/w-1, 2*(v+.5)/h-1), -1)
        # Avoid huge/NaN grid entries behind camera; validity still masks them.
        grid = torch.where(valid[...,None], grid, torch.full_like(grid, 2.))
        return grid, valid

    @staticmethod
    def sample_plane(image, grid):
        """Zero-padded bilinear align_corners=False sampling on FP32 MPS.

        Equivalent to grid_sample, using gather because the pinned Torch lacks
        MPS grid_sampler_2d_backward. All learned arithmetic/backward stays on
        the original device; only known camera matrices use CPU NumPy.
        """
        b,c,h,w = image.shape
        oh,ow = grid.shape[1:3]
        x = ((grid[...,0]+1)*w-1)/2
        y = ((grid[...,1]+1)*h-1)/2
        x0,y0 = x.floor(),y.floor()
        wx,wy = x-x0,y-y0
        flat = image.reshape(b,c,h*w)
        sampled = image.new_zeros((b,c,oh*ow))
        for dx,dy,weight in ((0,0,(1-wx)*(1-wy)),(1,0,wx*(1-wy)),(0,1,(1-wx)*wy),(1,1,wx*wy)):
            ix,iy = x0+dx,y0+dy
            valid = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
            index = (iy.clamp(0,h-1).long()*w+ix.clamp(0,w-1).long()).reshape(b,1,-1).expand(-1,c,-1)
            sampled = sampled+flat.gather(2,index)*(weight*valid).reshape(b,1,-1)
        return sampled.reshape(b,c,oh,ow)

    def forward(self, features: ActFeaturePacket, calibration) -> BEVOutput:
        if type(features) is not ActFeaturePacket:
            raise TypeError('BEV requires safe spatial feature packet')
        features = replace(features)  # revalidate mutable tensor/mapping payloads
        validate_calibration(features, calibration)
        first = features.spatial[CAMERA_VIEWS[0]]['layer1']
        if first.dtype != torch.float32 or self.scale_logits.dtype != torch.float32:
            raise ValueError('BEV requires FP32')
        b, device = first.shape[0], first.device
        world = self.world_grid(device=device)
        grids, masks = zip(*(self.project_points(world, calibration.cameras[view]) for view in CAMERA_VIEWS))
        view_mask = torch.stack(masks)[None].expand(b, -1, -1, -1)
        count = view_mask.sum(1, keepdim=True).clamp_min(1).to(torch.float32)
        fused = torch.zeros((b,128,*self.spec.shape), device=device, dtype=torch.float32)
        weights = self.scale_logits.softmax(0)
        for layer_index, layer in enumerate(LAYERS):
            scale = torch.zeros_like(fused)
            for i, view in enumerate(CAMERA_VIEWS):
                image = self.projections[layer](features.spatial[view][layer])
                # Frozen ResNet odd kernels/padding keep receptive center0 at
                # input pixel0; strides4/8/32 are not a resize with half-cell
                # centers. Convert original pixel coordinates to that lattice.
                stride = 320 // LAYERS[layer][1]
                grid = grids[i] + (stride-1)/320.
                projected = self.sample_plane(image,grid[None].expand(b,-1,-1,-1))
                scale = scale + projected * view_mask[:,i:i+1]
            fused = fused + weights[layer_index]*scale/count
        return BEVOutput(fused, view_mask.any(1,keepdim=True), view_mask, features.timestamp, self.spec)
