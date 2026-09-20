"""Wrist / palm *conditioning* signal for the EgoAllo denoiser.

This is the first-party addition that lets the network consume observed wrist
poses directly, instead of only through the test-time guidance optimizer
(`guidance_optimizer_jax.py`). See `plans/based-on-the-egoallo-sequential-wave.md`.

Per hand, per frame, in the **CPF frame**, we build a 10-dim block:

    encoding == "palm_raw" (default): [ valid(1), wrist_pos(3), dir(palm-wrist)(3), palm_normal(3) ]
    encoding == "rot6d":              [ valid(1), wrist_pos(3), wrist_rot6d(6) ]

Both hands -> 20 dims, concatenated onto the conditioning vector *before* the
Fourier encoding in `EgoDenoiserConfig.make_cond`.

`palm_raw` is the verified-consistent encoding. All three quantities are aligned
between training (SMPL-H FK) and inference (Aria MPS):
  - `wrist_pos`               : CPF position, agrees within ~2 cm (inside the noise aug);
  - `dir(palm_pos - wrist_pos)`: unit hand-pointing vector, dot ~0.99 vs Aria;
  - `palm_normal`             : unit, dot ~0.98 vs Aria *after* flipping the LEFT hand
                                 in `data/dataclass.py` (mirror-imaged MCP winding).
See `verify_palm_normal.py` for the measurement.

`rot6d` is kept for experiments but is NOT convention-safe: training used the raw
SMPL-H wrist joint frame while inference builds a palm Gram-Schmidt frame -- do
not use it without re-verifying.
"""

from __future__ import annotations

from typing import Literal

import torch
from jaxtyping import Bool, Float
from torch import Tensor

from .transforms import SE3, SO3

WristCondEncoding = Literal["palm_raw", "rot6d"]

# 10 dims per hand, 2 hands.
HAND_COND_DIM_PER_HAND = 10
HAND_COND_DIM = HAND_COND_DIM_PER_HAND * 2

# Root-less 21-joint body layout: left wrist == 19, right wrist == 20.
LEFT_WRIST_INDEX = 19
RIGHT_WRIST_INDEX = 20

_EPS = 1e-6


def rot6d_from_matrix(R: Float[Tensor, "*batch 3 3"]) -> Float[Tensor, "*batch 6"]:
    """Zhou et al. 6-D rotation representation: the first two columns of R,
    stacked as [col0 (3), col1 (3)]."""
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)


def wrist_rotmat_from_palm(
    wrist_pos: Float[Tensor, "*batch 3"],
    palm_pos: Float[Tensor, "*batch 3"],
    palm_normal: Float[Tensor, "*batch 3"],
    left0_right1: Float[Tensor, "*batch"] | float,
) -> Float[Tensor, "*batch 3 3"]:
    """Estimate a wrist-frame rotation from palm geometry.

    Torch port of the orientation block of
    `guidance_optimizer_jax.py::wrist_pose_cost` (lines ~729-748):

        palm_forward = normalize(palm_pos - wrist_pos)
        palm_normal  = normalize(palm_normal)
        palm_forward = palm_forward * (+1 left / -1 right)
        palm_forward = palm_forward - (palm_forward . palm_normal) palm_normal   # Gram-Schmidt
        R = [ palm_forward | -palm_normal | palm_normal x palm_forward ]         # columns

    All inputs share one frame (world at inference-from-Aria, CPF for us). The
    output rotation is expressed in that same frame.
    """
    if not isinstance(left0_right1, Tensor):
        left0_right1 = wrist_pos.new_full(wrist_pos.shape[:-1], float(left0_right1))
    sign = torch.where(left0_right1 > 0.5, -1.0, 1.0)[..., None]

    fwd = palm_pos - wrist_pos
    fwd = fwd / (torch.linalg.norm(fwd, dim=-1, keepdim=True) + _EPS)
    n = palm_normal / (torch.linalg.norm(palm_normal, dim=-1, keepdim=True) + _EPS)
    fwd = fwd * sign
    fwd = fwd - (fwd * n).sum(dim=-1, keepdim=True) * n
    fwd = fwd / (torch.linalg.norm(fwd, dim=-1, keepdim=True) + _EPS)
    col2 = torch.linalg.cross(n, fwd, dim=-1)
    return torch.stack([fwd, -n, col2], dim=-1)


def fov_visibility_mask(
    pos_cpf: Float[Tensor, "*batch 3"],
    half_fov_deg: float = 55.0,
    pitch_deg: float = 0.0,
    max_range_m: float = 1.25,
) -> Bool[Tensor, "*batch"]:
    """Coarse "is this point in the Aria RGB frustum" test, in the CPF frame.

    CPF +Z is forward (see `EgoDenoiserConfig.make_cond`, `forward_cpf = [0,0,1]`).
    The real Aria RGB camera sits slightly below/ahead of the CPF and is pitched
    down; `pitch_deg` rotates the test cone about the CPF +X axis to approximate
    that. Keep the cone generous — this only shapes which hands the model learns
    to treat as "observed", it is not a real projection.
    """
    if pitch_deg != 0.0:
        c = torch.cos(pos_cpf.new_tensor(pitch_deg * torch.pi / 180.0))
        s = torch.sin(pos_cpf.new_tensor(pitch_deg * torch.pi / 180.0))
        x, y, z = pos_cpf.unbind(-1)
        pos_cpf = torch.stack([x, c * y - s * z, s * y + c * z], dim=-1)

    dist = torch.linalg.norm(pos_cpf, dim=-1)
    cos_half_fov = torch.cos(pos_cpf.new_tensor(half_fov_deg * torch.pi / 180.0))
    forward_cos = pos_cpf[..., 2] / (dist + _EPS)
    return (pos_cpf[..., 2] > 0.0) & (forward_cos > cos_half_fov) & (dist < max_range_m)


def assemble_hand_cond(
    valid: Bool[Tensor, "*batch 2"],
    wrist_pos_cpf: Float[Tensor, "*batch 2 3"],
    wrist_rot_cpf: Float[Tensor, "*batch 2 3 3"] | None = None,
    palm_pos_cpf: Float[Tensor, "*batch 2 3"] | None = None,
    palm_normal_cpf: Float[Tensor, "*batch 2 3"] | None = None,
    encoding: WristCondEncoding = "palm_raw",
) -> Float[Tensor, "*batch 20"]:
    """Pack the per-hand 10-dim blocks into the 20-dim conditioning vector.

    `valid[..., i] == False` zeroes that hand's entire 10-dim block. Index 0 is
    the left hand, index 1 the right.

    - "rot6d" needs `wrist_rot_cpf` (or `palm_pos_cpf` + `palm_normal_cpf`, from
      which the rotation is derived via `wrist_rotmat_from_palm`).
    - "palm_raw" needs `palm_pos_cpf` + `palm_normal_cpf`.
    """
    *batch, two, _ = wrist_pos_cpf.shape
    assert two == 2
    device = wrist_pos_cpf.device
    left0_right1 = wrist_pos_cpf.new_tensor([0.0, 1.0]).expand(*batch, 2)

    if encoding == "rot6d":
        if wrist_rot_cpf is None:
            assert palm_pos_cpf is not None and palm_normal_cpf is not None
            wrist_rot_cpf = wrist_rotmat_from_palm(
                wrist_pos_cpf, palm_pos_cpf, palm_normal_cpf, left0_right1
            )
        per_hand = torch.cat(
            [wrist_pos_cpf, rot6d_from_matrix(wrist_rot_cpf)], dim=-1
        )  # (*batch, 2, 9)
    elif encoding == "palm_raw":
        assert palm_pos_cpf is not None and palm_normal_cpf is not None
        palm_vec = palm_pos_cpf - wrist_pos_cpf
        per_hand = torch.cat(
            [
                wrist_pos_cpf,
                # unit hand-pointing vector: length (MCP-mean vs Aria palm-centre)
                # differs ~3 cm between train/infer, direction agrees to dot ~0.99.
                palm_vec / (torch.linalg.norm(palm_vec, dim=-1, keepdim=True) + _EPS),
                palm_normal_cpf
                / (torch.linalg.norm(palm_normal_cpf, dim=-1, keepdim=True) + _EPS),
            ],
            dim=-1,
        )  # (*batch, 2, 9)
    else:
        raise ValueError(f"unknown encoding {encoding!r}")

    valid_f = valid.to(per_hand.dtype)[..., None]
    per_hand = torch.cat([valid_f, per_hand * valid_f], dim=-1)  # (*batch, 2, 10)
    return per_hand.reshape(*batch, HAND_COND_DIM).to(device)


def hand_cond_from_detections(
    Ts_world_cpf: Float[Tensor, "time 7"],
    aria_detections=None,  # CorrespondedAriaHandWristPoseDetections | None
    hamer_detections=None,  # CorrespondedHamerDetections | None
    encoding: WristCondEncoding = "palm_raw",
    frame_offset: Float[Tensor, "3 3"] | None = None,
) -> Float[Tensor, "time 20"]:
    """Build the inference-time hand-conditioning tensor.

    Aria MPS wrist/palm poses (world frame) are the primary source; HaMeR wrist +
    MANO-MCP palm (camera frame, via `T_cpf_cam`) fills frames where Aria has no
    detection. `frame_offset`, if given, is a fixed rotation applied to every
    wrist frame to absorb a systematic convention gap between the SMPL-H wrist
    frame (training) and the Aria/HaMeR frame (inference) -- calibrate it once on
    a trajectory that has both signals.
    """
    time = Ts_world_cpf.shape[0]
    device = Ts_world_cpf.device
    valid = torch.zeros((time, 2), dtype=torch.bool, device=device)
    wrist_pos = torch.zeros((time, 2, 3), device=device)
    palm_pos = torch.zeros((time, 2, 3), device=device)
    palm_normal = wrist_pos.new_zeros((time, 2, 3))
    palm_normal[..., 2] = 1.0

    T_cpf_world = SE3(Ts_world_cpf).inverse()

    def _fill_from_aria(side_idx: int, det) -> None:
        if det is None:
            return
        idx = det.indices.to(device).long()
        in_range = (idx >= 0) & (idx < time)
        idx = idx[in_range]
        if idx.numel() == 0:
            return
        w = (SE3(T_cpf_world.wxyz_xyz[idx]) @ det.wrist_position.to(device)[in_range])
        p = (SE3(T_cpf_world.wxyz_xyz[idx]) @ det.palm_position.to(device)[in_range])
        n = SO3(T_cpf_world.rotation().wxyz[idx]) @ det.palm_normal.to(device)[in_range]
        wrist_pos[idx, side_idx] = w
        palm_pos[idx, side_idx] = p
        palm_normal[idx, side_idx] = n
        valid[idx, side_idx] = True

    if aria_detections is not None:
        _fill_from_aria(0, getattr(aria_detections, "detections_left_concat", None))
        _fill_from_aria(1, getattr(aria_detections, "detections_right_concat", None))

    if hamer_detections is not None:
        _fill_from_hamer(hamer_detections, time, device, valid, wrist_pos, palm_pos, palm_normal)

    wrist_rot = None
    if encoding == "rot6d":
        left0_right1 = wrist_pos.new_tensor([0.0, 1.0]).expand(time, 2)
        wrist_rot = wrist_rotmat_from_palm(wrist_pos, palm_pos, palm_normal, left0_right1)
        if frame_offset is not None:
            wrist_rot = wrist_rot @ frame_offset.to(device)

    return assemble_hand_cond(
        valid=valid,
        wrist_pos_cpf=wrist_pos,
        wrist_rot_cpf=wrist_rot,
        palm_pos_cpf=palm_pos,
        palm_normal_cpf=palm_normal,
        encoding=encoding,
    )


# OpenPose hand keypoint indices (HaMeR `keypoints_3d` order): 0 wrist,
# 5 index-MCP, 9 middle-MCP, 13 ring-MCP, 17 pinky-MCP.
_OPENPOSE_WRIST = 0
_OPENPOSE_MCPS = (5, 9, 13, 17)


def _fill_from_hamer(hamer_detections, time, device, valid, wrist_pos, palm_pos, palm_normal) -> None:
    """Fill still-missing frames from HaMeR (camera frame -> CPF via T_cpf_cam)."""
    T_cpf_cam = SE3(torch.as_tensor(hamer_detections.T_cpf_cam, device=device).float())
    for side_idx, concat in (
        (0, getattr(hamer_detections, "detections_left_concat", None)),
        (1, getattr(hamer_detections, "detections_right_concat", None)),
    ):
        if concat is None:
            continue
        kp = torch.as_tensor(concat.keypoints_3d, device=device).float()  # (n, 21, 3)
        idx = torch.as_tensor(concat.indices, device=device).long()
        in_range = (idx >= 0) & (idx < time)
        idx = idx[in_range]
        kp = kp[in_range]
        if idx.numel() == 0:
            continue
        take = ~valid[idx, side_idx]  # do not overwrite an Aria detection
        idx, kp = idx[take], kp[take]
        if idx.numel() == 0:
            continue
        w_cam = kp[:, _OPENPOSE_WRIST, :]
        mcp_cam = kp[:, _OPENPOSE_MCPS, :].mean(dim=1)
        n_cam = torch.linalg.cross(
            kp[:, _OPENPOSE_MCPS[0], :] - w_cam,
            kp[:, _OPENPOSE_MCPS[-1], :] - w_cam,
            dim=-1,
        )
        wrist_pos[idx, side_idx] = T_cpf_cam @ w_cam
        palm_pos[idx, side_idx] = T_cpf_cam @ mcp_cam
        palm_normal[idx, side_idx] = T_cpf_cam.rotation() @ n_cam
        valid[idx, side_idx] = True
