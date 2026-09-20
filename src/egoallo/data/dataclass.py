from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from jaxtyping import Bool, Float
from torch import Tensor

from .. import fncsmpl, fncsmpl_extensions
from .. import transforms as tf
from ..tensor_dataclass import TensorDataclass


class EgoTrainingData(TensorDataclass):
    """Dictionary of tensors we use for EgoAllo training."""

    T_world_root: Float[Tensor, "*#batch timesteps 7"]
    """Transformation from the world frame to the root frame at each timestep."""

    contacts: Float[Tensor, "*#batch timesteps 21"]
    """Contact boolean for each joint."""

    betas: Float[Tensor, "*#batch 1 16"]
    """Body shape parameters."""

    # Excluded because not needed.
    # joints_wrt_world: Float[Tensor, "*#batch timesteps 21 3"]
    # """Joint positions relative to the world frame."""
    @property
    def joints_wrt_world(self) -> Tensor:
        return tf.SE3(self.T_world_cpf[..., None, :]) @ self.joints_wrt_cpf

    body_quats: Float[Tensor, "*#batch timesteps 21 4"]
    """Local orientations for each body joint."""

    T_cpf_tm1_cpf_t: Float[Tensor, "*#batch timesteps 7"]
    """Transformation to the next central pupil frame, from this timestep's
    central pupil frame."""

    T_world_cpf: Float[Tensor, "*#batch timesteps 7"]
    """Transformation from the world frame to the central pupil frame at each timestep."""

    height_from_floor: Float[Tensor, "*#batch timesteps 1"]
    """Distance from CPF to floor at each timestep."""

    joints_wrt_cpf: Float[Tensor, "*#batch timesteps 21 3"]
    """Joint positions relative to the central pupil frame."""

    mask: Bool[Tensor, "*#batch timesteps"]
    """Mask to support variable-length sequence."""

    hand_quats: Float[Tensor, "*#batch timesteps 30 4"] | None
    """Local orientations for each hand joint."""

    # --- First-party: wrist / palm ground truth for wrist-pose conditioning. ---
    # Index 0 is the left hand, index 1 the right. All in the CPF frame.
    wrist_pos_wrt_cpf: Float[Tensor, "*#batch timesteps 2 3"]
    """Left/right wrist joint (SMPL-H joints 19/20) position, CPF frame."""

    wrist_rot_wrt_cpf: Float[Tensor, "*#batch timesteps 2 3 3"]
    """Left/right wrist joint orientation (rotation matrix), CPF frame."""

    palm_pos_wrt_cpf: Float[Tensor, "*#batch timesteps 2 3"]
    """Left/right palm-centre position (mean of the index/middle/ring/pinky MCP
    joints of the posed hand), CPF frame."""

    palm_normal_wrt_cpf: Float[Tensor, "*#batch timesteps 2 3"]
    """Left/right palm normal (out of the palm), CPF frame. Geometric: the unit
    normal of the plane through the wrist and the index/pinky MCP joints, signed
    to point away from the back of the hand. Only used by the "palm_raw"
    encoding; "rot6d" training reads `wrist_rot_wrt_cpf` directly."""

    @staticmethod
    def load_from_npz(
        body_model: fncsmpl.SmplhModel,
        path: Path,
        include_hands: bool,
    ) -> EgoTrainingData:
        """Load a single trajectory from a (processed_30fps) npz file."""
        raw_fields = {
            k: torch.from_numpy(v.astype(np.float32) if v.dtype == np.float64 else v)
            for k, v in np.load(path).items()
            if v.dtype in (np.float32, np.float64)
        }

        timesteps = raw_fields["root_orient"].shape[0]
        assert raw_fields["root_orient"].shape == (timesteps, 3)
        assert raw_fields["pose_body"].shape == (timesteps, 63)
        assert raw_fields["pose_hand"].shape == (timesteps, 90)
        assert raw_fields["joints"].shape == (timesteps, 22, 3)

        T_world_root = torch.cat(
            [
                tf.SO3.exp(raw_fields["root_orient"]).wxyz,
                raw_fields["joints"][:, 0, :],
            ],
            dim=-1,
        )
        body_quats = tf.SO3.exp(raw_fields["pose_body"].reshape(timesteps, 21, 3)).wxyz
        hand_quats = tf.SO3.exp(raw_fields["pose_hand"].reshape(timesteps, 30, 3)).wxyz

        device = body_model.weights.device
        shaped = body_model.with_shape(raw_fields["betas"][0:1, :].to(device))

        # Batch the SMPL body model operations, this can be pretty memory-intensive...
        # Pose the finger joints too (needed for the wrist/palm conditioning GT);
        # body joints 0..20 are unaffected by the hand quats.
        posed = shaped.with_pose_decomposed(
            T_world_root=T_world_root.to(device),
            body_quats=body_quats.to(device),
            left_hand_quats=hand_quats[:, :15, :].to(device),
            right_hand_quats=hand_quats[:, 15:, :].to(device),
        )
        T_world_cpf = (
            tf.SE3(posed.Ts_world_joint[:, 14, :])  # T_world_head
            @ tf.SE3(fncsmpl_extensions.get_T_head_cpf(shaped))
        ).parameters()
        assert T_world_cpf.shape == (timesteps, 7)

        # --- Wrist / palm ground truth for wrist-pose conditioning (CPF frame). ---
        Ts_world_joint = posed.Ts_world_joint  # (T, 51, 7); 0..20 body, 21..35 L, 36..50 R
        T_cpf_world = tf.SE3(T_world_cpf[:, None, :]).inverse()  # SE3, batch (T, 1)
        # SMPL-H MANO local layout per hand: MCP (proximal) joints are local
        # indices 0 (index), 3 (middle), 6 (pinky), 9 (ring).
        left_mcp_idx = [21 + 0, 21 + 3, 21 + 6, 21 + 9]
        right_mcp_idx = [36 + 0, 36 + 3, 36 + 6, 36 + 9]

        wrist_pos_w = Ts_world_joint[:, [19, 20], 4:7]  # (T, 2, 3)
        wrist_quat_w = Ts_world_joint[:, [19, 20], 0:4]  # (T, 2, 4)
        mcp_pos_w = torch.stack(
            [
                Ts_world_joint[:, left_mcp_idx, 4:7].mean(dim=1),
                Ts_world_joint[:, right_mcp_idx, 4:7].mean(dim=1),
            ],
            dim=1,
        )  # (T, 2, 3)
        # Geometric palm normal: n = (index_MCP - wrist) x (pinky_MCP - wrist).
        # The winding is mirror-imaged between hands, so this points palmar for one
        # hand and dorsal for the other. Aria MPS's `palm_normal_device` uses a
        # hand-consistent palm-facing convention; verified (euler/verify_palm_normal.py)
        # that the RIGHT hand already matches Aria and the LEFT hand is sign-flipped,
        # so negate index 0 (left) to align both hands with Aria.
        idx_mcp_w = Ts_world_joint[:, [21 + 0, 36 + 0], 4:7]  # (T, 2, 3)
        pinky_mcp_w = Ts_world_joint[:, [21 + 6, 36 + 6], 4:7]  # (T, 2, 3)
        palm_normal_w = torch.linalg.cross(
            idx_mcp_w - wrist_pos_w, pinky_mcp_w - wrist_pos_w, dim=-1
        )
        palm_normal_w = palm_normal_w / (
            torch.linalg.norm(palm_normal_w, dim=-1, keepdim=True) + 1e-6
        )
        palm_normal_w = torch.stack(
            [-palm_normal_w[:, 0], palm_normal_w[:, 1]], dim=1
        )  # left (index 0) flipped to match Aria's palm-facing convention

        wrist_pos_cpf = T_cpf_world @ wrist_pos_w  # (T, 2, 3)
        palm_pos_cpf = T_cpf_world @ mcp_pos_w
        R_cpf_world = T_cpf_world.rotation()  # SO3, batch (T, 1)
        palm_normal_cpf = R_cpf_world @ palm_normal_w
        wrist_rot_cpf = (
            R_cpf_world @ tf.SO3(wrist_quat_w)
        ).as_matrix()  # (T, 2, 3, 3)

        # Construct the training data elements that we want to keep.
        return EgoTrainingData(
            T_world_root=T_world_root[1:].cpu(),
            contacts=raw_fields["contacts"][1:, 1:].cpu(),  # Root is no longer a joint.
            betas=raw_fields["betas"][0:1, :].cpu(),
            # joints_wrt_world=raw_fields["joints"][
            #     1:, 1:
            # ].cpu(),  # Root is no longer a joint.
            body_quats=body_quats[1:].cpu(),
            # CPF frame stuff.
            T_world_cpf=T_world_cpf[1:].cpu(),
            # Get translational z coordinate from wxyz_xyz.
            height_from_floor=T_world_cpf[1:, 6:7].cpu(),
            T_cpf_tm1_cpf_t=(
                tf.SE3(T_world_cpf[:-1, :]).inverse() @ tf.SE3(T_world_cpf[1:, :])
            )
            .parameters()
            .cpu(),
            joints_wrt_cpf=(
                # unsqueeze so both shapes are (timesteps, joints, dim)
                tf.SE3(T_world_cpf[1:, None, :]).inverse()
                @ raw_fields["joints"][1:, 1:, :].to(T_world_cpf.device)
            ).cpu(),
            mask=torch.ones((timesteps - 1,), dtype=torch.bool),
            hand_quats=hand_quats[1:].cpu() if include_hands else None,
            wrist_pos_wrt_cpf=wrist_pos_cpf[1:].float().cpu(),
            wrist_rot_wrt_cpf=wrist_rot_cpf[1:].float().cpu(),
            palm_pos_wrt_cpf=palm_pos_cpf[1:].float().cpu(),
            palm_normal_wrt_cpf=palm_normal_cpf[1:].float().cpu(),
        )


def collate_dataclass[T](batch: list[T]) -> T:
    """Collate function that works for dataclasses."""
    keys = vars(batch[0]).keys()
    return type(batch[0])(
        **{k: torch.stack([getattr(b, k) for b in batch]) for k in keys}
    )
