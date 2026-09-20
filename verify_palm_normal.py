"""Check whether the TRAINING-side palm_normal convention matches Aria MPS's, per hand.

Training (`data/dataclass.py`) computes, from SMPL-H forward kinematics:
    palm_normal = normalize( (index_MCP - wrist) x (pinky_MCP - wrist) )
whose sign flips between the left and right hand (mirror-imaged winding).

Aria MPS emits `palm_normal_device` with a hand-consistent "palm-facing" convention.

This script runs SMPL-H FK on the released model's estimate for a trajectory that
has `wrist_and_palm_poses.csv`, and reports, per hand, the mean dot product
between the two normals (frame-invariant, so compared in world frame):

    dot ~ +1  -> training convention already matches Aria for that hand
    dot ~ -1  -> that hand's training normal must be negated
    dot ~  0  / scattered -> the two 'normals' are not the same quantity
                              -> drop the normal channel, use pos_pair

It also reports the wrist-position and palm-pointing-vector agreement, which is
what a positions-only (pos_pair) encoding relies on.

Run in the egoallo container on the workstation (needs the trajectory's VRS + MPS):

    TRAJ=third_party/egoallo/egoallo_example_trajectories/EgoExo4d/takes/sfu_cooking022_9
    docker run --gpus all --rm \
      -v "$(pwd)/third_party/egoallo:/workspace/egoallo" \
      -v "$(pwd)/$TRAJ:/data/traj" \
      ego-humanoid-policy \
      python3 /workspace/egoallo/verify_palm_normal.py \
        --traj-root /data/traj \
        --npz /data/traj/egoallo_outputs/<a RELEASED-model output>.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from egoallo import fncsmpl
from egoallo.hand_detection_structs import CorrespondedAriaHandWristPoseDetections
from egoallo.inference_utils import InferenceInputTransforms, InferenceTrajectoryPaths


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj-root", type=Path, required=True)
    ap.add_argument("--npz", type=Path, required=True)
    ap.add_argument("--smplh-npz-path", type=Path,
                    default=Path("./data/smplh/neutral/model.npz"))
    args = ap.parse_args()

    npz = np.load(args.npz)
    start_index = int(npz["frame_nums"][0])
    traj_length = int(len(npz["frame_nums"]))
    print(f"npz window: start_index={start_index} traj_length={traj_length}")

    # --- SMPL-H FK on the released-model estimate (sample 0) ------------------
    def drop_sample(a: np.ndarray, want_ndim: int) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32)
        while a.ndim > want_ndim:  # strip leading num_samples / batch axes
            a = a[0]
        return a

    body_quats = torch.from_numpy(drop_sample(npz["body_quats"], 3))       # (T,21,4)
    lh = torch.from_numpy(drop_sample(npz["left_hand_quats"], 3))          # (T,15,4)
    rh = torch.from_numpy(drop_sample(npz["right_hand_quats"], 3))         # (T,15,4)
    T_world_root = torch.from_numpy(drop_sample(npz["Ts_world_root"], 2))  # (T,7)
    betas0 = torch.from_numpy(np.asarray(npz["betas"]).reshape(-1, 16)[0].astype(np.float32))

    model = fncsmpl.SmplhModel.load(args.smplh_npz_path)
    shaped = model.with_shape(betas0[None])
    posed = shaped.with_pose_decomposed(
        T_world_root=T_world_root, body_quats=body_quats,
        left_hand_quats=lh, right_hand_quats=rh,
    )
    J = posed.Ts_world_joint.numpy()  # (T, 51, 7)

    wrist_w = J[:, [19, 20], 4:7]                       # (T,2,3)
    idx_mcp_w = J[:, [21 + 0, 36 + 0], 4:7]             # index MCP  L,R
    pinky_mcp_w = J[:, [21 + 6, 36 + 6], 4:7]           # pinky MCP  L,R
    mcp_mean_w = np.stack([
        J[:, [21 + 0, 21 + 3, 21 + 6, 21 + 9], 4:7].mean(1),
        J[:, [36 + 0, 36 + 3, 36 + 6, 36 + 9], 4:7].mean(1),
    ], axis=1)                                          # (T,2,3)
    my_normal_w = _unit(np.cross(idx_mcp_w - wrist_w, pinky_mcp_w - wrist_w))  # (T,2,3)
    my_point_w = _unit(mcp_mean_w - wrist_w)            # hand-pointing vector

    # --- Aria MPS wrist/palm, world frame -----------------------------------
    traj_paths = InferenceTrajectoryPaths.find(args.traj_root)
    if traj_paths.wrist_and_palm_poses_csv is None:
        print("\nNo wrist_and_palm_poses.csv in this trajectory -> cannot verify "
              "against Aria. Pick a trajectory with MPS hand tracking, or use pos_pair.")
        return
    transforms = InferenceInputTransforms.load(
        traj_paths.vrs_file, traj_paths.slam_root_dir, fps=30
    ).to("cpu")
    sl = slice(start_index + 1, start_index + traj_length + 1)
    pose_ts = transforms.pose_timesteps[sl]
    Ts_world_device = transforms.Ts_world_device[sl].numpy(force=True)
    aria = CorrespondedAriaHandWristPoseDetections.load(
        traj_paths.wrist_and_palm_poses_csv, pose_ts, Ts_world_device=Ts_world_device
    )

    print("\n" + "=" * 68)
    for side, name, det in ((0, "LEFT ", aria.detections_left_concat),
                            (1, "RIGHT", aria.detections_right_concat)):
        if det is None:
            print(f"{name}: no Aria detections")
            continue
        idx = det.indices.numpy()
        m = (idx >= 0) & (idx < traj_length)
        idx = idx[m]
        if idx.size == 0:
            print(f"{name}: 0 in-range detections")
            continue
        aria_normal = _unit(det.palm_normal.numpy()[m])
        aria_wrist = det.wrist_position.numpy()[m]
        aria_palm = det.palm_position.numpy()[m]
        aria_point = _unit(aria_palm - aria_wrist)

        dot_normal = (my_normal_w[idx, side] * aria_normal).sum(-1)
        dot_point = (my_point_w[idx, side] * aria_point).sum(-1)
        d_wrist = np.linalg.norm(wrist_w[idx, side] - aria_wrist, axis=-1)
        d_palm = np.linalg.norm(mcp_mean_w[idx, side] - aria_palm, axis=-1)

        print(f"{name}  n={idx.size}")
        print(f"   palm_normal   : mean dot {dot_normal.mean():+.3f}  std {dot_normal.std():.3f}"
              f"  (|dot|>0.5 in {np.mean(np.abs(dot_normal) > 0.5) * 100:.0f}% of frames)")
        print(f"   pointing vec  : mean dot {dot_point.mean():+.3f}  std {dot_point.std():.3f}")
        print(f"   wrist pos err : mean {d_wrist.mean() * 100:.1f} cm")
        print(f"   palm  pos err : mean {d_palm.mean() * 100:.1f} cm  "
              f"(MCP-mean vs Aria palm_position)")
    print("=" * 68)
    print("verdict:")
    print("  normal dot ~ +1 both hands      -> palm_raw, no flip")
    print("  normal dot ~ +1 L / -1 R (or v) -> palm_raw, negate the -1 hand in dataclass.py")
    print("  normal dot ~ 0 / scattered      -> drop normal, use pos_pair")
    print("  (pointing-vec dot should be ~ +1 regardless -> pos_pair is safe)")


if __name__ == "__main__":
    main()
