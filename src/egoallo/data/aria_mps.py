from pathlib import Path
from typing import Literal

import numpy as np
from projectaria_tools.core import mps
from projectaria_tools.core.mps.utils import filter_points_from_confidence


def compute_gravity_alignment_rotation(slam_root_dir: Path) -> np.ndarray:
    """Compute a rotation that aligns this trajectory's own measured gravity
    direction to (0, 0, -1).

    Aria MPS's closed-loop trajectory is only guaranteed to be in "an
    arbitrary gravity aligned world coordinate frame" -- which axis ends up
    "up" isn't fixed across capture pipelines/hardware. The rest of this
    codebase (floor-height RANSAC below, and the network's canonicalization
    in egoallo/network.py) hard-assumes +Z is up. Applying this rotation to
    both the point cloud and every world-frame pose before anything else
    touches them makes that assumption hold regardless of the source
    convention, without needing to know it in advance -- for data that's
    already Z-up this comes out ~identity.
    """
    closed_loop_path = slam_root_dir / "closed_loop_trajectory.csv"
    if not closed_loop_path.exists():
        closed_loop_path = slam_root_dir / "aria_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(str(closed_loop_path))  # type: ignore

    a = np.mean([p.gravity_world for p in closed_loop_traj], axis=0)
    a = a / np.linalg.norm(a)
    b = np.array([0.0, 0.0, -1.0])

    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.dot(a, b)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180-degree flip; any axis perpendicular to `a` works.
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = axis - a * np.dot(axis, a)
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array(
        [[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]]
    )
    R = np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))
    angle_deg = np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))
    print(
        f"Gravity alignment: measured direction {a}, correcting by {angle_deg:.1f} degrees"
    )
    return R


def load_point_cloud_and_find_ground(
    points_path: Path,
    return_points: Literal["all", "filtered", "less_filtered"] = "less_filtered",
) -> tuple[np.ndarray, float]:
    """Load an Aria MPS point cloud and find the ground plane."""

    filtered_points_npz_cache_path = points_path.parent / "_cached_filtered_points.npz"
    less_filtered_points_npz_cache_path = (
        points_path.parent / "_cached_less_filtered_points.npz"
    )

    # Read world points as an Nx3 array.
    if (
        filtered_points_npz_cache_path.exists()
        and less_filtered_points_npz_cache_path.exists()
    ):
        if return_points == "all":
            points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore
        else:
            points_data = None

        print("Loading pre-filtered points")
        filtered_points_data = np.load(filtered_points_npz_cache_path)["points"]
        less_filtered_points_data = np.load(less_filtered_points_npz_cache_path)[
            "points"
        ]
    else:
        points_data = mps.read_global_point_cloud(str(points_path))  # type: ignore

        print("Loading + filtering points")
        assert points_path.exists()
        filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.0001,
            threshold_dep=0.005,
        )
        less_filtered_points_data = filter_points_from_confidence(
            points_data,
            threshold_invdep=0.001,
            threshold_dep=0.05,
        )
        filtered_points_data = np.array(
            [x.position_world for x in filtered_points_data]
        )  # type: ignore
        less_filtered_points_data = np.array(
            [x.position_world for x in less_filtered_points_data]
        )

        filtered_points_npz_cache_path.parent.mkdir(exist_ok=True, parents=True)

        np.savez_compressed(filtered_points_npz_cache_path, points=filtered_points_data)
        print("Cached filtered points to", filtered_points_npz_cache_path)
        np.savez_compressed(
            less_filtered_points_npz_cache_path, points=less_filtered_points_data
        )
        print("Cached less filtered points to", less_filtered_points_npz_cache_path)

    assert filtered_points_data.shape == (filtered_points_data.shape[0], 3)

    # Align to this trajectory's own measured gravity direction before doing
    # anything Z-height-based below.
    R_gravity = compute_gravity_alignment_rotation(points_path.parent)
    filtered_points_data = filtered_points_data @ R_gravity.T
    less_filtered_points_data = less_filtered_points_data @ R_gravity.T

    # RANSAC floor plane.
    # We consider points in the lowest 10% of the point cloud.
    filtered_zs = filtered_points_data[:, 2]
    zs = filtered_zs

    # Median-based outlier dropping, this doesn't work very well.
    # d = np.abs(zs - np.median(zs))
    # mdev = np.median(d)
    # zs = zs[d / mdev < 2.0]

    done = False
    best_z = 0.0
    while not done:
        # Slightly silly outlier dropping that just... works better.
        zs = np.sort(zs)[len(zs) // 10_000 : -len(zs) // 10_000]

        # Get bottom 10% or 15%.
        alpha = 0.1 if filtered_points_data.shape[0] < 10_000 else 0.15
        min_z = np.min(zs)
        max_z = np.max(zs)

        zs = zs[zs <= min_z + (max_z - min_z) * alpha]

        best_inliers = 0
        best_z = 0.0
        for i in range(10_000):
            z = np.random.choice(zs)
            inliers_bool = np.abs(zs - z) < 0.01
            inliers = np.sum(inliers_bool)
            if inliers > best_inliers:
                best_z = z
                best_inliers = inliers

        looser_inliers = np.sum(np.abs(filtered_zs - best_z) <= 0.075)
        if looser_inliers <= 3:
            # If we found a really small group... seems like noise. Let's remove the inlier points and re-compute.
            filtered_zs = filtered_zs[np.abs(filtered_zs - best_z) >= 0.01]
            zs = filtered_zs
        else:
            done = True

    # Re-fit plane to inliers.
    floor_z = float(np.median(zs[np.abs(zs - best_z) < 0.01]))
    if return_points == "filtered":
        return filtered_points_data, floor_z
    elif return_points == "less_filtered":
        return less_filtered_points_data, floor_z
    else:
        assert points_data is not None
        return (
            np.array([x.position_world for x in points_data]) @ R_gravity.T,
            floor_z,
        )
