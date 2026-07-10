"""
congestionSimulator.py — 3D UUDM Congestion Simulator

Derives empirical SUI thresholds for the Congestion Index (Low / Medium / High)
by incrementally populating a 3D voxel grid representing an LA_SpatialUnit
(ISO 19152-1:2022) with random utility corridors and using Weighted A*
pathfinding to detect when routing a new pipe becomes significantly deviated.

Key Design Decisions
--------------------
- Voxel grid representation: standard for 3D underground spatial analysis
  (Atazadeh et al., 2017; Guo et al., 2020).
- WA* with ε=1.5 for computational tractability over 200 iterations × 30 seeds
  (Pohl, 1970; Likhachev et al., 2003). PDR values are upper bounds (C ≤ ε·C*).
- PDR > 1.05 (Low→Medium): local optimality criterion, Prato & Bekhor (2006).
- PDR > 1.20 (Medium→High): urban network circuity threshold, Barthelemy (2022).
- Orthogonal routing + turn penalty: models Manhattan utility installation practice.
- Surface cover 1.5 m: Singapore LTA Code of Practice for Works on Public Streets (2021).

Reproducibility
---------------
All stochastic components use RANDOM_SEED (default 42). For the N=30
multi-run analysis, seeds 0–29 are used sequentially.

References
----------
- ISO 19152-1:2022; ISO 19152-2:2025
- Atazadeh et al. (2017), ISPRS Int. J. Geo-Inf., 6(12), 393
- Guo et al. (2020), Automation in Construction, 119, 103309
- He et al. (2021), TUST, 107, 103660
- Pohl (1970), Artificial Intelligence, 1(3), 193–204
- Likhachev et al. (2003), NIPS
- Prato & Bekhor (2006), Transportation Research B, 40(5), 395–421
- Barthelemy (2022), Nature Reviews Physics, 4, 624–642
- Ewing & Cervero (2010), JAPA, 76(3), 265–294
- Singapore LTA CoPWPS (2021), §4.3
"""

import argparse
import heapq
import logging
import random
import sys

import numpy as np
import trimesh

try:
    from scipy.ndimage import binary_dilation
except ImportError:
    print("Please install scipy: pip install scipy")
    sys.exit(1)

try:
    from SyntheticSGUtilities import generate_marina_bay_synthetic_data
except ImportError:
    print("Error: Could not find SyntheticSGUtilities.py in the current directory.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("congestionSimulator")

# Default random seed for reproducibility (report in paper's methodology)
DEFAULT_SEED = 42

# PDR thresholds (see module docstring for full rationale and citations)
PDR_LOW_MEDIUM  = 1.05   # Prato & Bekhor (2006); HK DSD (2020)
PDR_MEDIUM_HIGH = 1.20   # Barthelemy (2022); Ewing & Cervero (2010)


# ============================================================
# LADM / UUDM Class Definitions
# ============================================================

class LA_SpatialUnit:
    """
    3D cadastral parcel bounding volume.
    LADM ISO 19152-1:2022: LA_SpatialUnit
    """
    def __init__(self, bounds_min, bounds_max, resolution=0.5):
        self.bounds_min = np.array(bounds_min, dtype=float)
        self.bounds_max = np.array(bounds_max, dtype=float)
        self.extents    = self.bounds_max - self.bounds_min
        self.volume     = float(np.prod(self.extents))
        self.resolution = resolution

        self.shape = np.ceil(self.extents / self.resolution).astype(int)
        # Boolean voxel grid: True = legally occupied
        self.grid  = np.zeros(self.shape, dtype=bool)


class PhysicalUtilityNetwork:
    """
    Physical pipe / cable geometry segment.
    UUDM: PhysicalUtilityNetwork (Yan et al., 2021)
    """
    def __init__(self, p1, p2, radius):
        self.p1     = np.array(p1, dtype=float)
        self.p2     = np.array(p2, dtype=float)
        self.radius = float(radius)


class LA_LegalSpaceUtilityNetwork:
    """
    Statutory clearance zone around a physical utility.
    ISO 19152-2:2025: LA_LegalSpaceUtilityNetworkElement
    (formerly LA_LegalSpaceUtilityNetwork in ISO 19152:2012)
    """
    def __init__(self, physical_utility: PhysicalUtilityNetwork, clearance: float = 0.5):
        self.physical     = physical_utility
        self.clearance    = clearance
        self.legal_radius = physical_utility.radius + clearance


# ============================================================
# Voxelization Engine
# ============================================================

def voxelize_legal_space(
    spatial_unit: LA_SpatialUnit,
    legal_space:  LA_LegalSpaceUtilityNetwork,
) -> None:
    """
    Mark voxels in the LA_SpatialUnit grid that fall within the legal space
    cylinder (LA_LegalSpaceUtilityNetworkElement, ISO 19152-2:2025).

    Uses point-to-line-segment distance for accurate cylindrical masking.
    """
    p1     = legal_space.physical.p1
    p2     = legal_space.physical.p2
    radius = legal_space.legal_radius
    res    = spatial_unit.resolution
    min_b  = spatial_unit.bounds_min

    min_pt = np.minimum(p1, p2) - radius
    max_pt = np.maximum(p1, p2) + radius

    min_idx = np.floor((min_pt - min_b) / res).astype(int)
    max_idx = np.ceil( (max_pt - min_b) / res).astype(int)

    min_idx = np.maximum(min_idx, 0)
    max_idx = np.minimum(max_idx, np.array(spatial_unit.shape) - 1)

    if np.any(min_idx > max_idx):
        return  # Entirely outside the parcel

    x  = np.arange(min_idx[0], max_idx[0] + 1)
    y  = np.arange(min_idx[1], max_idx[1] + 1)
    z  = np.arange(min_idx[2], max_idx[2] + 1)

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1) * res + min_b + (res / 2.0)

    v  = p1
    w  = p2
    l2 = np.sum((w - v) ** 2)

    if l2 == 0.0:
        dist = np.linalg.norm(points - v, axis=-1)
    else:
        t          = np.clip(np.sum((points - v) * (w - v), axis=-1) / l2, 0.0, 1.0)
        projection = v + t[..., np.newaxis] * (w - v)
        dist       = np.linalg.norm(points - projection, axis=-1)

    mask = dist <= radius
    spatial_unit.grid[
        min_idx[0] : max_idx[0] + 1,
        min_idx[1] : max_idx[1] + 1,
        min_idx[2] : max_idx[2] + 1,
    ][mask] = True


def calculate_sui(spatial_unit: LA_SpatialUnit) -> float:
    """
    Spatial Utilization Index (SUI) as a percentage [0, 100].

    SUI = (legally occupied voxels / total voxels) × 100

    The boolean grid natively handles overlapping statutory clearances without
    double-counting (Guo et al., 2020).

    Note: SpatialClaimCalc.py returns SUI as a ratio [0, 1].
    Conversion: ratio = SUI_percent / 100.
    """
    occupied = np.sum(spatial_unit.grid)
    total    = spatial_unit.grid.size
    return float(occupied / total) * 100.0


# ============================================================
# 3D A* Pathfinding
# ============================================================

def check_routability(
    spatial_unit: LA_SpatialUnit,
    new_pipe_legal_radius: float = 0.5,
):
    """
    Attempt to route a new utility corridor along the X-axis using WA*.

    Parameters
    ----------
    spatial_unit : LA_SpatialUnit
    new_pipe_legal_radius : float
        Legal clearance radius of the new utility being routed (m).

    Returns
    -------
    tuple : (status, path_length, direct_distance)
        status: 'STRAIGHT' | 'DEVIATED' | 'FAILED'
    """
    grid   = spatial_unit.grid
    shape  = spatial_unit.shape
    res    = spatial_unit.resolution

    # Dilate existing obstacles by the new pipe's legal radius so A* treats
    # the pipe centre as a point agent (configuration-space approach)
    dilation_voxels = int(np.ceil(new_pipe_legal_radius / res))
    z_g, y_g, x_g  = np.ogrid[
        -dilation_voxels : dilation_voxels + 1,
        -dilation_voxels : dilation_voxels + 1,
        -dilation_voxels : dilation_voxels + 1,
    ]
    struct       = x_g**2 + y_g**2 + z_g**2 <= dilation_voxels**2
    dilated_grid = binary_dilation(grid, structure=struct)

    # Minimum surface cover: 1.5 m reserved for traffic loads
    # (Singapore LTA CoPWPS, 2021, §4.3)
    cover_voxels = int(np.ceil(1.5 / res))
    dilated_grid[:, :, -cover_voxels:] = True

    target_x = shape[0] - 1

    # Fast check: any column clear across the full X-extent?
    if np.any(~np.any(dilated_grid, axis=0)):
        return "STRAIGHT", target_x, target_x

    valid_starts = np.argwhere(~dilated_grid[0, :, :])
    if len(valid_starts) == 0:
        return "FAILED", 0, 0

    open_set = []
    g_score  = {}

    def heuristic(pos):
        # Weighted A* (WA*) with ε=1.5 — Pohl (1970, AI, 1(3), 193–204);
        # Likhachev et al. (2003, NIPS, ARA*).
        # Guarantee: C ≤ ε·C* → PDR values are conservative upper bounds.
        # For a planning-conservative congestion threshold tool this is appropriate:
        # the simulator flags congestion at the same or slightly lower SUI than
        # optimal routing would, ensuring thresholds never under-classify congestion.
        EPSILON = 1.5
        return EPSILON * (target_x - pos[0])

    for start_yz in valid_starts:
        sp  = (0, int(start_yz[0]), int(start_yz[1]))
        idr = (1, 0, 0)
        g_score[(sp, idr)] = 0
        heapq.heappush(open_set, (heuristic(sp), sp, idr))

    visited = set()

    # Orthogonal directions only — Manhattan routing model for rigid utilities
    # (Yan et al., 2019; Xu et al., 2009)
    directions = [
        (1,  0,  0, 1.0),
        (0,  1,  0, 1.0),
        (0, -1,  0, 1.0),
        (0,  0,  1, 1.0),
        (0,  0, -1, 1.0),
    ]
    TURN_PENALTY = 5.0

    while open_set:
        _, cur, cur_dir = heapq.heappop(open_set)

        if cur[0] == target_x:
            return "DEVIATED", g_score[(cur, cur_dir)], target_x

        state = (cur, cur_dir)
        if state in visited:
            continue
        visited.add(state)

        for dx, dy, dz, cost in directions:
            nx, ny, nz = cur[0] + dx, cur[1] + dy, cur[2] + dz
            new_dir    = (dx, dy, dz)
            move_cost  = cost + (TURN_PENALTY if new_dir != cur_dir else 0.0)

            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if not dilated_grid[nx, ny, nz]:
                    tg        = g_score[state] + move_cost
                    nb_state  = ((nx, ny, nz), new_dir)
                    if nb_state not in g_score or tg < g_score[nb_state]:
                        g_score[nb_state] = tg
                        heapq.heappush(open_set, (tg + heuristic((nx, ny, nz)), (nx, ny, nz), new_dir))

    return "FAILED", 0, 0


# ============================================================
# Single simulation run (parameterised for multi-run analysis)
# ============================================================

_BOUNDS_MIN = [-5.0, -25.0, -25.0]
_BOUNDS_MAX  = [45.0,  25.0,   0.0]


def _single_run(seed: int, resolution: float = 0.5, max_iterations: int = 200) -> dict:
    """
    Run one complete incremental congestion simulation with a fixed seed.

    Parameters
    ----------
    seed : int
        Random seed for this run. Set in paper as: seed ∈ {0, 1, …, 29}.
    resolution : float
        Voxel resolution in metres (default 0.5 for speed).
    max_iterations : int
        Maximum number of random pipes to add before stopping.

    Returns
    -------
    dict with keys 'LOW_MAX' and 'MEDIUM_MAX' (SUI percentages, or None).
    """
    random.seed(seed)
    np.random.seed(seed)

    spatial_unit = LA_SpatialUnit(_BOUNDS_MIN, _BOUNDS_MAX, resolution=resolution)

    # Seed with the base Marina Bay network
    utils_3d = generate_marina_bay_synthetic_data()
    for u in utils_3d:
        rec  = u["record"]
        path = rec["path_2d"]
        r    = rec["radius"]
        z1   = rec["depth_start"]
        z2   = rec["depth_end"]
        for i in range(len(path) - 1):
            p1 = np.array([path[i][0],   path[i][1],   z1])
            p2 = np.array([path[i+1][0], path[i+1][1], z2])
            voxelize_legal_space(spatial_unit, LA_LegalSpaceUtilityNetwork(
                PhysicalUtilityNetwork(p1, p2, r), clearance=0.5
            ))

    thresholds     = {"LOW_MAX": None, "MEDIUM_MAX": None}
    status, _, _   = check_routability(spatial_unit)
    current_status = status
    iteration      = 0

    while current_status != "FAILED" and iteration < max_iterations:
        iteration += 1

        # Random pipe generation (50/50 X- or Y-aligned)
        if random.random() < 0.5:
            y_s = random.uniform(_BOUNDS_MIN[1], _BOUNDS_MAX[1])
            z_s = random.uniform(_BOUNDS_MIN[2], _BOUNDS_MAX[2] - 1.5)
            p1  = [_BOUNDS_MIN[0] - 5, y_s,                    z_s]
            p2  = [_BOUNDS_MAX[0] + 5, y_s + random.uniform(-2, 2), z_s + random.uniform(-2, 2)]
        else:
            x_s = random.uniform(_BOUNDS_MIN[0], _BOUNDS_MAX[0])
            z_s = random.uniform(_BOUNDS_MIN[2], _BOUNDS_MAX[2] - 1.5)
            p1  = [x_s,                    _BOUNDS_MIN[1] - 5, z_s]
            p2  = [x_s + random.uniform(-2, 2), _BOUNDS_MAX[1] + 5, z_s + random.uniform(-2, 2)]

        r = random.uniform(0.1, 0.4)
        voxelize_legal_space(spatial_unit, LA_LegalSpaceUtilityNetwork(
            PhysicalUtilityNetwork(p1, p2, r), clearance=0.5
        ))

        sui = calculate_sui(spatial_unit)
        status, path_len, direct_len = check_routability(spatial_unit)

        pdr = 1.0 if status == "STRAIGHT" else (
            path_len / direct_len if status == "DEVIATED" else float("inf")
        )

        # PDR > 1.05: Low → Medium (Prato & Bekhor, 2006)
        if pdr > PDR_LOW_MEDIUM and thresholds["LOW_MAX"] is None:
            thresholds["LOW_MAX"] = sui

        # PDR > 1.20: Medium → High (Barthelemy, 2022)
        if pdr > PDR_MEDIUM_HIGH and thresholds["MEDIUM_MAX"] is None:
            thresholds["MEDIUM_MAX"] = sui

        current_status = status

    return thresholds


# ============================================================
# Multi-run statistical simulation (N=30)
# ============================================================

def run_multi_simulation(n_runs: int = 30, resolution: float = 0.5, max_iterations: int = 200) -> dict:
    """
    Run N independent simulations and report mean ± std of SUI thresholds.

    This is the scientifically robust method for deriving the congestion
    thresholds reported in the paper. Each run uses a fixed seed (0 to N-1)
    for full reproducibility.

    Parameters
    ----------
    n_runs : int
        Number of independent simulation runs (default 30).
    resolution : float
        Voxel resolution in metres.
    max_iterations : int
        Maximum random pipes per run.

    Returns
    -------
    dict
        {
          'low_mean': float, 'low_std': float, 'low_cv_pct': float,
          'med_mean': float, 'med_std': float, 'med_cv_pct': float,
          'n_runs': int, 'raw': list of dict
        }
    """
    logger.info("=== Multi-Run Simulation (N=%d) for Threshold Derivation ===", n_runs)
    logger.info("Seeds: 0 to %d | Resolution: %.2f m | Max iterations: %d",
                n_runs - 1, resolution, max_iterations)

    low_vals = []
    med_vals = []
    raw_results = []

    for seed in range(n_runs):
        result = _single_run(seed=seed, resolution=resolution, max_iterations=max_iterations)
        raw_results.append({"seed": seed, **result})

        if result["LOW_MAX"] is not None:
            low_vals.append(result["LOW_MAX"])
        if result["MEDIUM_MAX"] is not None:
            med_vals.append(result["MEDIUM_MAX"])

        logger.info(
            "  Run %2d/%d | LOW_MAX=%-6s | MEDIUM_MAX=%-6s",
            seed + 1, n_runs,
            f"{result['LOW_MAX']:.2f}%" if result["LOW_MAX"] else "None",
            f"{result['MEDIUM_MAX']:.2f}%" if result["MEDIUM_MAX"] else "None",
        )

    def _stats(vals, label):
        if not vals:
            logger.warning("%s threshold never triggered in any run.", label)
            return None, None, None
        arr = np.array(vals)
        mean, std = float(np.mean(arr)), float(np.std(arr, ddof=1))
        cv   = (std / mean * 100) if mean > 0 else 0.0
        return mean, std, cv

    low_mean, low_std, low_cv   = _stats(low_vals,  "LOW_MAX")
    med_mean, med_std, med_cv   = _stats(med_vals, "MEDIUM_MAX")

    logger.info("\n=== Simulation Results (N=%d runs) ===", n_runs)
    if low_mean is not None:
        logger.info(
            "Low/Medium boundary  (PDR>1.05): %.2f%% ± %.2f%% (CV=%.1f%%)",
            low_mean, low_std, low_cv,
        )
    if med_mean is not None:
        logger.info(
            "Medium/High boundary (PDR>1.20): %.2f%% ± %.2f%% (CV=%.1f%%)",
            med_mean, med_std, med_cv,
        )
    logger.info(
        "Thresholds as SCC ratios: LOW < %.3f | MEDIUM < %.3f",
        (low_mean / 100) if low_mean else 0,
        (med_mean / 100) if med_mean else 0,
    )

    return {
        "low_mean":   low_mean,  "low_std":   low_std,  "low_cv_pct":  low_cv,
        "med_mean":   med_mean,  "med_std":   med_std,  "med_cv_pct":  med_cv,
        "n_runs":     n_runs,
        "seeds_used": list(range(n_runs)),
        "raw":        raw_results,
    }


# ============================================================
# Single-run entry (original run_simulation, seed-aware)
# ============================================================

def run_simulation(seed: int = DEFAULT_SEED, resolution: float = 0.5) -> None:
    """
    Run a single simulation with a fixed seed and log results.

    For full statistical derivation of thresholds, use run_multi_simulation().
    """
    logger.info("=== Single-Run Simulation (seed=%d) ===", seed)

    try:
        parcel_mesh = trimesh.load("full_network_parcel.obj")
        if isinstance(parcel_mesh, trimesh.Scene):
            parcel_mesh = trimesh.util.concatenate(list(parcel_mesh.geometry.values()))
    except Exception as exc:
        logger.error("Failed to load full_network_parcel.obj: %s", exc)
        sys.exit(1)

    logger.info("LA_SpatialUnit Bounds: %s to %s", _BOUNDS_MIN, _BOUNDS_MAX)
    spatial_unit = LA_SpatialUnit(_BOUNDS_MIN, _BOUNDS_MAX, resolution=resolution)
    logger.info("Parcel Volume: %.2f m³ | Grid Shape: %s", spatial_unit.volume, spatial_unit.shape)

    thresholds = _single_run(seed=seed, resolution=resolution)

    low = thresholds.get("LOW_MAX")
    med = thresholds.get("MEDIUM_MAX")

    logger.info("\n--- Simulation Complete (seed=%d) ---", seed)
    logger.info("Low/Medium boundary (PDR>1.05, Prato & Bekhor 2006):  SUI < %s",
                f"{low:.2f}%" if low else "None (threshold not reached)")
    logger.info("Medium/High boundary (PDR>1.20, Barthelemy 2022):     SUI < %s",
                f"{med:.2f}%" if med else "None (threshold not reached)")


# ============================================================
# CLI entry point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D UUDM Congestion Simulator — LADM/ISO 19152 Edition"
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Run N independent simulations (default N=30) and report mean ± std.",
    )
    parser.add_argument(
        "-n", "--n-runs",
        type=int,
        default=30,
        help="Number of independent runs for --multi mode (default: 30).",
    )
    parser.add_argument(
        "-s", "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for single-run mode (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=float,
        default=0.5,
        help="Voxel resolution in metres (default: 0.5). Lower = slower but more accurate.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=200,
        help="Maximum random pipes per simulation run (default: 200).",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.multi:
        results = run_multi_simulation(
            n_runs=args.n_runs,
            resolution=args.resolution,
            max_iterations=args.max_iter,
        )
        import json
        print(json.dumps(
            {k: v for k, v in results.items() if k != "raw"},
            indent=4,
        ))
    else:
        run_simulation(seed=args.seed, resolution=args.resolution)
