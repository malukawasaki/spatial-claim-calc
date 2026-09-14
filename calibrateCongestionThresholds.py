"""
calibrateCongestionThresholds.py — Empirical SUI Threshold Calibration

Derives statistically grounded, depth-stratified Congestion Index thresholds
for SpatialClaimCalc.py via Monte Carlo constructability insertion stress-testing.

Methodology
-----------
For each vertical stratum (Shallow, Intermediate, Deep), a standardised
right-of-way corridor sandbox is incrementally populated with clustered
utility duct banks. After each insertion, the Constructability Insertion
Probability (P_insertion) is evaluated: can a standardised candidate utility
pass from corridor inlet (X=0) to outlet (X=corridor_length) without
colliding with existing legal buffers?

P_insertion is computed by probing K random insertion positions and checking
for collision-free passage via 3D connected-component analysis
(scipy.ndimage.label) on the free-space mask.

Two transition thresholds are recorded per run:
  - LOW -> MEDIUM:  SUI at which P_insertion first drops below 0.80
  - MEDIUM -> HIGH: SUI at which P_insertion first drops below 0.20

After N independent runs (seeds 0 to N-1), robust summary statistics
(mean, std, median, IQR, 95% CI) are computed for each threshold.

Key Design Decisions
--------------------
- Constructability insertion probability replaces A* pathfinding/PDR to
  directly measure whether new infrastructure can physically fit.
- Clustered duct-bank generation (80% longitudinal, 20% transverse,
  bundles of 2-4) models realistic municipal utility layouts.
- Per-stratum calibration with stratum-appropriate candidate sizes and
  clearance buffers reflects Singapore policy (LTA CoPWPS / PUB).
- Default resolution 0.25 m matches SpatialClaimCalc.py production voxels.

References
----------
- ISO 19152-1:2022; ISO 19152-2:2025
- Singapore LTA CoPWPS (2021), section 4.3
- AWWA C600/C605 — pipe joint deflection tolerances
- ASCE MOP 144 — thrust restraint design
"""

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, label

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("calibrateCongestionThresholds")


# ============================================================
# Stratum Configuration
# ============================================================

@dataclass
class StratumConfig:
    """Configuration for a single vertical stratum calibration."""
    name: str
    z_min: float              # Bottom of stratum (most negative)
    z_max: float              # Top of stratum (least negative)
    candidate_diameter: float  # Candidate utility diameter (m)
    candidate_clearance: float  # Statutory clearance buffer (m)
    pipe_radius_range: tuple = (0.05, 0.20)  # Existing utility radius range
    pipe_clearance: float = 0.3  # Clearance for existing utilities


# Default strata aligned with Singapore civil codes
DEFAULT_STRATA = [
    StratumConfig(
        name="Shallow_Utilities",
        z_min=-3.0, z_max=-1.5,
        candidate_diameter=0.3, candidate_clearance=0.3,
        pipe_radius_range=(0.05, 0.15), pipe_clearance=0.3,
    ),
    StratumConfig(
        name="Intermediate_Utilities",
        z_min=-7.0, z_max=-3.0,
        candidate_diameter=1.0, candidate_clearance=0.5,
        pipe_radius_range=(0.15, 0.50), pipe_clearance=0.5,
    ),
    StratumConfig(
        name="Deep_Infrastructure",
        z_min=-30.0, z_max=-7.0,
        candidate_diameter=6.5, candidate_clearance=6.0,
        pipe_radius_range=(1.0, 3.0), pipe_clearance=2.0,
    ),
]

# Default corridor sandbox dimensions
CORRIDOR_LENGTH = 40.0   # X-axis (m)
CORRIDOR_WIDTH = 10.0    # Y-axis (m), centred at Y=0 -> [-5, 5]

# Constructability insertion probability thresholds
P_LOW_MEDIUM = 0.80   # Below this = onset of spatial restriction
P_MEDIUM_HIGH = 0.20  # Below this = constructability failure dominant


# ============================================================
# Voxelization Utilities
# ============================================================

def _voxelize_cylinder_segment(
    grid: np.ndarray,
    grid_min: np.ndarray,
    resolution: float,
    p1: np.ndarray,
    p2: np.ndarray,
    radius: float,
) -> None:
    """Mark voxels intersecting a cylinder segment defined by endpoints and radius."""
    min_pt = np.minimum(p1, p2) - radius
    max_pt = np.maximum(p1, p2) + radius

    min_idx = np.floor((min_pt - grid_min) / resolution).astype(int)
    max_idx = np.ceil((max_pt - grid_min) / resolution).astype(int)

    min_idx = np.maximum(min_idx, 0)
    max_idx = np.minimum(max_idx, np.array(grid.shape) - 1)

    if np.any(min_idx > max_idx):
        return

    x = np.arange(min_idx[0], max_idx[0] + 1)
    y = np.arange(min_idx[1], max_idx[1] + 1)
    z = np.arange(min_idx[2], max_idx[2] + 1)

    if len(x) == 0 or len(y) == 0 or len(z) == 0:
        return

    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    points = np.stack([xx, yy, zz], axis=-1) * resolution + grid_min + (resolution / 2.0)

    v = p1
    w = p2
    l2 = np.sum((w - v) ** 2)

    if l2 == 0.0:
        dist = np.linalg.norm(points - v, axis=-1)
    else:
        t = np.clip(np.sum((points - v) * (w - v), axis=-1) / l2, 0.0, 1.0)
        projection = v + t[..., np.newaxis] * (w - v)
        dist = np.linalg.norm(points - projection, axis=-1)

    mask = dist <= radius
    grid[
        min_idx[0]: max_idx[0] + 1,
        min_idx[1]: max_idx[1] + 1,
        min_idx[2]: max_idx[2] + 1,
    ][mask] = True


def _make_spherical_structuring_element(radius_voxels: int) -> np.ndarray:
    """Create a spherical binary structuring element for morphological dilation."""
    r = radius_voxels
    z_g, y_g, x_g = np.ogrid[-r: r + 1, -r: r + 1, -r: r + 1]
    return (x_g ** 2 + y_g ** 2 + z_g ** 2 <= r ** 2)


# ============================================================
# Clustered Utility Population Generator
# ============================================================

def _generate_clustered_utilities(
    stratum: StratumConfig,
    corridor_length: float,
    corridor_half_width: float,
    rng: random.Random,
    np_rng: np.random.RandomState,
) -> list:
    """
    Generate a single batch of clustered utility duct bank segments.

    Returns a list of (p1, p2, total_radius) tuples where total_radius
    includes the physical radius plus the stratum clearance.
    """
    segments = []

    # Decide bundle size (2 to 4 parallel runs)
    bundle_size = rng.randint(2, 4)

    # 80% longitudinal (X-axis), 20% transverse (Y-axis)
    is_longitudinal = rng.random() < 0.80

    for _ in range(bundle_size):
        r = rng.uniform(*stratum.pipe_radius_range)
        total_r = r + stratum.pipe_clearance

        z_pos = rng.uniform(stratum.z_min + total_r, stratum.z_max - total_r)

        if is_longitudinal:
            # X-axis aligned with slight lateral offset
            y_pos = rng.uniform(-corridor_half_width + total_r,
                                corridor_half_width - total_r)
            y_jitter = rng.uniform(-0.5, 0.5)
            y_pos_end = np.clip(y_pos + y_jitter,
                                -corridor_half_width + total_r,
                                corridor_half_width - total_r)
            p1 = np.array([0.0, y_pos, z_pos])
            p2 = np.array([corridor_length, y_pos_end, z_pos])
        else:
            # Y-axis transverse crossing
            x_pos = rng.uniform(total_r, corridor_length - total_r)
            p1 = np.array([x_pos, -corridor_half_width, z_pos])
            p2 = np.array([x_pos, corridor_half_width, z_pos])

        segments.append((p1, p2, total_r))

    return segments


# ============================================================
# Constructability Insertion Test (Connected-Component Passage)
# ============================================================

def _test_constructability(
    occupied_grid: np.ndarray,
    grid_min: np.ndarray,
    resolution: float,
    stratum: StratumConfig,
    corridor_length: float,
    corridor_half_width: float,
    rng: random.Random,
    n_probes: int = 10,
) -> float:
    """
    Evaluate constructability insertion probability.

    Dilates existing obstacles by the candidate utility's total legal radius,
    then uses 3D connected-component labelling to check whether free space
    connects the inlet face (X=0) to the outlet face (X=corridor_length).

    Probes n_probes random Y/Z insertion positions. Returns the fraction
    that have a collision-free passage (P_insertion).
    """
    candidate_total_radius = (stratum.candidate_diameter / 2.0) + stratum.candidate_clearance
    if np.any(occupied_grid):
        dist = distance_transform_edt(~occupied_grid) * resolution
        dilated = dist <= candidate_total_radius
    else:
        dilated = np.zeros_like(occupied_grid)

    # Free-space mask (True = unoccupied)
    free_space = ~dilated

    # Label connected components in free space (6-connectivity)
    labelled, n_components = label(free_space)

    if n_components == 0:
        return 0.0

    # Find components that touch both inlet (X=0) and outlet (X=max) faces
    inlet_labels = set(np.unique(labelled[0, :, :])) - {0}
    outlet_labels = set(np.unique(labelled[-1, :, :])) - {0}
    through_labels = inlet_labels & outlet_labels

    if not through_labels:
        return 0.0

    # Create a mask of voxels belonging to through-going components
    through_mask = np.isin(labelled, list(through_labels))

    # Probe random positions on the inlet face
    successes = 0
    grid_shape = occupied_grid.shape

    for _ in range(n_probes):
        # Random Y/Z position for the candidate utility centre
        y_pos = rng.uniform(-corridor_half_width + candidate_total_radius,
                            corridor_half_width - candidate_total_radius)
        z_pos = rng.uniform(stratum.z_min + candidate_total_radius,
                            stratum.z_max - candidate_total_radius)

        # Convert to voxel index on the inlet face (X=0)
        iy = int((y_pos - grid_min[1]) / resolution)
        iz = int((z_pos - grid_min[2]) / resolution)

        # Bounds check
        if 0 <= iy < grid_shape[1] and 0 <= iz < grid_shape[2]:
            if through_mask[0, iy, iz]:
                successes += 1

    return successes / n_probes


# ============================================================
# SUI Calculation
# ============================================================

def _calculate_sui(occupied_grid: np.ndarray) -> float:
    """Spatial Utilization Index as a ratio [0, 1]."""
    occupied = np.sum(occupied_grid)
    total = occupied_grid.size
    if total == 0:
        return 0.0
    return float(occupied / total)


# ============================================================
# Single Calibration Run (Per Stratum)
# ============================================================

def _single_calibration_run(
    seed: int,
    stratum: StratumConfig,
    resolution: float = 0.25,
    corridor_length: float = CORRIDOR_LENGTH,
    corridor_half_width: float = CORRIDOR_WIDTH / 2.0,
    max_insertions: int = 200,
    n_probes: int = 10,
) -> dict:
    """
    Execute a single simulation run for one stratum.

    Returns dict with seed, threshold SUI values, and insertion curve data.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # Corridor sandbox bounds
    grid_min = np.array([0.0, -corridor_half_width, stratum.z_min])
    grid_max = np.array([corridor_length, corridor_half_width, stratum.z_max])
    extents = grid_max - grid_min
    shape = np.ceil(extents / resolution).astype(int)

    # Ensure non-zero shape
    shape = np.maximum(shape, 1)

    occupied_grid = np.zeros(shape, dtype=bool)

    result = {
        "seed": seed,
        "low_medium_sui": None,
        "medium_high_sui": None,
        "final_sui": None,
        "insertions": 0,
        "termination": "MAX_ITER",
    }

    for step in range(max_insertions):
        # Generate and insert a clustered utility bundle
        segments = _generate_clustered_utilities(
            stratum, corridor_length, corridor_half_width, rng, np_rng
        )

        for p1, p2, total_r in segments:
            _voxelize_cylinder_segment(
                occupied_grid, grid_min, resolution, p1, p2, total_r
            )

        sui = _calculate_sui(occupied_grid)
        result["insertions"] = step + 1

        # Evaluate constructability
        p_insertion = _test_constructability(
            occupied_grid, grid_min, resolution, stratum,
            corridor_length, corridor_half_width, rng, n_probes
        )

        # Check threshold transitions
        if p_insertion < P_LOW_MEDIUM and result["low_medium_sui"] is None:
            result["low_medium_sui"] = sui

        if p_insertion < P_MEDIUM_HIGH and result["medium_high_sui"] is None:
            result["medium_high_sui"] = sui

        # Early termination if corridor is fully severed
        if p_insertion == 0.0 and result["medium_high_sui"] is not None:
            result["termination"] = "SEVERED"
            break

    result["final_sui"] = _calculate_sui(occupied_grid)
    return result


# ============================================================
# Monte Carlo Calibration (Per Stratum)
# ============================================================

def calibrate_stratum(
    stratum: StratumConfig,
    n_runs: int = 30,
    resolution: float = 0.25,
    max_insertions: int = 200,
    n_probes: int = 10,
) -> dict:
    """
    Run N independent simulations for a single stratum and compute statistics.
    """
    logger.info("--- Calibrating stratum: %s (Z: %.1f to %.1f m) ---",
                stratum.name, stratum.z_min, stratum.z_max)
    logger.info("    Candidate: D=%.2f m, clearance=%.2f m | N=%d runs | res=%.2f m",
                stratum.candidate_diameter, stratum.candidate_clearance,
                n_runs, resolution)

    low_vals = []
    high_vals = []
    raw_results = []

    for seed in range(n_runs):
        t0 = time.time()
        result = _single_calibration_run(
            seed=seed, stratum=stratum, resolution=resolution,
            max_insertions=max_insertions, n_probes=n_probes,
        )
        elapsed = time.time() - t0
        raw_results.append(result)

        if result["low_medium_sui"] is not None:
            low_vals.append(result["low_medium_sui"])
        if result["medium_high_sui"] is not None:
            high_vals.append(result["medium_high_sui"])

        logger.info(
            "  Run %2d/%d [%.1fs] | Low->Med: %s | Med->High: %s | "
            "Final SUI: %.3f (%s)",
            seed + 1, n_runs, elapsed,
            f"{result['low_medium_sui']:.3f}" if result["low_medium_sui"] else "---",
            f"{result['medium_high_sui']:.3f}" if result["medium_high_sui"] else "---",
            result["final_sui"],
            result["termination"],
        )

    # Compute robust statistics
    def _stats(vals, label_name):
        if not vals:
            logger.warning("  %s threshold never triggered.", label_name)
            return None
        arr = np.array(vals)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
        cv = (std / mean * 100) if mean > 0 else 0.0
        median = float(np.median(arr))
        q25, q75 = float(np.percentile(arr, 25)), float(np.percentile(arr, 75))
        iqr = q75 - q25

        if n > 1:
            from scipy.stats import t as t_dist
            t_crit = t_dist.ppf(0.975, df=n - 1)
            se = std / np.sqrt(n)
            ci_low = mean - t_crit * se
            ci_high = mean + t_crit * se
        else:
            ci_low = ci_high = mean

        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "cv_pct": round(cv, 2),
            "median": round(median, 4),
            "q25": round(q25, 4),
            "q75": round(q75, 4),
            "iqr": round(iqr, 4),
            "ci95_low": round(ci_low, 4),
            "ci95_high": round(ci_high, 4),
            "n_triggered": n,
        }

    low_stats = _stats(low_vals, "LOW->MEDIUM")
    high_stats = _stats(high_vals, "MEDIUM->HIGH")

    # Derive recommended thresholds
    thresholds = {
        "low_medium": round(low_stats["mean"], 4) if low_stats else None,
        "medium_high": round(high_stats["mean"], 4) if high_stats else None,
    }

    return {
        "stratum": stratum.name,
        "depth_range_m": [stratum.z_min, stratum.z_max],
        "candidate_diameter_m": stratum.candidate_diameter,
        "candidate_clearance_m": stratum.candidate_clearance,
        "thresholds": thresholds,
        "statistics": {
            "low_medium": low_stats,
            "medium_high": high_stats,
        },
        "per_run_results": raw_results,
    }


# ============================================================
# Full Calibration (All Strata)
# ============================================================

def calibrate_all_strata(
    strata: list = None,
    n_runs: int = 30,
    resolution: float = 0.25,
    max_insertions: int = 200,
    n_probes: int = 10,
) -> dict:
    """
    Run calibration across all strata and return combined results.
    """
    if strata is None:
        strata = DEFAULT_STRATA

    logger.info("=" * 60)
    logger.info(" CONGESTION THRESHOLD CALIBRATION")
    logger.info(" Method: Constructability Insertion Probability")
    logger.info("=" * 60)
    logger.info("Strata: %d | Runs/stratum: %d | Resolution: %.2f m",
                len(strata), n_runs, resolution)

    t_start = time.time()
    strata_results = {}

    for stratum in strata:
        result = calibrate_stratum(
            stratum, n_runs=n_runs, resolution=resolution,
            max_insertions=max_insertions, n_probes=n_probes,
        )
        strata_results[stratum.name] = result

    total_elapsed = time.time() - t_start

    output = {
        "calibration_method": "Monte Carlo Constructability Insertion Probability",
        "p_low_medium": P_LOW_MEDIUM,
        "p_medium_high": P_MEDIUM_HIGH,
        "resolution_m": resolution,
        "n_runs": n_runs,
        "max_insertions": max_insertions,
        "n_probes": n_probes,
        "corridor_length_m": CORRIDOR_LENGTH,
        "corridor_width_m": CORRIDOR_WIDTH,
        "elapsed_seconds": round(total_elapsed, 1),
        "strata": strata_results,
    }

    # Print human-readable summary
    print()
    print("=" * 60)
    print(" CONGESTION THRESHOLD CALIBRATION RESULTS")
    print("=" * 60)
    print(f" Method:     Constructability Insertion Probability")
    print(f" Resolution: {resolution} m")
    print(f" Runs:       {n_runs} per stratum")
    print(f" Total time: {total_elapsed:.1f}s")
    print()

    for name, sr in strata_results.items():
        print(f" --- {name} (Z: {sr['depth_range_m'][0]} to {sr['depth_range_m'][1]} m) ---")
        t = sr["thresholds"]
        s = sr["statistics"]

        if s["low_medium"]:
            lm = s["low_medium"]
            print(f"   LOW -> MEDIUM (P < {P_LOW_MEDIUM}):")
            print(f"     Mean SUI:  {lm['mean']:.3f} +/- {lm['std']:.3f}  (CV={lm['cv_pct']:.1f}%)")
            print(f"     Median:    {lm['median']:.3f}")
            print(f"     95% CI:    [{lm['ci95_low']:.3f}, {lm['ci95_high']:.3f}]")
            print(f"     Triggered: {lm['n_triggered']}/{n_runs} runs")
        else:
            print(f"   LOW -> MEDIUM: NOT TRIGGERED")

        if s["medium_high"]:
            mh = s["medium_high"]
            print(f"   MEDIUM -> HIGH (P < {P_MEDIUM_HIGH}):")
            print(f"     Mean SUI:  {mh['mean']:.3f} +/- {mh['std']:.3f}  (CV={mh['cv_pct']:.1f}%)")
            print(f"     Median:    {mh['median']:.3f}")
            print(f"     95% CI:    [{mh['ci95_low']:.3f}, {mh['ci95_high']:.3f}]")
            print(f"     Triggered: {mh['n_triggered']}/{n_runs} runs")
        else:
            print(f"   MEDIUM -> HIGH: NOT TRIGGERED")

        print(f"   -> Recommended thresholds: low_medium={t['low_medium']}, medium_high={t['medium_high']}")
        print()

    print("=" * 60)

    return output


# ============================================================
# CLI Entry Point
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Empirical SUI Threshold Calibration via Monte Carlo "
            "Constructability Insertion Probability testing."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python calibrateCongestionThresholds.py                       # Full (N=30, 0.25m)\n"
            "  python calibrateCongestionThresholds.py -n 5 -r 0.5          # Quick test\n"
            "  python calibrateCongestionThresholds.py --json thresholds.json  # Save JSON\n"
            "  python calibrateCongestionThresholds.py --stratum Shallow_Utilities  # Single stratum\n"
        ),
    )
    parser.add_argument(
        "-n", "--n-runs",
        type=int, default=30,
        help="Number of Monte Carlo runs per stratum (default: 30).",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=float, default=0.25,
        help="Voxel resolution in metres (default: 0.25). Use 0.5 for faster runs.",
    )
    parser.add_argument(
        "--max-insertions",
        type=int, default=200,
        help="Maximum utility bundle insertions per run (default: 200).",
    )
    parser.add_argument(
        "--n-probes",
        type=int, default=10,
        help="Number of random insertion probes per step (default: 10).",
    )
    parser.add_argument(
        "--stratum",
        type=str, default=None,
        choices=["Shallow_Utilities", "Intermediate_Utilities", "Deep_Infrastructure"],
        help="Calibrate a single stratum only (default: all three).",
    )
    parser.add_argument(
        "--json",
        type=str, default=None, metavar="FILE",
        help="Save full results to a JSON file.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.stratum:
        strata = [s for s in DEFAULT_STRATA if s.name == args.stratum]
    else:
        strata = DEFAULT_STRATA

    results = calibrate_all_strata(
        strata=strata,
        n_runs=args.n_runs,
        resolution=args.resolution,
        max_insertions=args.max_insertions,
        n_probes=args.n_probes,
    )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {args.json}")
