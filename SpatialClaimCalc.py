"""
SpatialClaimCalc.py — 3D Spatial Claim Calculator (UUDM / LADM Edition)

Implements an ETL pipeline from a UUDM-encoded GLB spatial database to an
LADM ExtSpatialClaim object (ISO 19152-5:2024), quantifying underground
spatial congestion via voxel-based volumetric analysis.

References
----------
- ISO 19152-1:2022  — LADM core (LA_SpatialUnit, LA_RRR)
- ISO 19152-2:2025  — LA_LegalSpaceUtilityNetworkElement
- ISO 19152-5:2024  — ExtSpatialClaim (spatial plan information)
- Atazadeh et al. (2017), ISPRS Int. J. Geo-Inf., 6(12), 393
- Yan et al. (2021), Remote Sensing, 13(17), 3494  — Singapore UUDM + GLB
- Jaw et al. (2018), ISPRS Ann., IV-4/W6  — 2D→3D UUDM ETL
- He et al. (2021), Tunnelling & Underground Space Technology, 107, 103660
- Guo et al. (2020), Automation in Construction, 119, 103309
"""

import json
import logging
import sys

import numpy as np
import trimesh
import argparse

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("SpatialClaimCalc")


class SpatialClaimCalc:
    """
    Spatial ETL class for 3D Land Administration (LADM Part 5).

    Maps the LADM conceptual chain:
      PhysicalUtilityNetwork  -->  LA_LegalSpaceUtilityNetworkElement
      (ISO 19152-2:2025)               (physical pipe + statutory buffer)
          |
          v  volumetric intersection
      LA_SpatialUnit (ISO 19152-1:2022)   <-- 3D parcel bounding volume
          |
          v
      ExtSpatialClaim (ISO 19152-5:2024)  <-- this class's output

    Takes 3D underground utility data from a UUDM-encoded GLB, applies a
    legal safety buffer (LA_LegalSpaceUtilityNetworkElement), calculates the
    volumetric intersection with a spatial plan unit (LA_SpatialUnit), and
    translates the result into a Spatial Utilization Index (SUI) and
    Congestion Index.

    SUI Unit Note
    -------------
    This class returns SUI as a dimensionless ratio in [0, 1].
    The congestionSimulator.py returns SUI as a percentage in [0, 100].
    The thresholds (0.1540, 0.1671) are the ratio equivalents of the
    simulator's N=30 empirically derived means (15.40%, 16.71%).
    Conversion: ratio = percentage / 100.
    """

    def __init__(self, parcel_mesh, utilities, parcel_id="Unknown"):
        """
        Initialise the calculator.

        Parameters
        ----------
        parcel_mesh : trimesh.Trimesh
            The 3D parcel bounding volume (LA_SpatialUnit geometry).
        utilities : list of dict
            Each dict must contain:
              - 'mesh': trimesh.Trimesh of the physical utility pipe
              - 'uudm': dict of strict UUDM metadata (utilityId, utilityType, …)
        parcel_id : str
            Identifier for the spatial plan unit (LA_SpatialUnit).

        Raises
        ------
        ValueError
            If parcel_mesh has zero or negative volume.
        """
        if parcel_mesh is None or parcel_mesh.is_empty:
            raise ValueError("parcel_mesh must be a non-empty trimesh.Trimesh.")
        if not np.isfinite(parcel_mesh.volume) or parcel_mesh.volume <= 0:
            raise ValueError(
                f"parcel_mesh has invalid volume ({parcel_mesh.volume:.4f} m³). "
                "Ensure the mesh is a closed, watertight solid."
            )
        if not parcel_mesh.is_watertight:
            logger.warning(
                "parcel_mesh '%s' is NOT watertight. The contains() test will be "
                "skipped and all voxels within the bounding grid will be counted. "
                "Consider repairing the mesh for higher accuracy.",
                parcel_id,
            )

        self.parcel_mesh = parcel_mesh
        self.utilities = utilities
        self.parcel_id = parcel_id

    def calculate_congestion(self, buffer_radius: float = 1.5, resolution: float = 0.25) -> dict:
        """
        Perform the volumetric analysis and classify congestion.

        Applies a spherical legal safety buffer (binary dilation) to all
        utility meshes within the parcel extent, then computes the fraction
        of the parcel volume occupied by legal spaces
        (LA_LegalSpaceUtilityNetworkElement, ISO 19152-2:2025).

        The global dilation is applied *after* merging all utility voxel grids
        to correctly handle overlapping statutory clearances without
        double-counting (Guo et al., 2020).

        Parameters
        ----------
        buffer_radius : float
            Statutory clearance buffer in metres (default 1.5 m).
            Represents the legal protection zone radius of
            LA_LegalSpaceUtilityNetworkElement (ISO 19152-2:2025).
        resolution : float
            Voxel edge length in metres (default 0.25 m).
            Voxelization discretization error is ±resolution.
            Recommended range: 0.10 m (high accuracy) to 0.50 m (fast).

        Returns
        -------
        dict
            An ExtSpatialClaim-compatible dict (ISO 19152-5:2024) containing
            parcel volume, occupied legal volume, SUI ratio, Congestion Index,
            and intersecting UUDM asset metadata.
        """
        from scipy.ndimage import binary_dilation, binary_fill_holes

        # --- 1. Total volume of the LA_SpatialUnit ---
        parcel_volume = self.parcel_mesh.volume

        # --- 2. Voxelization engine setup ---
        bounds_min = self.parcel_mesh.bounds[0]
        bounds_max = self.parcel_mesh.bounds[1]
        extents = bounds_max - bounds_min
        shape = np.ceil(extents / resolution).astype(int)

        # Global occupied-space grid (physical pipe voxels, pre-dilation)
        occupied_grid = np.zeros(shape, dtype=bool)

        # Spherical structural element for the legal safety buffer (binary dilation)
        # Implements the statutory clearance zone of LA_LegalSpaceUtilityNetworkElement
        dilation_voxels = int(np.ceil(buffer_radius / resolution))
        z_g, y_g, x_g = np.ogrid[
            -dilation_voxels : dilation_voxels + 1,
            -dilation_voxels : dilation_voxels + 1,
            -dilation_voxels : dilation_voxels + 1,
        ]
        struct = x_g**2 + y_g**2 + z_g**2 <= dilation_voxels**2

        # --- 3. Process each UUDM utility asset ---
        uudm_claims = []

        for u in self.utilities:
            mesh = u["mesh"]
            uudm_data = u["uudm"]
            uid = uudm_data.get("utilityId", "unknown")

            # Broad-phase bounding box check (extended by buffer_radius)
            u_bounds_min = bounds_min - buffer_radius
            u_bounds_max = bounds_max + buffer_radius

            if np.any(mesh.bounds[0] > u_bounds_max) or np.any(mesh.bounds[1] < u_bounds_min):
                logger.debug("Utility %s is entirely outside parcel extent — skipped.", uid)
                continue

            # Narrow-phase: crop mesh to parcel + buffer extent.
            # Handles synthetic geometries with very long triangles (e.g. 10 km MRT tunnel)
            # that would exceed trimesh's voxelizer max_iter without clipping.
            try:
                cropped_mesh = mesh.copy()
                for i in range(3):
                    normal_pos = [0, 0, 0]
                    normal_pos[i] = 1
                    origin_pos = [0, 0, 0]
                    origin_pos[i] = u_bounds_min[i]
                    cropped_mesh = cropped_mesh.slice_plane(origin_pos, normal_pos)

                    normal_neg = [0, 0, 0]
                    normal_neg[i] = -1
                    origin_neg = [0, 0, 0]
                    origin_neg[i] = u_bounds_max[i]
                    cropped_mesh = cropped_mesh.slice_plane(origin_neg, normal_neg)

                if cropped_mesh.is_empty:
                    logger.debug("Utility %s is empty after mesh clipping — skipped.", uid)
                    continue
            except Exception as exc:
                logger.warning("Mesh slicing failed for utility %s: %s — skipped.", uid, exc)
                continue

            uudm_claims.append(uudm_data)

            try:
                # Voxelise the clipped physical pipe geometry.
                # trimesh defaults to max_iter=10, which is exhausted by long pipe
                # cylinders (e.g. 150 m at 0.25 m resolution needs ~600 iterations).
                # Compute a safe limit from the mesh diagonal in voxel units.
                mesh_diag_voxels = int(np.ceil(
                    np.linalg.norm(cropped_mesh.bounds[1] - cropped_mesh.bounds[0])
                    / resolution
                ))
                safe_max_iter = max(50, mesh_diag_voxels * 2)
                vox = cropped_mesh.voxelized(pitch=resolution, max_iter=safe_max_iter)

                # binary_fill_holes fills enclosed interior voids of the voxel solid.
                # Note: for slice_plane-clipped open cylinders the fill may be a no-op;
                # this is safe and produces no artefacts for solid meshes.
                vox_matrix = binary_fill_holes(vox.matrix)

                # Map physical pipe voxels onto the global parcel grid.
                # Legal dilation is applied globally afterwards (Step 4) to prevent
                # truncation at individual utility bounding boxes.
                offset = np.round((vox.translation - bounds_min) / resolution).astype(int)
                u_shape = vox_matrix.shape

                start_idx = np.maximum(offset, 0)
                end_idx = np.minimum(offset + np.array(u_shape), shape)

                if np.any(start_idx >= end_idx):
                    logger.debug("Utility %s voxel offset outside grid — skipped.", uid)
                    continue

                u_start = start_idx - offset
                u_end = end_idx - offset

                occupied_grid[
                    start_idx[0] : end_idx[0],
                    start_idx[1] : end_idx[1],
                    start_idx[2] : end_idx[2],
                ] |= vox_matrix[u_start[0] : u_end[0], u_start[1] : u_end[1], u_start[2] : u_end[2]]

            except Exception as exc:
                logger.warning("Voxelization failed for utility %s: %s — skipped.", uid, exc)

        # --- 4. Apply the legal safety buffer globally (Guo et al., 2020) ---
        # Dilating the merged grid (not per-utility) correctly handles overlapping
        # statutory clearances without double-counting.
        occupied_grid = binary_dilation(occupied_grid, structure=struct)

        # --- 5. Intersect with actual parcel geometry ---
        occupied_indices = np.argwhere(occupied_grid)
        if len(occupied_indices) > 0:
            is_bbox = isinstance(self.parcel_mesh, trimesh.primitives.Box) or np.isclose(
                self.parcel_mesh.volume, self.parcel_mesh.bounding_box.volume, rtol=0.005
            )

            if is_bbox:
                # All voxels in the grid are inside a box-shaped parcel
                actual_occupied_count = len(occupied_indices)
            elif self.parcel_mesh.is_watertight:
                occupied_centers = occupied_indices * resolution + bounds_min + (resolution / 2)
                try:
                    inside_mask = self.parcel_mesh.contains(occupied_centers)
                    actual_occupied_count = int(np.sum(inside_mask))
                except Exception as exc:
                    logger.warning("contains() test failed (%s); using bounding box count.", exc)
                    actual_occupied_count = len(occupied_indices)
            else:
                # Non-watertight mesh: fall back to bounding box (warned at init)
                actual_occupied_count = len(occupied_indices)
        else:
            actual_occupied_count = 0

        occupied_volume = actual_occupied_count * (resolution**3)

        # --- 6. Spatial Utilization Index (SUI) ---
        # SUI is returned as a dimensionless ratio in [0, 1].
        # The congestionSimulator returns SUI as a percentage [0, 100].
        # Thresholds below correspond to 15.40% and 16.71% in simulator units.
        # Reference: He et al. (2021), Tunnelling & Underground Space Technology, 107, 103660.
        sui_ratio = occupied_volume / parcel_volume

        # --- 7. Congestion Index (empirically derived from N=30 multi-run simulation) ---
        #
        # Thresholds computed by congestionSimulator.py --multi -n 30
        # (seeds 0–29, resolution=0.5 m, max_iterations=200):
        #
        #   Low/Medium boundary (PDR>1.05): SUI = 15.40% ± 2.40%  (CV=15.6%)
        #   Medium/High boundary (PDR>1.20): SUI = 16.71% ± 3.25% (CV=19.5%)
        #
        # Decision: use the mean values as the classification boundaries.
        # PDR transitions anchored to Prato & Bekhor (2006) at PDR=1.05
        # and Barthelemy (2022) at PDR=1.20.
        #
        # Classification:
        #   Low    : SUI < 15.40%  (SUI_ratio < 0.1540)
        #   Medium : 15.40% ≤ SUI < 16.71%  (0.1540 ≤ SUI_ratio < 0.1671)
        #   High   : SUI ≥ 16.71%  (SUI_ratio ≥ 0.1671)
        if sui_ratio < 0.1540:
            congestion_index = "Low"
        elif sui_ratio < 0.1671:
            congestion_index = "Medium"
        else:
            congestion_index = "High"

        logger.info(
            "Parcel '%s' | Volume: %.2f m³ | Occupied legal: %.2f m³ | "
            "SUI: %.4f (%.2f%%) | Congestion: %s",
            self.parcel_id,
            parcel_volume,
            occupied_volume,
            sui_ratio,
            sui_ratio * 100,
            congestion_index,
        )

        # --- 8. Format output as ExtSpatialClaim (ISO 19152-5:2024) ---
        result = {
            # LADM class identification
            "LADM_Class": "ExtSpatialClaim",              # ISO 19152-5:2024
            "LADM_Standard": "ISO 19152-5:2024",
            # LA_SpatialUnit identifier (ISO 19152-1:2022 §6.4)
            "LA_SpatialUnit_ID": self.parcel_id,
            # Calculation results
            "Spatial_Claim_Calculations": {
                "Parcel_Volume_m3": round(parcel_volume, 2),
                "Occupied_Legal_Volume_m3": round(occupied_volume, 2),
                "Voxel_Resolution_m": resolution,
                "Voxelization_Error_m3": round((resolution**3), 6),
                # SUI as ratio [0,1]; multiply by 100 for percentage
                "SUI_Ratio": round(sui_ratio, 4),
                "SUI_Percent": round(sui_ratio * 100, 2),
                "Congestion_Index": congestion_index,
                # LA_LegalSpaceUtilityNetworkElement buffer (ISO 19152-2:2025)
                "LA_LegalSpaceBuffer_m": buffer_radius,
            },
            # Intersecting UUDM assets (Yan et al., 2021)
            "Intersecting_UUDM_Assets": uudm_claims,
        }

        return result


# ---------------------------------------------------------------------------
# UUDM GLB loader
# ---------------------------------------------------------------------------
import pygltflib


def load_uudm_glb(filepath: str) -> list:
    """
    Parse a GLB file as a UUDM spatial database.

    Extracts both 3D geometries (via trimesh) and harmonised UUDM semantic
    attributes from glTF node extras (Yan et al., 2021; Jaw et al., 2018).

    Parameters
    ----------
    filepath : str
        Path to the UUDM-encoded GLB file.

    Returns
    -------
    list of dict
        Each dict contains 'mesh' (trimesh.Trimesh) and 'uudm' (metadata).
    """
    logger.info("Loading UUDM spatial database: %s", filepath)

    # Load 3D geometry via trimesh
    scene = trimesh.load(filepath)

    # Load UUDM semantic metadata from glTF node extras
    glb = pygltflib.GLTF2().load(filepath)
    metadata_map = {}
    for node in glb.nodes:
        if node.extras:
            metadata_map[node.name] = node.extras

    utilities = []

    for geom_name, node_names in scene.graph.geometry_nodes.items():
        node_name = node_names[0]
        mesh = scene.geometry[geom_name]

        if node_name in metadata_map:
            uudm_data = metadata_map[node_name]
            utilities.append({"mesh": mesh, "uudm": uudm_data})
            logger.info(
                "  Parsed: %s (%s)",
                uudm_data.get("utilityId", node_name),
                uudm_data.get("utilityType", "unknown"),
            )
        else:
            # Geometry exists but has no UUDM metadata (e.g. terrain, building shell)
            logger.warning(
                "  Node '%s' has no UUDM extras — skipped. "
                "Ensure all utility nodes have glTF extras injected by SyntheticSGUtilities.py.",
                node_name,
            )

    logger.info("Loaded %d UUDM utility assets from %s.", len(utilities), filepath)
    return utilities


# ---------------------------------------------------------------------------
# LADM parcel GLB loader
# ---------------------------------------------------------------------------

def load_ladm_parcel_glb(filepath: str) -> tuple:
    """
    Load a parcel geometry from a GLB file and extract any LADM / CLIMA-LADM
    metadata stored in glTF node extras.

    Works with:
      - GLBs produced by ``generate_sample_assets.py`` (with CLIMA-LADM extras)
      - Any GLB whose root node carries LADM-compatible extras
      - Plain GLBs with no extras (metadata returned as empty dict)

    Parameters
    ----------
    filepath : str
        Path to a GLB file containing the parcel geometry.

    Returns
    -------
    tuple of (trimesh.Trimesh, dict)
        The merged parcel mesh and a dict of LADM extras (may be empty).

    Raises
    ------
    ValueError
        If the GLB contains no geometry.
    """
    logger.info("Loading LADM parcel from GLB: %s", filepath)

    # --- 1. Load geometry ---
    loaded = trimesh.load(filepath)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if not geoms:
            raise ValueError(f"No geometry found in GLB: {filepath}")
        parcel_mesh = trimesh.util.concatenate(geoms)
    else:
        parcel_mesh = loaded

    logger.info("Parcel mesh loaded — Volume: %.2f m³", parcel_mesh.volume)

    # --- 2. Read LADM extras from glTF nodes ---
    ladm_metadata = {}
    try:
        glb = pygltflib.GLTF2().load(filepath)
        for node in glb.nodes:
            if node.extras:
                ladm_metadata = dict(node.extras)
                logger.info(
                    "LADM extras found on node '%s': %s",
                    node.name,
                    list(ladm_metadata.keys()),
                )
                break  # Use the first node that has extras
    except Exception as exc:
        logger.warning("Could not read glTF extras from %s: %s", filepath, exc)

    if not ladm_metadata:
        logger.info(
            "No LADM extras found in %s — proceeding with geometry only.", filepath
        )

    return parcel_mesh, ladm_metadata


# ---------------------------------------------------------------------------
# Resolution sensitivity analysis
# ---------------------------------------------------------------------------

def run_sensitivity_analysis(
    parcel_mesh,
    utilities,
    parcel_id: str = "SensitivityTest",
    buffer_radius: float = 1.5,
    resolutions: list = None,
) -> list:
    """
    Run SpatialClaimCalc at multiple voxel resolutions and return comparative results.

    Demonstrates that SUI values are stable across resolutions, validating the
    choice of 0.25 m as the operational resolution.

    Parameters
    ----------
    parcel_mesh : trimesh.Trimesh
    utilities : list of dict
    parcel_id : str
    buffer_radius : float
    resolutions : list of float
        Voxel sizes to test. Defaults to [0.10, 0.25, 0.50].

    Returns
    -------
    list of dict
        One result dict per resolution, each including the resolution and SUI.
    """
    if resolutions is None:
        resolutions = [0.10, 0.25, 0.50]

    logger.info("--- Resolution Sensitivity Analysis ---")
    results = []

    calc = SpatialClaimCalc(parcel_mesh=parcel_mesh, utilities=utilities, parcel_id=parcel_id)

    for res in resolutions:
        logger.info("Testing resolution: %.2f m ...", res)
        result = calc.calculate_congestion(buffer_radius=buffer_radius, resolution=res)
        summary = {
            "Resolution_m": res,
            "SUI_Ratio": result["Spatial_Claim_Calculations"]["SUI_Ratio"],
            "SUI_Percent": result["Spatial_Claim_Calculations"]["SUI_Percent"],
            "Congestion_Index": result["Spatial_Claim_Calculations"]["Congestion_Index"],
        }
        results.append(summary)
        logger.info(
            "  res=%.2f m → SUI=%.4f (%.2f%%) | %s",
            res,
            summary["SUI_Ratio"],
            summary["SUI_Percent"],
            summary["Congestion_Index"],
        )

    logger.info("--- Sensitivity Analysis Complete ---")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3D Spatial Claim Calculator — LADM/UUDM Edition (ISO 19152-5:2024)"
    )
    parser.add_argument(
        "-p", "--parcel",
        help="Path to the 3D parcel geometry file (e.g., .obj, .stl, .glb)",
    )
    parser.add_argument(
        "-u", "--utilities",
        default="sample_uudm_utilities.glb",
        help="Path to the UUDM Spatial Database (GLB file). "
             "Default: sample_uudm_utilities.glb (generate with generate_sample_assets.py)",
    )
    parser.add_argument(
        "-b", "--buffer",
        type=float,
        default=1.5,
        help="Legal safety buffer radius in metres (default: 1.5)",
    )
    parser.add_argument(
        "-r", "--resolution",
        type=float,
        default=0.25,
        help="Voxel resolution in metres (default: 0.25). Lower = more accurate but slower.",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Run resolution sensitivity analysis at 0.10, 0.25, and 0.50 m.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("=== 3D Spatial Claim Calculator (LADM/UUDM Edition) ===")

    # Load UUDM spatial database
    logger.info("Connecting to UUDM Spatial Database: %s", args.utilities)
    try:
        util_data = load_uudm_glb(args.utilities)
    except Exception as exc:
        logger.error("Error loading utilities: %s", exc)
        sys.exit(1)

    def _load_parcel(path):
        """
        Smart parcel loader.

        - If the file is a GLB: calls ``load_ladm_parcel_glb()`` to extract both
          the mesh and any CLIMA-LADM extras stored in glTF node extras.
        - For any other format (.obj, .stl, …): loads geometry only via trimesh
          and returns an empty metadata dict.

        Returns
        -------
        tuple of (trimesh.Trimesh, dict)
            Parcel mesh and LADM metadata (may be empty for non-GLB formats).
        """
        ext = path.lower().rsplit(".", 1)[-1]
        if ext in ("glb", "gltf"):
            return load_ladm_parcel_glb(path)
        # Fallback: plain geometry file
        loaded = trimesh.load(path)
        if isinstance(loaded, trimesh.Scene):
            mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        else:
            mesh = loaded
        return mesh, {}

    if args.parcel:
        logger.info("Loading custom 3D Spatial Plan Unit: %s", args.parcel)
        try:
            parcel_mesh, ladm_meta = _load_parcel(args.parcel)
            logger.info("Parcel loaded — Volume: %.2f m³", parcel_mesh.volume)

            parcel_id = (
                ladm_meta.get("parcelId")
                or ladm_meta.get("LA_SpatialUnit_ID")
                or args.parcel
            )

            calc = SpatialClaimCalc(
                parcel_mesh=parcel_mesh,
                utilities=util_data,
                parcel_id=parcel_id,
            )
            result = calc.calculate_congestion(
                buffer_radius=args.buffer,
                resolution=args.resolution,
            )

            # Embed LADM parcel metadata in the report when available
            if ladm_meta:
                result["LADM_ParcelMetadata"] = ladm_meta

            print("\n--- LADM ExtSpatialClaim Report ---")
            print(json.dumps(result, indent=4))

            if args.sensitivity:
                sens = run_sensitivity_analysis(
                    parcel_mesh=parcel_mesh,
                    utilities=util_data,
                    parcel_id=parcel_id,
                    buffer_radius=args.buffer,
                )
                print("\n--- Resolution Sensitivity Analysis ---")
                print(json.dumps(sens, indent=4))

        except Exception as exc:
            logger.error("Error processing custom parcel: %s", exc)
            sys.exit(1)

    else:
        logger.info("No --parcel provided. Running demonstration scenarios...")

        # --- SCENARIO 1: Shared Utility Trench (High Congestion) ---
        logger.info("\n--- SCENARIO 1: Shared Utility Trench (High Congestion) ---")
        parcel_A = trimesh.creation.box(extents=[5, 8, 5])
        parcel_A.apply_translation([25, -4, 0])
        logger.info("Parcel 'MarinaBay_Shared_Trench' generated (Volume: %.2f m³)", parcel_A.volume)

        calc_A = SpatialClaimCalc(
            parcel_mesh=parcel_A,
            utilities=util_data,
            parcel_id="MarinaBay_Shared_Trench",
        )
        result_A = calc_A.calculate_congestion(
            buffer_radius=args.buffer,
            resolution=args.resolution,
        )
        print(json.dumps(result_A, indent=4))

        if args.sensitivity:
            sens_A = run_sensitivity_analysis(
                parcel_mesh=parcel_A,
                utilities=util_data,
                parcel_id="MarinaBay_Shared_Trench",
                buffer_radius=args.buffer,
            )
            print("\n--- Resolution Sensitivity (Scenario 1) ---")
            print(json.dumps(sens_A, indent=4))

        # --- SCENARIO 2: Deep Infrastructure Easement (Low Congestion) ---
        logger.info("\n--- SCENARIO 2: Deep Infrastructure Easement (Low Congestion) ---")
        parcel_B = trimesh.creation.box(extents=[20, 20, 15])
        parcel_B.apply_translation([25, -10, -22.5])
        logger.info("Parcel 'MarinaBay_MRT_Easement' generated (Volume: %.2f m³)", parcel_B.volume)

        calc_B = SpatialClaimCalc(
            parcel_mesh=parcel_B,
            utilities=util_data,
            parcel_id="MarinaBay_MRT_Easement",
        )
        result_B = calc_B.calculate_congestion(
            buffer_radius=args.buffer,
            resolution=args.resolution,
        )
        print(json.dumps(result_B, indent=4))
