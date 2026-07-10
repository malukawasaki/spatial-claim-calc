"""
generate_sample_assets.py — Sample GLB Asset Generator

Produces two stand-alone, user-replaceable GLB files for use with SpatialClaimCalc.py:

  sample_uudm_utilities.glb
      Six underground utility pipe cylinders (Marina Bay synthetic network).
      Geometry and attributes mirror cesium_sandcastle_snippet.js.
      glTF node extras carry strict UUDM semantic metadata (Yan et al., 2021).

  sample_ladm_parcel.glb
      A single box parcel [50 m × 30 m × 25 m] centred at [10, 30, -12.5].
      Matches the "MarinaBay_Parcel" entity in cesium_sandcastle_snippet.js.
      glTF node extras carry CLIMA-LADM metadata (ISO 19152-5:2024).

  sample_geometry_preview.png
      Combined 3D matplotlib figure — utilities (colour-coded) + parcel (transparent
      wireframe box) — for quick visual validation.

Usage
-----
    python generate_sample_assets.py

Dependencies: trimesh, numpy, pygltflib, shapely, matplotlib
"""

import json
import logging
import sys
from datetime import date

import numpy as np
import trimesh
import trimesh.transformations as tf
import pygltflib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_sample_assets")

# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def _cylinder_from_endpoints(p1, p2, radius, sections=32):
    """Return a trimesh.Trimesh cylinder aligned from p1 to p2."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)
    vec = p2 - p1
    length = np.linalg.norm(vec)
    if length < 1e-9:
        raise ValueError(f"Degenerate cylinder: p1={p1}, p2={p2}")

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=sections)

    direction = vec / length
    z_axis = np.array([0.0, 0.0, 1.0])
    axis = np.cross(z_axis, direction)
    axis_norm = np.linalg.norm(axis)
    dot = float(np.clip(np.dot(z_axis, direction), -1.0, 1.0))
    angle = np.arccos(dot)

    if axis_norm > 1e-6:
        rot = tf.rotation_matrix(angle, axis / axis_norm)
    elif dot < 0:
        rot = tf.rotation_matrix(np.pi, [1.0, 0.0, 0.0])
    else:
        rot = np.eye(4)

    cyl.apply_transform(rot)
    cyl.apply_translation((p1 + p2) / 2.0)
    return cyl


# ---------------------------------------------------------------------------
# Utility definitions — sourced from cesium_sandcastle_snippet.js
# ---------------------------------------------------------------------------

# Each entry: id, type, radius_m, depth_start_m, depth_end_m, path_2d
_UTILITY_RECORDS = [
    {
        "id": "DCS-MB-001",
        "type": "DISTRICT_COOLING",
        "radius": 0.8,
        "depth_start": -6.0,
        "depth_end": -6.0,
        "path_2d": [(0, 0), (150, 0)],
        "material": "Steel",
        "operator": "SP_Group",
        "color": [0, 200, 255, 200],   # Cyan
    },
    {
        "id": "MRT-NSL-001",
        "type": "MRT_TUNNEL",
        "radius": 3.0,
        "depth_start": -20.0,
        "depth_end": -25.0,
        "path_2d": [(20, -50), (20, 150)],
        "material": "Steel",
        "operator": "SMRT",
        "color": [220, 50, 50, 200],   # Red
    },
    {
        "id": "PWR-22KV-01",
        "type": "POWER_CABLE",
        "radius": 0.15,
        "depth_start": -2.5,
        "depth_end": -2.5,
        "path_2d": [(0, 0), (150, 0)],
        "material": "HDPE",
        "operator": "Singapore_Power",
        "color": [255, 165, 0, 200],   # Orange
    },
    {
        "id": "TEL-FIBER-01",
        "type": "TELECOM",
        "radius": 0.1,
        "depth_start": -2.0,
        "depth_end": -2.0,
        "path_2d": [(0, 0), (150, 0)],
        "material": "HDPE",
        "operator": "Singtel",
        "color": [50, 220, 50, 200],   # Green
    },
    {
        "id": "WAT-POT-01",
        "type": "WATER_PIPE",
        "radius": 0.3,
        "depth_start": -3.5,
        "depth_end": -3.5,
        "path_2d": [(0, 0), (150, 0)],
        "material": "Steel",
        "operator": "PUB",
        "color": [30, 100, 255, 200],  # Blue
    },
    {
        "id": "DTSS-MB-01",
        "type": "SEWER_TUNNEL",
        "radius": 1.5,
        "depth_start": -40.0,
        "depth_end": -42.0,
        "path_2d": [(-50, 100), (200, 100)],
        "material": "Concrete",
        "operator": "PUB",
        "color": [139, 69, 19, 200],   # Brown
    },
]


def _build_3d_endpoints(rec):
    """Interpolate depth across the 2D path to produce 3D endpoints."""
    path = rec["path_2d"]
    ds, de = rec["depth_start"], rec["depth_end"]
    if len(path) == 2:
        return (path[0][0], path[0][1], ds), (path[1][0], path[1][1], de)
    # Multi-segment: use first and last only (single segment assumed for these records)
    return (path[0][0], path[0][1], ds), (path[-1][0], path[-1][1], de)


# ---------------------------------------------------------------------------
# PARCEL definition — matches cesium_sandcastle_snippet.js entity
# ---------------------------------------------------------------------------

_PARCEL_CENTRE = np.array([10.0, 30.0, -12.5])   # local CRS metres
_PARCEL_EXTENTS = np.array([50.0, 30.0, 25.0])   # X × Y × Z metres

_CLIMA_LADM_EXTRAS = {
    "LADM_Class": "ExtSpatialClaim",
    "LADM_Standard": "ISO 19152-5:2024",
    "parcelId": "MarinaBay_Sample_Parcel",
    "administrativeSource": "Singapore_SLA",
    "referenceFrame": "SVY21",
    "verticalDatum": "Singapore_Height_Datum",
    "registrationDate": str(date.today()),
    "climaAdaptation": json.dumps({
        "profile": "CLIMA_LADM",
        "hazardCategory": "Underground_Congestion",
        "riskLevel": "TBD",
    }),
}


# ---------------------------------------------------------------------------
# GLB generation: UUDM utilities
# ---------------------------------------------------------------------------

def generate_uudm_utilities_glb(output_path: str = "sample_uudm_utilities.glb"):
    """
    Build six utility pipe cylinders from cesium_sandcastle_snippet.js and export
    as a UUDM-compliant GLB with semantic extras injected into glTF nodes.
    """
    logger.info("Building UUDM utility geometries...")

    scene = trimesh.Scene()
    built = []

    for rec in _UTILITY_RECORDS:
        p1, p2 = _build_3d_endpoints(rec)
        try:
            mesh = _cylinder_from_endpoints(p1, p2, rec["radius"])
        except ValueError as e:
            logger.warning("Skipping %s: %s", rec["id"], e)
            continue

        # Apply Z-up → Y-up rotation to match glTF coordinate system
        # (same convention as SyntheticSGUtilities.py)
        rot = tf.rotation_matrix(-np.pi / 2, [1, 0, 0])
        mesh.apply_transform(rot)

        mesh.visual.face_colors = rec["color"]
        scene.add_geometry(mesh, node_name=rec["id"])
        built.append(rec)
        logger.info("  ✔ %s  r=%.2f m  depth=[%.1f → %.1f] m",
                    rec["id"], rec["radius"], rec["depth_start"], rec["depth_end"])

    scene.export(output_path)
    logger.info("Exported %d meshes → %s", len(built), output_path)

    # --- Inject UUDM metadata into glTF node extras ---
    glb = pygltflib.GLTF2().load(output_path)
    rec_map = {r["id"]: r for r in built}
    injected = 0
    for node in glb.nodes:
        if node.name in rec_map:
            rec = rec_map[node.name]
            node.extras = {
                "utilityId":         rec["id"],
                "utilityType":       rec["type"],
                "diameter_m":        rec["radius"] * 2,
                "depthStart_m":      rec["depth_start"],
                "depthEnd_m":        rec["depth_end"],
                "material":          rec["material"],
                "operationalStatus": "In_Use",
                "qualityLevel":      "QL-B",
                "operator":          rec["operator"],
                "dataStandard":      "Singapore_UUDM",
            }
            injected += 1

    glb.save(output_path)
    logger.info("UUDM metadata injected into %d glTF nodes → %s", injected, output_path)
    return output_path


# ---------------------------------------------------------------------------
# GLB generation: LADM parcel
# ---------------------------------------------------------------------------

def generate_ladm_parcel_glb(output_path: str = "sample_ladm_parcel.glb"):
    """
    Build the MarinaBay sample parcel box and export as a GLB with CLIMA-LADM
    metadata in glTF node extras.

    The parcel matches the Cesium entity in cesium_sandcastle_snippet.js:
      dimensions: [50 m, 30 m, 25 m]
      centre: [10 m, 30 m, -12.5 m]  (local CRS)
    """
    logger.info("Building LADM parcel geometry...")

    # Box parcel — watertight by construction
    parcel = trimesh.creation.box(extents=_PARCEL_EXTENTS)

    # Apply Z-up → Y-up rotation to match glTF convention
    rot = tf.rotation_matrix(-np.pi / 2, [1, 0, 0])
    parcel.apply_transform(rot)

    # Translate to parcel centre (in glTF Y-up space the Z-up centre becomes [x, z, -y])
    # Original centre in Z-up: [10, 30, -12.5]  → Y-up: [10, -12.5, -30]
    # Note: SpatialClaimCalc operates in Z-up; the GLB loader (trimesh) transparently
    # handles the Y-up→Z-up conversion when loading, so we store in Y-up here.
    centre_yup = np.array([_PARCEL_CENTRE[0], _PARCEL_CENTRE[2], -_PARCEL_CENTRE[1]])
    parcel.apply_translation(centre_yup)

    parcel.visual.face_colors = [0, 255, 255, 60]  # Transparent cyan

    scene = trimesh.Scene()
    scene.add_geometry(parcel, node_name="MarinaBay_Sample_Parcel")

    scene.export(output_path)
    logger.info("Exported parcel mesh → %s", output_path)

    # --- Inject CLIMA-LADM metadata into glTF node extras ---
    glb = pygltflib.GLTF2().load(output_path)
    for node in glb.nodes:
        if node.name == "MarinaBay_Sample_Parcel":
            node.extras = _CLIMA_LADM_EXTRAS
            logger.info("CLIMA-LADM extras injected into node '%s'", node.name)
            break

    glb.save(output_path)
    logger.info("LADM parcel GLB saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Preview image: combined geometry
# ---------------------------------------------------------------------------

def generate_preview_image(output_path: str = "sample_geometry_preview.png"):
    """
    Render a combined 3D matplotlib figure showing all six utility pipes
    (colour-coded by type) and the parcel bounding box (transparent wireframe).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    except ImportError:
        logger.warning("matplotlib not available — skipping preview image.")
        return None

    logger.info("Rendering combined geometry preview...")

    fig = plt.figure(figsize=(14, 9), facecolor="#0e1117")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#0e1117")

    _LABEL_COLORS = {
        "DISTRICT_COOLING": ("#00C8FF", "District Cooling (DCS-MB-001)"),
        "MRT_TUNNEL":       ("#DC3232", "MRT Tunnel (MRT-NSL-001)"),
        "POWER_CABLE":      ("#FFA500", "Power Cable (PWR-22KV-01)"),
        "TELECOM":          ("#32DC32", "Telecom Fibre (TEL-FIBER-01)"),
        "WATER_PIPE":       ("#1E64FF", "Water Pipe (WAT-POT-01)"),
        "SEWER_TUNNEL":     ("#8B4513", "Sewer Tunnel (DTSS-MB-01)"),
    }

    # --- Draw each utility as a thick line segment ---
    for rec in _UTILITY_RECORDS:
        p1, p2 = _build_3d_endpoints(rec)
        color, label = _LABEL_COLORS.get(rec["type"], ("#AAAAAA", rec["id"]))
        # Scale visual width by radius for readability
        lw = max(1.5, rec["radius"] * 3)
        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, linewidth=lw, label=label, solid_capstyle="round",
        )

    # --- Draw parcel as a wireframe box ---
    cx, cy, cz = _PARCEL_CENTRE
    dx, dy, dz = _PARCEL_EXTENTS / 2.0

    # 8 corners
    corners = np.array([
        [cx - dx, cy - dy, cz - dz],
        [cx + dx, cy - dy, cz - dz],
        [cx + dx, cy + dy, cz - dz],
        [cx - dx, cy + dy, cz - dz],
        [cx - dx, cy - dy, cz + dz],
        [cx + dx, cy - dy, cz + dz],
        [cx + dx, cy + dy, cz + dz],
        [cx - dx, cy + dy, cz + dz],
    ])

    # 6 faces (indices into corners)
    faces = [
        [corners[0], corners[1], corners[2], corners[3]],  # bottom
        [corners[4], corners[5], corners[6], corners[7]],  # top
        [corners[0], corners[1], corners[5], corners[4]],  # front
        [corners[2], corners[3], corners[7], corners[6]],  # back
        [corners[0], corners[3], corners[7], corners[4]],  # left
        [corners[1], corners[2], corners[6], corners[5]],  # right
    ]

    parcel_poly = Poly3DCollection(
        faces,
        alpha=0.08,
        facecolor="#00FFFF",
        edgecolor="#00FFFF",
        linewidth=0.8,
        label="LADM Parcel (50×30×25 m)",
    )
    ax.add_collection3d(parcel_poly)

    # Axes styling
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.pane.fill = False
        spine.pane.set_edgecolor("#333344")

    ax.tick_params(colors="#8888AA", labelsize=7)
    ax.set_xlabel("X  (m)", color="#8888AA", labelpad=6, fontsize=8)
    ax.set_ylabel("Y  (m)", color="#8888AA", labelpad=6, fontsize=8)
    ax.set_zlabel("Z depth (m)", color="#8888AA", labelpad=6, fontsize=8)

    # Set view limits
    ax.set_xlim(-60, 210)
    ax.set_ylim(-60, 160)
    ax.set_zlim(-50, 5)

    ax.view_init(elev=25, azim=-50)

    # Title
    fig.text(
        0.5, 0.96,
        "Marina Bay Synthetic Underground Network + LADM Parcel",
        ha="center", va="top", fontsize=13, color="#E0E0FF",
        fontweight="bold",
    )
    fig.text(
        0.5, 0.925,
        "UUDM Utilities (ISO 19152-2:2025)  ·  ExtSpatialClaim Parcel (ISO 19152-5:2024)",
        ha="center", va="top", fontsize=9, color="#8888AA",
    )

    # Legend
    legend = ax.legend(
        loc="upper left",
        fontsize=7.5,
        framealpha=0.25,
        facecolor="#1a1d2e",
        edgecolor="#555577",
        labelcolor="#DDDDFF",
        bbox_to_anchor=(0.01, 0.98),
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Preview image saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("=== Sample Asset Generator ===")

    util_path   = generate_uudm_utilities_glb("sample_uudm_utilities.glb")
    parcel_path = generate_ladm_parcel_glb("sample_ladm_parcel.glb")
    preview     = generate_preview_image("sample_geometry_preview.png")

    logger.info("")
    logger.info("Generated assets:")
    logger.info("  %s  — UUDM utility database (6 pipes)", util_path)
    logger.info("  %s     — LADM parcel with CLIMA-LADM extras", parcel_path)
    if preview:
        logger.info("  %s  — combined geometry preview", preview)

    logger.info("")
    logger.info("Usage:")
    logger.info("  python SpatialClaimCalc.py")
    logger.info("  python SpatialClaimCalc.py --parcel sample_ladm_parcel.glb")
    logger.info("  python SpatialClaimCalc.py --parcel my_custom_parcel.obj \\")
    logger.info("                             --utilities sample_uudm_utilities.glb")
