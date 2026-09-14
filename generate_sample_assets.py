"""
generate_sample_assets.py — Realistic Urban Cadastral Lot & Subsurface Network

Produces two stand-alone GLB files:
  1. sample_uudm_utilities.glb:
     Synthetic Marina Bay multi-utility network with road verge services 
     and an oblique diagonal MRT tunnel.
  2. sample_ladm_parcel.glb:
     A realistic urban mixed-use development parcel (50 m × 25 m × 30 m)
     centred at [45.0, 0.0, -15.0], intersecting both boundary utility 
     easements and deep transit protection reserves.

Usage:
    python generate_sample_assets.py
"""

import logging
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

def _cylinder_from_endpoints(p1, p2, radius, sections=32):
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
# Utility Definitions
# ---------------------------------------------------------------------------
_UTILITY_RECORDS = [
    {
        "id": "TEL-FIBER-01",
        "type": "TELECOM",
        "radius": 0.1,
        "depth_start": -2.0, "depth_end": -2.0,
        "path_2d": [(0.0, -2.0), (150.0, -2.0)],  # Frontage edge
        "material": "HDPE", "operator": "Singtel",
        "color": [50, 220, 50, 200],
    },
    {
        "id": "PWR-22KV-01",
        "type": "POWER_CABLE",
        "radius": 0.15,
        "depth_start": -2.5, "depth_end": -2.5,
        "path_2d": [(0.0, -1.0), (150.0, -1.0)],  # Frontage edge
        "material": "HDPE", "operator": "Singapore_Power",
        "color": [255, 165, 0, 200],
    },
    {
        "id": "WAT-POT-01",
        "type": "WATER_PIPE",
        "radius": 0.3,
        "depth_start": -3.5, "depth_end": -3.5,
        "path_2d": [(0.0, 1.0), (150.0, 1.0)],
        "material": "Steel", "operator": "PUB",
        "color": [30, 100, 255, 200],
    },
    {
        "id": "DCS-MB-001",
        "type": "DISTRICT_COOLING",
        "radius": 0.8,
        "depth_start": -6.0, "depth_end": -6.0,
        "path_2d": [(0.0, 2.5), (150.0, 2.5)],
        "material": "Steel", "operator": "SP_Group",
        "color": [0, 200, 255, 200],
    },
    {
        "id": "MRT-NSL-001",
        "type": "MRT_TUNNEL",
        "radius": 3.0,
        "depth_start": -20.0, "depth_end": -25.0,
        # Diagonal traverse directly slicing across the parcel footprint:
        "path_2d": [(10.0, -30.0), (80.0, 30.0)],
        "material": "Steel", "operator": "SMRT",
        "color": [220, 50, 50, 200],
    },
    {
        "id": "DTSS-MB-01",
        "type": "SEWER_TUNNEL",
        "radius": 1.5,
        "depth_start": -40.0, "depth_end": -42.0,
        "path_2d": [(-20.0, 0.0), (180.0, 0.0)],  # Far beneath parcel floor
        "material": "Concrete", "operator": "PUB",
        "color": [139, 69, 19, 200],
    },
]

def _build_3d_endpoints(rec):
    path = rec["path_2d"]
    return (path[0][0], path[0][1], rec["depth_start"]), (path[1][0], path[1][1], rec["depth_end"])

# ---------------------------------------------------------------------------
# REALISTIC URBAN PARCEL (Matching Magenta Box)
# Dimensions: 50 m (X: [20, 70]) × 25 m (Y: [-12.5, 12.5]) × 30 m (Z: [0, -30])
# ---------------------------------------------------------------------------
_PARCEL_CENTRE = np.array([45.0, 0.0, -15.0])
_PARCEL_EXTENTS = np.array([50.0, 25.0, 30.0])  # Area: 1,250 m2, Volume: 37,500 m3

_CLIMA_LADM_EXTRAS = {
    "LADM_Class":                     "ExtSpatialClaim",
    "LADM_Standard":                  "ISO 19152-5:2024",
    "parcelId":                       "MarinaBay_Commercial_Lot401",
    "administrativeSource":           "Singapore_SLA",
    "referenceFrame":                 "SVY21",
    "verticalDatum":                  "Singapore_Height_Datum",
    "registrationDate":               str(date.today()),
    "climaAdaptation_profile":        "CLIMA_LADM",
    "climaAdaptation_hazardCategory": "Underground_Congestion",
    "climaAdaptation_riskLevel":      "TBD",
}

def generate_uudm_utilities_glb(output_path: str = "sample_uudm_utilities.glb"):
    scene = trimesh.Scene()
    built = []

    for rec in _UTILITY_RECORDS:
        p1, p2 = _build_3d_endpoints(rec)
        mesh = _cylinder_from_endpoints(p1, p2, rec["radius"])
        mesh.visual.face_colors = rec["color"]
        scene.add_geometry(mesh, node_name=rec["id"])
        built.append(rec)

    scene.export(output_path)
    glb = pygltflib.GLTF2().load(output_path)
    mesh_nodes = [node for node in glb.nodes if node.mesh is not None]

    for i, node in enumerate(mesh_nodes):
        if i < len(built):
            rec = built[i]
            node.name = rec["id"]
            node.extras = {
                "utilityId":         rec["id"],
                "utilityType":       rec["type"],
                "diameter_m":        rec["radius"] * 2.0,
                "depthStart_m":      rec["depth_start"],
                "depthEnd_m":        rec["depth_end"],
                "material":          rec["material"],
                "operationalStatus": "In_Use",
                "qualityLevel":      "QL-B",
                "operator":          rec["operator"],
                "dataStandard":      "Singapore_UUDM",
            }

    glb.save(output_path)
    logger.info("Exported UUDM database with %d assets → %s", len(built), output_path)
    return output_path

def generate_ladm_parcel_glb(output_path: str = "sample_ladm_parcel.glb"):
    parcel = trimesh.creation.box(extents=_PARCEL_EXTENTS)
    parcel.apply_translation(_PARCEL_CENTRE)
    parcel.visual.face_colors = [255, 0, 255, 60]  # Magenta tint

    scene = trimesh.Scene([parcel])
    scene.export(output_path)

    glb = pygltflib.GLTF2().load(output_path)
    for node in glb.nodes:
        if node.mesh is not None:
            node.name = "MarinaBay_Commercial_Lot401"
            node.extras = _CLIMA_LADM_EXTRAS

    glb.save(output_path)
    logger.info("Exported realistic LADM parcel (50×25×30 m) → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Preview image: combined geometry (Publication White Aesthetic)
# ---------------------------------------------------------------------------

def _make_box_poly(x_range, y_range, z_range, facecolor, edgecolor, alpha=0.92, lw=1.1):
    """Helper to create a solid architectural 3D box mesh."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    x0, x1 = x_range
    y0, y1 = y_range
    z0, z1 = z_range
    
    corners = np.array([
        [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
        [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
    ])
    
    faces = [
        [corners[0], corners[1], corners[2], corners[3]], # bottom
        [corners[4], corners[5], corners[6], corners[7]], # top
        [corners[0], corners[1], corners[5], corners[4]], # front
        [corners[2], corners[3], corners[7], corners[6]], # back
        [corners[0], corners[3], corners[7], corners[4]], # left
        [corners[1], corners[2], corners[6], corners[5]], # right
    ]
    
    return Poly3DCollection(
        faces,
        facecolor=facecolor,
        edgecolor=edgecolor,
        alpha=alpha,
        linewidth=lw,
        linestyle="-"
    )


def generate_preview_image(output_path: str = "sample_geometry_preview.png"):
    """
    Render an elegant, publication-grade 3D visualization showing all utility pipes,
    the urban cadastral parcel with high-contrast black dashed outline, transparent
    ground plane, and solid setback skyscraper with a clean legend placed above the diagram.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D          # noqa: F401
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:
        logger.warning("matplotlib not available — skipping preview image.")
        return None

    logger.info("Rendering combined geometry preview (white publication theme)...")

    fig = plt.figure(figsize=(12.0, 8.8), facecolor="#FFFFFF", dpi=300)
    ax = fig.add_axes([-0.26, 0.02, 1.28, 0.78], projection="3d")
    ax.set_facecolor("#FFFFFF")
    ax.computed_zorder = False

    STYLE_MAP = {
        "TEL-FIBER-01": {
            "color": "#059669",      # Emerald Green
            "name": "Telecom Fibre (TEL-FIBER-01)",
            "meta": "Ø0.20 m · Depth -2.0 m · Singtel",
            "lw": 2.4,
        },
        "PWR-22KV-01": {
            "color": "#854D0E",      # Deep Warm Bronze Brown
            "name": "Power Cable 22kV (PWR-22KV-01)",
            "meta": "Ø0.30 m · Depth -2.5 m · SP PowerGrid",
            "lw": 2.8,
        },
        "WAT-POT-01": {
            "color": "#2563EB",      # Cobalt Blue
            "name": "Potable Water Pipe (WAT-POT-01)",
            "meta": "Ø0.60 m · Depth -3.5 m · PUB",
            "lw": 3.4,
        },
        "DCS-MB-001": {
            "color": "#0891B2",      # Deep Teal / Cyan
            "name": "District Cooling Main (DCS-MB-001)",
            "meta": "Ø1.60 m · Depth -6.0 m · SP Group",
            "lw": 4.6,
        },
        "MRT-NSL-001": {
            "color": "#DC2626",      # Bright Crimson Transit Red
            "name": "MRT Bored Tunnel (MRT-NSL-001)",
            "meta": "Ø6.00 m · Depth -22.5 m · SMRT / LTA",
            "lw": 6.8,
        },
        "DTSS-MB-01": {
            "color": "#475569",      # Slate Graphite
            "name": "Deep Tunnel Sewer (DTSS-MB-01)",
            "meta": "Ø3.00 m · Depth -41.0 m · PUB",
            "lw": 5.0,
        },
    }

    # 1. Urban Ground Plane (Z = 0.0) — Translucent architectural streetscape
    ground_poly = Poly3DCollection(
        [np.array([[-15, -30, 0], [160, -30, 0], [160, 30, 0], [-15, 30, 0]])],
        facecolor="#F1F5F9",
        edgecolor="#CBD5E1",
        alpha=0.18,
        linewidth=0.8,
        linestyle="--",
    )
    ground_poly.set_zorder(1)
    ax.add_collection3d(ground_poly)

    road_strip = Poly3DCollection(
        [np.array([[-15, -4, 0], [160, -4, 0], [160, 4, 0], [-15, 4, 0]])],
        facecolor="#E2E8F0",
        edgecolor="#94A3B8",
        alpha=0.20,
        linewidth=0.6,
    )
    road_strip.set_zorder(2)
    ax.add_collection3d(road_strip)

    # 2. Simplified Solid Building — Set back to rear of Lot 401 (Y in [3.5, 11.5] m)
    tower = _make_box_poly(
        x_range=(28.0, 62.0),
        y_range=(3.5, 11.5),
        z_range=(0.0, 46.0),
        facecolor="#E2E8F0",
        edgecolor="#475569",
        alpha=0.92,
        lw=1.1
    )
    tower.set_zorder(5)
    ax.add_collection3d(tower)

    # 3. Subsurface Cadastral Parcel Cage (Z: -30 to 0 m)
    cx, cy, cz = _PARCEL_CENTRE
    dx, dy, dz = _PARCEL_EXTENTS / 2.0

    corners = np.array([
        [cx - dx, cy - dy, cz - dz], # 0: bottom front-left
        [cx + dx, cy - dy, cz - dz], # 1: bottom front-right
        [cx + dx, cy + dy, cz - dz], # 2: bottom back-right
        [cx - dx, cy + dy, cz - dz], # 3: bottom back-left
        [cx - dx, cy - dy, cz + dz], # 4: top front-left
        [cx + dx, cy - dy, cz + dz], # 5: top front-right
        [cx + dx, cy + dy, cz + dz], # 6: top back-right
        [cx - dx, cy + dy, cz + dz], # 7: top back-left
    ])

    box_faces = [
        [corners[0], corners[1], corners[2], corners[3]], # bottom
        [corners[4], corners[5], corners[6], corners[7]], # top
        [corners[0], corners[1], corners[5], corners[4]], # front
        [corners[2], corners[3], corners[7], corners[6]], # back
        [corners[0], corners[3], corners[7], corners[4]], # left
        [corners[1], corners[2], corners[6], corners[5]], # right
    ]

    parcel_poly = Poly3DCollection(
        box_faces,
        alpha=0.04,
        facecolor="#475569",
        edgecolor="none",
    )
    parcel_poly.set_zorder(4)
    ax.add_collection3d(parcel_poly)

    for z_div in [-1.5, -3.0, -7.0]:
        quad = np.array([
            [cx - dx, cy - dy, z_div],
            [cx + dx, cy - dy, z_div],
            [cx + dx, cy + dy, z_div],
            [cx - dx, cy + dy, z_div],
        ])
        strata_poly = Poly3DCollection([quad], alpha=0.05, facecolor="#94A3B8", edgecolor="#94A3B8", linewidth=0.7, linestyle=":")
        strata_poly.set_zorder(3)
        ax.add_collection3d(strata_poly)

    # 4. Draw Utilities (drawn with zorder 10 to 12)
    p_len, p_wid, p_hgt = _PARCEL_EXTENTS
    p_vol = p_len * p_wid * p_hgt
    pid = _CLIMA_LADM_EXTRAS.get("parcelId", "Lot401")

    legend_elements = [
        Line2D([0], [0], color="#000000", linestyle="--", linewidth=2.4,
               label=f"LADM Cadastral Lot 401: {p_len:.0f}×{p_wid:.0f}×{p_hgt:.0f} m ({p_vol:,.0f} m³)"),
        Patch(facecolor="#E2E8F0", edgecolor="#475569", linewidth=1.2,
              label="Superstructure: Commercial Tower (Z: 0 to +46 m)"),
        Line2D([0], [0], color=STYLE_MAP["TEL-FIBER-01"]["color"], linewidth=3.2,
               label=f"{STYLE_MAP['TEL-FIBER-01']['name']} [{STYLE_MAP['TEL-FIBER-01']['meta']}]"),
        Line2D([0], [0], color=STYLE_MAP["PWR-22KV-01"]["color"], linewidth=3.2,
               label=f"{STYLE_MAP['PWR-22KV-01']['name']} [{STYLE_MAP['PWR-22KV-01']['meta']}]"),
        Line2D([0], [0], color=STYLE_MAP["WAT-POT-01"]["color"], linewidth=3.2,
               label=f"{STYLE_MAP['WAT-POT-01']['name']} [{STYLE_MAP['WAT-POT-01']['meta']}]"),
        Line2D([0], [0], color=STYLE_MAP["DCS-MB-001"]["color"], linewidth=3.2,
               label=f"{STYLE_MAP['DCS-MB-001']['name']} [{STYLE_MAP['DCS-MB-001']['meta']}]"),
        Line2D([0], [0], color=STYLE_MAP["MRT-NSL-001"]["color"], linewidth=3.2,
               label=f"{STYLE_MAP['MRT-NSL-001']['name']} [{STYLE_MAP['MRT-NSL-001']['meta']}]"),
        Line2D([0], [0], color=STYLE_MAP["DTSS-MB-01"]["color"], linewidth=3.2,
               label=f"{STYLE_MAP['DTSS-MB-01']['name']} [{STYLE_MAP['DTSS-MB-01']['meta']}]"),
    ]

    for rec in _UTILITY_RECORDS:
        p1, p2 = _build_3d_endpoints(rec)
        uid = rec["id"]
        style = STYLE_MAP.get(uid, {"color": "#6B7280", "name": uid, "meta": "", "lw": 2.5})
        color = style["color"]

        try:
            cyl_mesh = _cylinder_from_endpoints(p1, p2, rec["radius"], sections=20)
            if len(cyl_mesh.faces) > 0:
                stride = max(1, len(cyl_mesh.faces) // 600)
                sub_faces = cyl_mesh.faces[::stride]
                poly = Poly3DCollection(cyl_mesh.vertices[sub_faces], alpha=0.75, facecolor=color, edgecolor="none")
                poly.set_zorder(10)
                ax.add_collection3d(poly)
        except Exception:
            pass

        ax.plot(
            [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
            color=color, linewidth=style["lw"], solid_capstyle="round",
            zorder=12,
        )

    # 5. Parcel Box 12 Edges — DRAWN LAST WITH zorder=100 SO DASHED LINES GO OVER UTILITIES
    box_edges = [
        # Bottom ring
        (corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0]),
        # Top ring
        (corners[4], corners[5]), (corners[5], corners[6]), (corners[6], corners[7]), (corners[7], corners[4]),
        # 4 Vertical pillars
        (corners[0], corners[4]), (corners[1], corners[5]), (corners[2], corners[6]), (corners[3], corners[7]),
    ]

    for p_start, p_end in box_edges:
        ax.plot(
            [p_start[0], p_end[0]],
            [p_start[1], p_end[1]],
            [p_start[2], p_end[2]],
            color="#000000",
            linestyle="--",
            linewidth=2.4,
            dash_capstyle="round",
            zorder=100
        )

    # 6. Minimalist Axes & Spines
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.pane.fill = False
        spine.pane.set_edgecolor("#E2E8F0")

    ax.tick_params(colors="#64748B", labelsize=8, pad=3)
    ax.set_xlabel("X (East) [m]", color="#334155", labelpad=8, fontsize=9, fontweight="500")
    ax.set_ylabel("Y (North) [m]", color="#334155", labelpad=8, fontsize=9, fontweight="500")
    ax.set_zlabel("Z (Elevation / Depth) [m]", color="#334155", labelpad=8, fontsize=9, fontweight="500")

    ax.set_box_aspect((3.6, 1.8, 2.0))
    ax.set_xlim(-15, 165)
    ax.set_ylim(-35, 35)
    ax.set_zlim(-46, 62)

    ax.view_init(elev=18, azim=-42)
    ax.grid(True, linestyle=":", color="#E2E8F0", alpha=0.9)

    # 7. Header
    fig.text(
        0.04, 0.965,
        "3D Cadastral Parcel, Urban Superstructure & Subsurface Utility Network",
        fontsize=13.5, color="#0F172A", fontweight="bold",
    )
    fig.text(
        0.04, 0.940,
        "Singapore CLIMA-LADM Profile (ISO 19152-5:2024) · Singapore UUDM Infrastructure Model (ISO 19152-2:2025)",
        fontsize=8.8, color="#64748B",
    )

    # 8. Clean Legend placed above the diagram on the left (UNCHANGED POSITION)
    fig.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(0.04, 0.925),
        ncol=2,
        fontsize=7.4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        framealpha=0.98,
        columnspacing=1.8,
        labelspacing=0.38,
        borderpad=0.55,
    )

    fig.savefig(output_path, dpi=300, facecolor="#FFFFFF")
    plt.close(fig)
    logger.info("Preview image saved → %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_uudm_utilities_glb("sample_uudm_utilities.glb")
    generate_ladm_parcel_glb("sample_ladm_parcel.glb")
    preview = generate_preview_image("sample_geometry_preview.png")

    print("\nAssets generated.")
    if preview:
        print(f"Preview image: {preview}")
    print("Run command:")
    print("python SpatialClaimCalc.py -p sample_ladm_parcel.glb -u sample_uudm_utilities.glb -r 0.25")
