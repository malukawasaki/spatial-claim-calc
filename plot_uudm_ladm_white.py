"""
plot_uudm_ladm_white.py
Renders a publication-grade 3D visualization of sample_uudm_utilities.glb
together with the tight trench-hugging sample_ladm_parcel.glb on a clean white background.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import trimesh

from SpatialClaimCalc import load_uudm_glb, load_ladm_parcel_glb

def render_uudm_ladm_white(
    parcel_glb: str = "sample_ladm_parcel.glb",
    utilities_glb: str = "sample_uudm_utilities.glb",
    output_png: str = "sample_geometry_preview.png",
    dual_output_png: str = "sample_uudm_ladm_white.png"
):
    print(f"Loading {parcel_glb} and {utilities_glb}...")
    parcel_mesh, parcel_meta = load_ladm_parcel_glb(parcel_glb)
    utilities = load_uudm_glb(utilities_glb)

    # 16:9 widescreen publication figure
    fig = plt.figure(figsize=(16, 10), facecolor="#FFFFFF", dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#FFFFFF")

    # Vibrant, distinct colors for white background
    COLOR_MAP = {
        "TEL-FIBER-01":    ("#16A34A", "Telecom Fibre (TEL-FIBER-01)", "Singtel", 0.20, -2.0),
        "PWR-22KV-01":     ("#EA580C", "Power Cable 22kV (PWR-22KV-01)", "SP PowerGrid", 0.30, -2.5),
        "WAT-POT-01":      ("#2563EB", "Potable Water Pipe (WAT-POT-01)", "PUB", 0.60, -3.5),
        "DCS-MB-001":      ("#0284C7", "District Cooling Supply (DCS-MB-001)", "SP Group", 1.60, -6.0),
        "MRT-NSL-001":     ("#DC2626", "MRT Bored Tunnel (MRT-NSL-001)", "SMRT / LTA", 6.00, -22.5),
        "DTSS-MB-01":      ("#854D0E", "Deep Tunnel Sewerage (DTSS-MB-01)", "PUB", 3.00, -41.0),
    }

    # -------------------------------------------------------------------------
    # 1. Draw Cadastral Parcel (Tight Trench-Hugging 150m x 2.8m x 30m)
    # -------------------------------------------------------------------------
    pb_min, pb_max = parcel_mesh.bounds
    cx = (pb_min[0] + pb_max[0]) / 2.0
    cy = (pb_min[1] + pb_max[1]) / 2.0
    cz = (pb_min[2] + pb_max[2]) / 2.0
    dx = (pb_max[0] - pb_min[0]) / 2.0
    dy = (pb_max[1] - pb_min[1]) / 2.0
    dz = (pb_max[2] - pb_min[2]) / 2.0

    # 8 parcel box corners
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

    box_faces = [
        [corners[0], corners[1], corners[2], corners[3]], # bottom (Z = -30m)
        [corners[4], corners[5], corners[6], corners[7]], # top (Z = 0m)
        [corners[0], corners[1], corners[5], corners[4]], # front
        [corners[2], corners[3], corners[7], corners[6]], # back
        [corners[0], corners[3], corners[7], corners[4]], # left
        [corners[1], corners[2], corners[6], corners[5]], # right
    ]

    parcel_poly = Poly3DCollection(
        box_faces,
        alpha=0.07,
        facecolor="#8B5CF6",
        edgecolor="#6D28D9",
        linewidth=1.4,
        linestyle="-",
        label="Trench-Hugging Parcel (150 m × 2.8 m × 30 m)"
    )
    ax.add_collection3d(parcel_poly)

    # -------------------------------------------------------------------------
    # 2. Draw Subsurface Depth Strata Dividers inside the Parcel
    # -------------------------------------------------------------------------
    strata_dividers = [
        (-1.5, "#3B82F6", "Cover / Shallow Boundary (-1.5 m)"),
        (-3.0, "#8B5CF6", "Shallow / Intermediate Boundary (-3.0 m)"),
        (-7.0, "#F59E0B", "Intermediate / Deep Boundary (-7.0 m)"),
    ]

    for z_div, div_col, div_lbl in strata_dividers:
        # Cross-slice quad at z_div across parcel X and Y
        quad = np.array([
            [cx - dx, cy - dy, z_div],
            [cx + dx, cy - dy, z_div],
            [cx + dx, cy + dy, z_div],
            [cx - dx, cy + dy, z_div],
        ])
        quad_poly = Poly3DCollection([quad], alpha=0.12, facecolor=div_col, edgecolor=div_col, linewidth=0.8, linestyle=":")
        ax.add_collection3d(quad_poly)

    # -------------------------------------------------------------------------
    # 3. Draw Utility Meshes & Centerlines
    # -------------------------------------------------------------------------
    legend_handles = []
    # Add parcel to legend first
    legend_handles.append(
        Patch(facecolor="#8B5CF6", edgecolor="#6D28D9", alpha=0.25, linewidth=1.5,
              label="LADM Parcel: 150 m × 2.8 m × 30 m (12,600 m³)")
    )

    # Label positions staggered along length X to prevent collisions
    annotation_config = {
        "TEL-FIBER-01": {"x": 30.0,  "y_off": 3.8,  "z_off": 1.2},
        "PWR-22KV-01":  {"x": 65.0,  "y_off": -4.2, "z_off": 1.2},
        "WAT-POT-01":   {"x": 100.0, "y_off": 4.0,  "z_off": 1.2},
        "DCS-MB-001":   {"x": 135.0, "y_off": -4.5, "z_off": 1.5},
        "MRT-NSL-001":  {"x": 20.0,  "y_off": 14.0, "z_off": 2.5},
        "DTSS-MB-01":   {"x": 80.0,  "y_off": 5.0,  "z_off": 2.5},
    }

    for u in utilities:
        mesh = u["mesh"]
        meta = u["uudm"]
        uid = meta.get("utilityId", "Unknown")
        utype = meta.get("utilityType", "Unknown")
        dia = meta.get("diameter_m", 0.5)
        d_start = meta.get("depthStart_m", 0.0)
        operator = meta.get("operator", "")

        color_info = COLOR_MAP.get(uid, ("#475569", f"{utype} ({uid})", operator, dia, d_start))
        color = color_info[0]
        full_label = color_info[1]
        op = color_info[2]

        mb_min, mb_max = mesh.bounds
        center_x = (mb_min[0] + mb_max[0]) / 2.0
        center_y = (mb_min[1] + mb_max[1]) / 2.0
        center_z = (mb_min[2] + mb_max[2]) / 2.0

        # Render 3D solid surface mesh of pipe
        if len(mesh.faces) > 0:
            stride = max(1, len(mesh.faces) // 1000)
            sub_faces = mesh.faces[::stride]
            poly = Poly3DCollection(
                mesh.vertices[sub_faces],
                alpha=0.72,
                facecolor=color,
                edgecolor="none"
            )
            ax.add_collection3d(poly)

        # Plot crisp vector line along cylinder axis
        extents = mb_max - mb_min
        if extents[0] >= extents[1]:
            line_x = [mb_min[0], mb_max[0]]
            line_y = [center_y, center_y]
            line_z = [center_z, center_z]
        else:
            line_x = [center_x, center_x]
            line_y = [mb_min[1], mb_max[1]]
            line_z = [center_z, center_z]

        lw = max(2.0, min(dia * 2.8, 9.0))
        ax.plot(line_x, line_y, line_z, color=color, linewidth=lw, solid_capstyle="round")

        # Custom legend handle
        legend_handles.append(
            Line2D([0], [0], color=color, linewidth=3.0,
                   label=f"{uid} [{utype}]  Ø{dia:.2f} m @ Z={center_z:.1f} m ({op})")
        )

    # -------------------------------------------------------------------------
    # 4. Styling Axes, Grid, Viewpoint
    # -------------------------------------------------------------------------
    for spine in [ax.xaxis, ax.yaxis, ax.zaxis]:
        spine.pane.fill = False
        spine.pane.set_edgecolor("#CBD5E1")

    ax.tick_params(colors="#475569", labelsize=8.5)
    ax.set_xlabel("X (Length, East) [m]", color="#1E293B", labelpad=8, fontsize=9.5, fontweight="500")
    ax.set_ylabel("Y (Verge Width, North) [m]", color="#1E293B", labelpad=8, fontsize=9.5, fontweight="500")
    ax.set_zlabel("Z (Depth) [m]", color="#1E293B", labelpad=8, fontsize=9.5, fontweight="500")

    # Camera limits centered tightly on corridor
    ax.set_xlim(-15, 165)
    ax.set_ylim(-16, 16)
    ax.set_zlim(-45, 4)

    # Axonometric view showing top surface, verge width, and underground depth
    ax.view_init(elev=20, azim=-55)
    ax.grid(True, linestyle=":", color="#E2E8F0", alpha=0.85)

    fig.text(
        0.05, 0.95,
        "3D Cadastral Parcel & Subsurface Utility Network",
        fontsize=14, color="#0F172A", fontweight="bold",
    )
    fig.text(
        0.05, 0.915,
        "Singapore CLIMA-LADM Profile (ISO 19152-5:2024) · Singapore UUDM Model (ISO 19152-2:2025)",
        fontsize=9.5, color="#64748B",
    )

    # Clean Legend on top right
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.95),
        fontsize=8.5,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#E2E8F0",
        framealpha=0.96,
        labelspacing=0.55,
        borderpad=0.8,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(output_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    fig.savefig(dual_output_png, dpi=300, bbox_inches="tight", facecolor="#FFFFFF")
    plt.close(fig)
    print(f"✔ Rendered: {output_png} and {dual_output_png}")
    return output_png

if __name__ == "__main__":
    render_uudm_ladm_white()
