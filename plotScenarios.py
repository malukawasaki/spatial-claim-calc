"""
plotScenarios.py
Generates 3D diagnostic visualizations for Singapore CLIMA-LADM scenarios.
Each plot title explicitly reports:
"Shallow Layer: ... | Intermediate Layer: ... | Deep Layer: ..."
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def draw_bounding_box(ax, x_limits, y_limits, z_limits, color="#a569bd", linewidth=1.2, label="Parcel Bounding Box"):
    """Draws the 12 wireframe edges of a 3D rectangular cadastral parcel."""
    x_min, x_max = x_limits
    y_min, y_max = y_limits
    z_min, z_max = z_limits

    edges = [
        # Bottom rectangle (at z_min)
        ([x_min, x_max], [y_min, y_min], [z_min, z_min]),
        ([x_max, x_max], [y_min, y_max], [z_min, z_min]),
        ([x_max, x_min], [y_max, y_max], [z_min, z_min]),
        ([x_min, x_min], [y_max, y_min], [z_min, z_min]),
        # Top rectangle (at z_max)
        ([x_min, x_max], [y_min, y_min], [z_max, z_max]),
        ([x_max, x_max], [y_min, y_max], [z_max, z_max]),
        ([x_max, x_min], [y_max, y_max], [z_max, z_max]),
        ([x_min, x_min], [y_max, y_min], [z_max, z_max]),
        # Vertical pillars
        ([x_min, x_min], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_min, y_min], [z_min, z_max]),
        ([x_max, x_max], [y_max, y_max], [z_min, z_max]),
        ([x_min, x_min], [y_max, y_max], [z_min, z_max]),
    ]

    for i, (xs, ys, zs) in enumerate(edges):
        ax.plot3D(xs, ys, zs, color=color, linewidth=linewidth, label=label if i == 0 else "")

def format_3d_axes(ax, x_limits, y_limits, z_limits, title):
    """Applies camera perspective, grid styles, labels, and title."""
    ax.set_xlim(x_limits)
    ax.set_ylim(y_limits)
    ax.set_zlim(z_limits)

    ax.set_xlabel("X (East)", labelpad=8)
    ax.set_ylabel("Y (North)", labelpad=8)
    ax.set_zlabel("Z (Depth)", labelpad=8)
    ax.set_title(title, pad=18, fontsize=11, fontweight="normal")

    ax.view_init(elev=22, azim=-65)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, framealpha=0.9, fontsize=8)

# ==============================================================================
# SCENARIO A: DENSE FOOTWAY (High Shallow, Empty Intermediate & Deep)
# ==============================================================================
fig_a = plt.figure(figsize=(11, 6.5), dpi=300)
ax_a = fig_a.add_subplot(111, projection="3d")

x_lim_a, y_lim_a, z_lim_a = [0, 40], [-2.5, 2.5], [-12.0, 0]
draw_bounding_box(ax_a, x_lim_a, y_lim_a, z_lim_a)

x_line = [0, 40]
ax_a.plot3D(x_line, [-1.8, -1.8], [-1.8, -1.8], color="#00ffcc", linewidth=1.8, label="WAT (Water Distribution)")
ax_a.plot3D(x_line, [-1.0, -1.0], [-2.1, -2.1], color="#f39c12", linewidth=1.5, label="PWR (22kV Circuit A)")
ax_a.plot3D(x_line, [-0.3, -0.3], [-2.5, -2.5], color="#e67e22", linewidth=1.5, label="PWR (22kV Circuit B)")
ax_a.plot3D(x_line, [0.5, 0.5], [-1.9, -1.9], color="#27ae60", linewidth=1.2, label="TEL (Singtel Fiber)")
ax_a.plot3D(x_line, [1.2, 1.2], [-2.3, -2.3], color="#2ecc71", linewidth=1.2, label="TEL (NetLink Trust)")
ax_a.plot3D(x_line, [1.9, 1.9], [-2.7, -2.7], color="#f1c40f", linewidth=2.0, label="GAS (Town Gas)")

format_3d_axes(
    ax_a, x_lim_a, y_lim_a, z_lim_a,
    "Scenario A: Dense Commercial Footway\nShallow Layer: HIGH (51.2%)  |  Intermediate Layer: LOW (0.0%)  |  Deep Layer: LOW (0.0%)"
)
plt.tight_layout()
plt.savefig("Scenario_A_Shallow_High.png")
plt.close(fig_a)

# ==============================================================================
# SCENARIO B: DRAINAGE RESERVE (High Intermediate, Low Shallow & Deep)
# ==============================================================================
fig_b = plt.figure(figsize=(11, 6.5), dpi=300)
ax_b = fig_b.add_subplot(111, projection="3d")

x_lim_b, y_lim_b, z_lim_b = [0, 50], [-6.0, 6.0], [-15.0, 0]
draw_bounding_box(ax_b, x_lim_b, y_lim_b, z_lim_b)

x_line_b = [0, 50]
ax_b.plot3D(x_line_b, [4.5, 4.5], [-2.0, -2.0], color="#27ae60", linewidth=1.2, label="TEL (Telecom Duct)")
ax_b.plot3D(x_line_b, [-2.2, -2.2], [-5.0, -5.0], color="#2c3e50", linewidth=5.5, label="DRN (Twin Box Culvert 1)")
ax_b.plot3D(x_line_b, [1.8, 1.8], [-5.0, -5.0], color="#34495e", linewidth=5.5, label="DRN (Twin Box Culvert 2)")
ax_b.plot3D(x_line_b, [-4.5, -4.5], [-4.2, -4.2], color="#8e44ad", linewidth=2.5, label="SEW (Gravity Sewer Main)")

format_3d_axes(
    ax_b, x_lim_b, y_lim_b, z_lim_b,
    "Scenario B: Municipal Drainage Reserve\nShallow Layer: LOW (3.8%)  |  Intermediate Layer: HIGH (42.6%)  |  Deep Layer: LOW (0.0%)"
)
plt.tight_layout()
plt.savefig("Scenario_B_Intermediate_High.png")
plt.close(fig_b)

# ==============================================================================
# SCENARIO C: MRT TRANSIT CORRIDOR (High Deep, Low Shallow & Intermediate)
# ==============================================================================
fig_c = plt.figure(figsize=(11, 6.5), dpi=300)
ax_c = fig_c.add_subplot(111, projection="3d")

x_lim_c, y_lim_c, z_lim_c = [0, 60], [-15.0, 15.0], [-30.0, 0]
draw_bounding_box(ax_c, x_lim_c, y_lim_c, z_lim_c)

x_line_c = [0, 60]
ax_c.plot3D(x_line_c, [-10.0, -10.0], [-2.2, -2.2], color="#00ffcc", linewidth=1.5, label="WAT (Potable Water)")
ax_c.plot3D(x_line_c, [10.0, 10.0], [-4.5, -4.5], color="#f39c12", linewidth=2.0, label="PWR (66kV Transmission)")
ax_c.plot3D(x_line_c, [-6.5, -6.5], [-21.5, -21.5], color="#c0392b", linewidth=6.5, label="MRT (Bored Tunnel - Track 1)")
ax_c.plot3D(x_line_c, [6.5, 6.5], [-21.5, -21.5], color="#962d22", linewidth=6.5, label="MRT (Bored Tunnel - Track 2)")

format_3d_axes(
    ax_c, x_lim_c, y_lim_c, z_lim_c,
    "Scenario C: RTS Railway Protection Reserve\nShallow Layer: LOW (2.1%)  |  Intermediate Layer: LOW (1.4%)  |  Deep Layer: HIGH (28.4%)"
)
plt.tight_layout()
plt.savefig("Scenario_C_Deep_High.png")
plt.close(fig_c)

# ==============================================================================
# SCENARIO D: BALANCED BOULEVARD (Medium Across Upper Layers)
# ==============================================================================
fig_d = plt.figure(figsize=(11, 6.5), dpi=300)
ax_d = fig_d.add_subplot(111, projection="3d")

x_lim_d, y_lim_d, z_lim_d = [0, 50], [-7.5, 7.5], [-25.0, 0]
draw_bounding_box(ax_d, x_lim_d, y_lim_d, z_lim_d)

x_line_d = [0, 50]
ax_d.plot3D(x_line_d, [-4.0, -4.0], [-2.1, -2.1], color="#27ae60", linewidth=1.2, label="TEL (Telecom)")
ax_d.plot3D(x_line_d, [-1.5, -1.5], [-2.4, -2.4], color="#f39c12", linewidth=1.5, label="PWR (Power)")
ax_d.plot3D(x_line_d, [2.0, 2.0], [-2.0, -2.0], color="#00ffcc", linewidth=1.5, label="WAT (Water)")
ax_d.plot3D(x_line_d, [-3.0, -3.0], [-5.5, -5.5], color="#0000ff", linewidth=3.5, label="DCS (District Cooling Supply)")
ax_d.plot3D(x_line_d, [0.0, 0.0], [-5.5, -5.5], color="#1a5276", linewidth=3.5, label="DCS (District Cooling Return)")
ax_d.plot3D(x_line_d, [4.5, 4.5], [-4.8, -4.8], color="#d35400", linewidth=2.0, label="PWR (66kV Circuit)")

format_3d_axes(
    ax_d, x_lim_d, y_lim_d, z_lim_d,
    "Scenario D: Multi-Utility Urban Boulevard\nShallow Layer: MEDIUM (28.5%)  |  Intermediate Layer: MEDIUM (22.1%)  |  Deep Layer: LOW (0.0%)"
)
plt.tight_layout()
plt.savefig("Scenario_D_Balanced_Medium.png")
plt.close(fig_d)
