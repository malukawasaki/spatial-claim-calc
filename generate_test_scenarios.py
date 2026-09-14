"""
generate_test_scenarios.py
Generates standardized synthetic GLB testbeds with full UUDM and LADM extras.
"""

import trimesh
import numpy as np
import pygltflib
import os

def create_pipe_mesh(p1, p2, radius):
    """Creates a cylinder mesh between two 3D endpoints."""
    vector = np.array(p2) - np.array(p1)
    height = np.linalg.norm(vector)
    cylinder = trimesh.creation.cylinder(radius=radius, height=height, sections=16)
    
    up = np.array([0, 0, 1])
    rot_axis = np.cross(up, vector / height)
    angle = np.arccos(np.clip(np.dot(up, vector / height), -1.0, 1.0))
    if np.linalg.norm(rot_axis) > 1e-6:
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        R = trimesh.transformations.rotation_matrix(angle, rot_axis)
        cylinder.apply_transform(R)
    
    midpoint = (np.array(p1) + np.array(p2)) / 2.0
    cylinder.apply_translation(midpoint)
    return cylinder

def export_uudm_scene(mesh_uudm_pairs, filepath):
    """Exports meshes and attaches UUDM extras strictly to mesh-bearing nodes."""
    meshes = [pair[0] for pair in mesh_uudm_pairs]
    metadata_list = [pair[1] for pair in mesh_uudm_pairs]
    
    # 1. Export raw geometry via trimesh
    scene = trimesh.Scene(meshes)
    scene.export(filepath)
    
    # 2. Inject UUDM extras only to nodes that reference a mesh
    gltf = pygltflib.GLTF2().load(filepath)
    mesh_nodes = [node for node in gltf.nodes if node.mesh is not None]
    
    for i, node in enumerate(mesh_nodes):
        if i < len(metadata_list):
            meta = metadata_list[i]
            node.name = meta.get("utilityId", f"util_{i}")
            node.extras = meta
            
    gltf.save(filepath)
    print(f"Exported UUDM Database: {filepath} ({len(metadata_list)} assets)")

def export_ladm_parcel(parcel_mesh, parcel_id, filepath):
    """Exports parcel mesh with LADM ExtSpatialClaim extras."""
    scene = trimesh.Scene([parcel_mesh])
    scene.export(filepath)
    
    gltf = pygltflib.GLTF2().load(filepath)
    ladm_extras = {
        "LADM_Class": "ExtSpatialClaim",
        "LADM_Standard": "ISO 19152-5:2024",
        "parcelId": parcel_id,
        "administrativeSource": "Singapore_SLA",
        "referenceFrame": "SVY21",
        "verticalDatum": "Singapore_Height_Datum",
        "climaAdaptation_profile": "CLIMA_LADM",
        "climaAdaptation_hazardCategory": "Underground_Congestion",
        "climaAdaptation_riskLevel": "TBD"
    }
    for node in gltf.nodes:
        node.name = parcel_id
        node.extras = ladm_extras
    gltf.save(filepath)
    print(f"Exported LADM Parcel: {filepath}")

# ==============================================================================
# SCENARIO A: DENSE FOOTWAY (High Shallow, Low Intermediate & Deep)
# Parcel: 4m wide (Y: [-2, 2]), 20m long (X: [0, 20]), 15m deep (Z: [0, -15])
# ==============================================================================
parcel_A = trimesh.creation.box(extents=[20.0, 4.0, 15.0])
parcel_A.apply_translation([10.0, 0.0, -7.5])
export_ladm_parcel(parcel_A, "Scenario_A_Footway_Parcel", "parcel_scenario_A.glb")

utils_A = []
configs_A = [
    {"id": "SG-WAT-DIS-01", "type": "WATER_PIPE", "y": -1.4, "z": -1.8, "r": 0.15, "op": "PUB"},
    {"id": "SG-PWR-22KV-01", "type": "POWER_CABLE", "y": -0.8, "z": -2.1, "r": 0.10, "op": "SP_PowerGrid"},
    {"id": "SG-PWR-22KV-02", "type": "POWER_CABLE", "y": -0.2, "z": -2.5, "r": 0.15, "op": "SP_PowerGrid"},
    {"id": "SG-TEL-FBR-01", "type": "TELECOM", "y": 0.4, "z": -1.9, "r": 0.10, "op": "Singtel"},
    {"id": "SG-TEL-FBR-02", "type": "TELECOM", "y": 1.0, "z": -2.3, "r": 0.10, "op": "NetLink_Trust"},
    {"id": "SG-GAS-TOWN-01", "type": "GAS", "y": 1.5, "z": -2.7, "r": 0.15, "op": "City_Energy"},
]

for cfg in configs_A:
    pipe = create_pipe_mesh([0.0, cfg["y"], cfg["z"]], [20.0, cfg["y"], cfg["z"]], radius=cfg["r"])
    uudm = {
        "utilityId": cfg["id"],
        "utilityType": cfg["type"],
        "diameter_m": cfg["r"] * 2.0,
        "depthStart_m": cfg["z"],
        "depthEnd_m": cfg["z"],
        "material": "HDPE",
        "operationalStatus": "In_Use",
        "qualityLevel": "QL-A",
        "operator": cfg["op"],
        "dataStandard": "Singapore_UUDM"
    }
    utils_A.append((pipe, uudm))

export_uudm_scene(utils_A, "utils_scenario_A.glb")

# ==============================================================================
# SCENARIO B: DRAINAGE RESERVE (Low Shallow, High Intermediate, Low Deep)
# Parcel: 10m wide (Y: [-5, 5]), 30m long (X: [0, 30]), 15m deep (Z: [0, -15])
# ==============================================================================
parcel_B = trimesh.creation.box(extents=[30.0, 10.0, 15.0])
parcel_B.apply_translation([15.0, 0.0, -7.5])
export_ladm_parcel(parcel_B, "Scenario_B_Drainage_Reserve", "parcel_scenario_B.glb")

utils_B = []
# 1 minor telecom in shallow
pipe_tel = create_pipe_mesh([0.0, 3.5, -2.0], [30.0, 3.5, -2.0], radius=0.10)
utils_B.append((pipe_tel, {
    "utilityId": "SG-TEL-FBR-99", "utilityType": "TELECOM", "diameter_m": 0.2,
    "depthStart_m": -2.0, "depthEnd_m": -2.0, "operator": "Singtel", "dataStandard": "Singapore_UUDM"
}))

# 2 massive box culverts in intermediate (-3.0 to -7.0m)
box1 = trimesh.creation.box(extents=[30.0, 3.2, 2.5])
box1.apply_translation([15.0, -2.2, -5.0])
utils_B.append((box1, {
    "utilityId": "PUB-DRN-BOX-01", "utilityType": "DRAINAGE", "diameter_m": 3.2,
    "depthStart_m": -5.0, "depthEnd_m": -5.0, "operator": "PUB", "dataStandard": "Singapore_UUDM"
}))

box2 = trimesh.creation.box(extents=[30.0, 3.2, 2.5])
box2.apply_translation([15.0, 2.2, -5.0])
utils_B.append((box2, {
    "utilityId": "PUB-DRN-BOX-02", "utilityType": "DRAINAGE", "diameter_m": 3.2,
    "depthStart_m": -5.0, "depthEnd_m": -5.0, "operator": "PUB", "dataStandard": "Singapore_UUDM"
}))

export_uudm_scene(utils_B, "utils_scenario_B.glb")

# ==============================================================================
# SCENARIO C: MRT TRANSIT CORRIDOR (Low Shallow & Intermediate, High Deep)
# Parcel: 25m wide (Y: [-12.5, 12.5]), 40m long (X: [0, 40]), 30m deep (Z: [0, -30])
# ==============================================================================
parcel_C = trimesh.creation.box(extents=[40.0, 25.0, 30.0])
parcel_C.apply_translation([20.0, 0.0, -15.0])
export_ladm_parcel(parcel_C, "Scenario_C_Transit_Corridor", "parcel_scenario_C.glb")

utils_C = []
# Minor water pipe in shallow
pipe_w = create_pipe_mesh([0.0, -8.0, -2.2], [40.0, -8.0, -2.2], radius=0.15)
utils_C.append((pipe_w, {
    "utilityId": "PUB-WAT-01", "utilityType": "WATER_PIPE", "diameter_m": 0.3,
    "depthStart_m": -2.2, "depthEnd_m": -2.2, "operator": "PUB", "dataStandard": "Singapore_UUDM"
}))

# Twin bored MRT running tunnels in deep layer (-18.0 to -24.5m)
tun1 = create_pipe_mesh([0.0, -6.0, -21.0], [40.0, -6.0, -21.0], radius=3.25)
utils_C.append((tun1, {
    "utilityId": "LTA-MRT-TUNNEL-01", "utilityType": "MRT_TUNNEL", "diameter_m": 6.5,
    "depthStart_m": -21.0, "depthEnd_m": -21.0, "operator": "LTA", "dataStandard": "Singapore_UUDM"
}))

tun2 = create_pipe_mesh([0.0, 6.0, -21.0], [40.0, 6.0, -21.0], radius=3.25)
utils_C.append((tun2, {
    "utilityId": "LTA-MRT-TUNNEL-02", "utilityType": "MRT_TUNNEL", "diameter_m": 6.5,
    "depthStart_m": -21.0, "depthEnd_m": -21.0, "operator": "LTA", "dataStandard": "Singapore_UUDM"
}))

export_uudm_scene(utils_C, "utils_scenario_C.glb")