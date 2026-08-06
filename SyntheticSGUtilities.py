import numpy as np
import trimesh
import json
import pygltflib

try:
    from shapely.geometry import LineString
except ImportError:
    print("Please install shapely to run this script: pip install shapely")
    exit(1)

def create_cylinder_from_line(line, radius):
    """
    Extrudes a 3D line string into a CityGML/LADM style 3D cylinder.
    Pipelines in spatial GIS are often represented as 2D swept profiles.
    """
    meshes = []
    coords = list(line.coords)
    for i in range(len(coords) - 1):
        p1 = np.array(coords[i])
        p2 = np.array(coords[i+1])
        
        vector = p2 - p1
        length = np.linalg.norm(vector)
        if length == 0: continue
        
        cyl = trimesh.creation.cylinder(radius=radius, height=length)
        
        # Orient the cylinder to point from p1 to p2
        # trimesh cylinders are along the Z axis by default, centered at origin
        direction = vector / length
        z_axis = np.array([0, 0, 1])
        
        # Calculate rotation matrix from z_axis to direction
        axis = np.cross(z_axis, direction)
        axis_norm = np.linalg.norm(axis)
        dot_prod = np.clip(np.dot(z_axis, direction), -1.0, 1.0)
        angle = np.arccos(dot_prod)
        
        if axis_norm > 1e-6:
            axis = axis / axis_norm
            rot = trimesh.transformations.rotation_matrix(angle, axis)
        elif dot_prod < 0:
            # Vector is pointing exactly in -Z
            rot = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
        else:
            rot = np.eye(4)
            
        cyl.apply_transform(rot)
        
        # Move to the midpoint of p1 and p2
        midpoint = (p1 + p2) / 2.0
        cyl.apply_translation(midpoint)
        
        meshes.append(cyl)
        
    if not meshes:
        return None
    elif len(meshes) == 1:
        return meshes[0]
    else:
        # Concatenate segments into one mesh
        return trimesh.util.concatenate(meshes)

def generate_marina_bay_synthetic_data():
    """
    Generates realistic 2D shapefile-like data for Marina Bay (highly congested area)
    and reconstructs them into 3D CityGML-style geometries.
    Coordinates use a local arbitrary metric CRS to simulate SVY21.
    """
    # 2D records mimicking a typical Utility GIS Database
    synthetic_2d_records = [
        # 1. District Cooling Network (Huge pipes, deep, very prominent in Marina Bay)
        {"id": "DCS-MB-001", "type": "district_cooling", "radius": 0.8, "depth_start": -6.0, "depth_end": -6.0, 
         "path_2d": [(0, 0), (150, 0)]},
        
        # 2. MRT Tunnel Box (Very large, acts as a major spatial constraint)
        {"id": "MRT-NSL-001", "type": "mrt_tunnel", "radius": 3.0, "depth_start": -20.0, "depth_end": -25.0, 
         "path_2d": [(20, -50), (20, 150)]},
         
        # 3. High-Voltage Power Grid (e.g. SP Group 22kV cable)
        {"id": "PWR-22KV-01", "type": "power_cable", "radius": 0.15, "depth_start": -2.5, "depth_end": -2.5, 
         "path_2d": [(0, 0), (150, 0)]},
         
        # 4. Telecommunications (e.g. Singtel / Starhub fiber bundles)
        {"id": "TEL-FIBER-01", "type": "telecom", "radius": 0.1, "depth_start": -2.0, "depth_end": -2.0, 
         "path_2d": [(0, 0), (150, 0)]},
         
        # 5. Potable Water Pipe (PUB)
        {"id": "WAT-POT-01", "type": "water_pipe", "radius": 0.3, "depth_start": -3.5, "depth_end": -3.5, 
         "path_2d": [(0, 0), (150, 0)]},
         
        # 6. Deep Tunnel Sewerage System (DTSS) (Very deep, huge diameter)
        {"id": "DTSS-MB-01", "type": "sewer_tunnel", "radius": 1.5, "depth_start": -40.0, "depth_end": -42.0, 
         "path_2d": [(-50, 100), (200, 100)]}
    ]
    
    print("Processing 2D GIS records into 3D LADM geometries...")
    utilities_3d = []
    
    for record in synthetic_2d_records:
        path_2d = record["path_2d"]
        depth_s = record["depth_start"]
        depth_e = record["depth_end"]
        
        # Interpolate depths to create 3D coordinates from the 2D path
        coords_3d = []
        # Calculate total length for linear depth interpolation
        total_len = sum(np.linalg.norm(np.array(path_2d[i+1]) - np.array(path_2d[i])) for i in range(len(path_2d)-1))
        
        current_len = 0
        for i in range(len(path_2d)):
            if i > 0:
                current_len += np.linalg.norm(np.array(path_2d[i]) - np.array(path_2d[i-1]))
            
            if total_len > 0:
                z = depth_s + (depth_e - depth_s) * (current_len / total_len)
            else:
                z = depth_s
                
            x = path_2d[i][0]
            y = path_2d[i][1]
            coords_3d.append((x, y, z))
            
        # Create a Shapely 3D LineString and extrude it
        line_3d = LineString(coords_3d)
        mesh_3d = create_cylinder_from_line(line_3d, radius=record["radius"])
        
        if mesh_3d:
            utilities_3d.append({
                "id": record["id"],
                "utility_type": record["type"],
                "mesh": mesh_3d,
                "record": record
            })
            print(f" -> Generated 3D Mesh for {record['id']} ({record['type']})")
            
    return utilities_3d

if __name__ == "__main__":
    print("--- Singapore Synthetic 3D Utilities Generator ---")
    print("Simulating Marina Bay's highly congested underground...")
    
    # Generate the realistic 3D geometries
    sg_utilities = generate_marina_bay_synthetic_data()
    
    # Save the synthetic network to a combined OBJ for visualization
    scene = trimesh.Scene()
    for u in sg_utilities:
        # Assign thematic colors based on utility type
        color = [100, 100, 100, 255] # Default Grey
        if u['utility_type'] == 'district_cooling': color = [0, 200, 255, 255]   # Cyan
        elif u['utility_type'] == 'mrt_tunnel': color = [255, 0, 0, 255]         # Red
        elif u['utility_type'] == 'power_cable': color = [255, 165, 0, 255]      # Orange
        elif u['utility_type'] == 'telecom': color = [0, 255, 0, 255]            # Green
        elif u['utility_type'] == 'water_pipe': color = [0, 0, 255, 255]         # Blue
        elif u['utility_type'] == 'sewer_tunnel': color = [139, 69, 19, 255]     # Brown
        
        u['mesh'].visual.face_colors = color
        scene.add_geometry(u['mesh'], node_name=u['id'])
        
    # Apply a -90-degree rotation around the X-axis to convert from Z-up to Y-up
    # This ensures negative Z depth correctly becomes negative Y depth in glTF!
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    for geom in scene.geometry.values():
        geom.apply_transform(rot)
        
    output_path = "marina_bay_synthetic_utilities.glb"
    scene.export(output_path)
    print(f"\nSuccess! Exported {len(sg_utilities)} 3D utilities to '{output_path}'")
    
    print("\n[UUDM Compliance]")
    print("Injecting semantic metadata into glTF extras for Spatial Database use...")
    
    # Load the GLB with pygltflib to inject metadata
    glb = pygltflib.GLTF2().load(output_path)
    
    # Map each glTF node to its original record data
    record_map = {u["id"]: u["record"] for u in sg_utilities}
    
    for node in glb.nodes:
        # trimesh sets the glTF node name to the node_name we provided
        if node.name in record_map:
            rec = record_map[node.name]
            
            # UUDM-compliant metadata payload (camelCase, Yan et al. 2021)
            # operationalStatus: all base-network utilities are In_Use.
            # Previous logic ("Proposed" if "01" in rec["id"]) was incorrect
            # because every synthetic ID ends in "-001", classifying all as Proposed.
            _OPERATOR_MAP = {
                "PWR":  "Singapore_Power",
                "WAT":  "PUB",
                "DTSS": "PUB",
                "TEL":  "Singtel",
                "MRT":  "SMRT",
                "DCS":  "SP_Group",
            }
            operator = next(
                (v for k, v in _OPERATOR_MAP.items() if k in rec["id"]),
                "Unknown",
            )
            metadata = {
                "utilityId":         rec["id"],
                "utilityType":       rec["type"].upper(),
                "diameter_m":        rec["radius"] * 2,
                "depthStart_m":      rec["depth_start"],
                "depthEnd_m":        rec["depth_end"],
                "material":          (
                    "HDPE"     if rec["type"] in ["telecom", "power_cable"]
                    else "Concrete" if rec["type"] == "sewer_tunnel"
                    else "Steel"
                ),
                "operationalStatus": "In_Use",   # All base-network assets are operational
                "qualityLevel":      "QL-B",
                "operator":          operator,
                "dataStandard":      "Singapore_UUDM",
            }
            
            # Inject into the glTF 'extras' 
            node.extras = metadata
            
    glb.save(output_path)
    print("Semantic injection complete! The .glb now contains UUDM information and 3D geometry.")
