import json
import numpy as np
import trimesh
import argparse
import sys

class SpatialClaimCalc:
    """
    Spatial ETL class for 3D Land Administration (LADM Part 5).
    Takes 3D underground utility data, applies a legal safety buffer,
    calculates volumetric intersection with a spatial plan unit (parcel),
    and translates the result into a congestion index.
    """
    
    def __init__(self, parcel_mesh, utilities, parcel_id="Unknown"):
        """
        Initialize the calculator.
        
        :param parcel_mesh: trimesh.Trimesh, representing the 3D parcel bounding box
        :param utilities: list of dicts, each containing:
            - 'mesh': trimesh.Trimesh of the utility
            - 'uudm': dict containing strict UUDM metadata (material, operator, etc.)
        :param parcel_id: str, identifier for the spatial plan unit
        """
        self.parcel_mesh = parcel_mesh
        self.utilities = utilities
        self.parcel_id = parcel_id

    def calculate_congestion(self, buffer_radius=1.5):
        """
        Performs the volumetric analysis and classifies congestion using robust 3D voxelization.
        
        :param buffer_radius: float, legal protection zone radius in meters
        :return: dict matching LADM External::ExtSpatialClaim structure
        """
        import numpy as np
        from scipy.ndimage import binary_dilation, binary_fill_holes
        
        # 1. Total volume of the spatial plan unit
        parcel_volume = self.parcel_mesh.volume
        
        # 2. Voxelization Engine Setup
        resolution = 0.25 # Voxel resolution in meters
        bounds_min = self.parcel_mesh.bounds[0]
        bounds_max = self.parcel_mesh.bounds[1]
        
        # Determine global parcel grid shape
        extents = bounds_max - bounds_min
        shape = np.ceil(extents / resolution).astype(int)
        
        # Grid representing occupied space
        occupied_grid = np.zeros(shape, dtype=bool)
        
        # Create structural element for the legal safety buffer (spherical dilation)
        dilation_voxels = int(np.ceil(buffer_radius / resolution))
        z_g, y_g, x_g = np.ogrid[-dilation_voxels:dilation_voxels+1, 
                                 -dilation_voxels:dilation_voxels+1, 
                                 -dilation_voxels:dilation_voxels+1]
        struct = x_g**2 + y_g**2 + z_g**2 <= dilation_voxels**2
        
        # 3. Process each utility
        uudm_claims = []
        
        for u in self.utilities:
            mesh = u['mesh']
            uudm_data = u['uudm']
            
            u_bounds_min = bounds_min - buffer_radius
            u_bounds_max = bounds_max + buffer_radius
            
            # Broad-phase bounding box check
            if np.any(mesh.bounds[0] > u_bounds_max) or np.any(mesh.bounds[1] < u_bounds_min):
                continue # Completely outside
                
            # Narrow-phase: strictly crop the mesh to the parcel bounding box
            # This handles synthetic geometries with extremely long triangles (e.g. a 10km MRT tunnel)
            # that would otherwise exceed Trimesh's voxelizer max_iter.
            try:
                cropped_mesh = mesh.copy()
                for i in range(3):
                    # Min plane (normal pointing positive)
                    normal = [0, 0, 0]; normal[i] = 1
                    origin = [0, 0, 0]; origin[i] = u_bounds_min[i]
                    cropped_mesh = cropped_mesh.slice_plane(origin, normal)
                    
                    # Max plane (normal pointing negative)
                    normal = [0, 0, 0]; normal[i] = -1
                    origin = [0, 0, 0]; origin[i] = u_bounds_max[i]
                    cropped_mesh = cropped_mesh.slice_plane(origin, normal)
                    
                if cropped_mesh.is_empty:
                    continue
            except Exception as e:
                print(f"Warning: Mesh slicing failed for {uudm_data.get('utilityId')}: {e}")
                continue
                
            uudm_claims.append(uudm_data)
            
            try:
                # Fast voxelization of the perfectly cropped physical pipe
                vox = cropped_mesh.voxelized(pitch=resolution)
                
                # Robustly fill interior holes using SciPy
                vox_matrix = binary_fill_holes(vox.matrix)
                
                # Map the PHYSICAL pipe directly to the global parcel grid first.
                # We will apply the legal dilation buffer globally afterwards to prevent array truncation.
                offset = np.round((vox.translation - bounds_min) / resolution).astype(int)
                
                u_shape = vox_matrix.shape
                start_idx = np.maximum(offset, 0)
                end_idx = np.minimum(offset + u_shape, shape)
                
                if np.any(start_idx >= end_idx):
                    continue
                    
                u_start = start_idx - offset
                u_end = end_idx - offset
                
                occupied_grid[start_idx[0]:end_idx[0],
                              start_idx[1]:end_idx[1],
                              start_idx[2]:end_idx[2]] |= vox_matrix[u_start[0]:u_end[0],
                                                                     u_start[1]:u_end[1],
                                                                     u_start[2]:u_end[2]]
            except Exception as e:
                print(f"Warning: Failed to voxelize utility {uudm_data.get('utilityId', 'unknown')}: {e}")
                
        # 4. Apply the Legal Safety Buffer (Dilation) globally on the parcel grid
        # This prevents the legal buffers from being truncated by the physical bounding boxes.
        occupied_grid = binary_dilation(occupied_grid, structure=struct)
        
        # 5. Calculate actual intersection with the parcel
        occupied_indices = np.argwhere(occupied_grid)
        if len(occupied_indices) > 0:
            # If the parcel is a simple bounding box (volume matches), all voxels in the grid are inside.
            if np.isclose(self.parcel_mesh.volume, self.parcel_mesh.bounding_box.volume, rtol=0.01):
                actual_occupied_count = len(occupied_indices)
            else:
                occupied_centers = occupied_indices * resolution + bounds_min + (resolution / 2)
                try:
                    if self.parcel_mesh.is_watertight:
                        inside_mask = self.parcel_mesh.contains(occupied_centers)
                        actual_occupied_count = np.sum(inside_mask)
                    else:
                        actual_occupied_count = len(occupied_indices)
                except Exception:
                    actual_occupied_count = len(occupied_indices)
        else:
            actual_occupied_count = 0
            
        occupied_volume = actual_occupied_count * (resolution ** 3)
        
        # 5. Volumetric Ratio (Spatial Utilization Index - SUI)
        ratio = occupied_volume / parcel_volume
        
        # 6. Congestion Classification (Empirically derived from rigid-body 3D Pathfinding Simulator)
        if ratio < 0.10:
            congestion = 'Low'
        elif ratio < 0.163:
            congestion = 'Medium'
        else:
            congestion = 'High'
            
        # 7. Format Output (Integrating strictly UUDM attributes)
        result = {
            "Spatial_Plan_Unit": self.parcel_id,
            "Spatial_Claim_Calculations": {
                "Parcel_Volume_m3": round(parcel_volume, 2),
                "Occupied_Legal_Volume_m3": round(occupied_volume, 2),
                "Volumetric_Ratio_SUI": round(ratio, 4),
                "Congestion_Index": congestion,
                "Legal_Safety_Buffer_m": buffer_radius
            },
            "Intersecting_UUDM_Assets": uudm_claims
        }
        
        return result

import pygltflib

def load_uudm_glb(filepath):
    """
    Parses a GLB file as a spatial database, extracting both 3D geometries
    and harmonized UUDM semantic attributes from the glTF nodes.
    """
    # Load 3D geometry
    scene = trimesh.load(filepath)
    
    # Load UUDM semantic metadata
    glb = pygltflib.GLTF2().load(filepath)
    metadata_map = {}
    for node in glb.nodes:
        if node.extras:
            metadata_map[node.name] = node.extras
            
    utilities = []
    
    # trimesh maps internal geometry names (geometry_0) back to node names
    for geom_name, node_names in scene.graph.geometry_nodes.items():
        node_name = node_names[0] # taking the first node reference
        mesh = scene.geometry[geom_name]
        
        if node_name in metadata_map:
            uudm_data = metadata_map[node_name]
            utilities.append({
                'mesh': mesh,
                'uudm': uudm_data
            })
            print(f" -> Successfully parsed {uudm_data['utilityId']} ({uudm_data['utilityType']})")
            
    return utilities

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D Spatial Claim Calculator (UUDM Edition)")
    parser.add_argument("-p", "--parcel", help="Path to the 3D parcel geometry file (e.g., .obj, .stl, .glb)")
    parser.add_argument("-u", "--utilities", default="marina_bay_synthetic_utilities.glb", help="Path to the UUDM Spatial Database (GLB file)")
    parser.add_argument("-b", "--buffer", type=float, default=1.5, help="Legal safety buffer radius in meters (default: 1.5)")
    args = parser.parse_args()

    print("--- 3D Spatial Claim Calculator (UUDM Edition) ---")
    
    # Extract utilities from our single source of truth (the GLB)
    print(f"\nConnecting to UUDM Spatial Database: {args.utilities} ...")
    try:
        util_data = load_uudm_glb(args.utilities)
    except Exception as e:
        print(f"Error loading utilities: {e}")
        sys.exit(1)
        
    if args.parcel:
        # User uploaded their own custom parcel geometry
        print(f"\nLoading custom 3D Spatial Plan Unit from: {args.parcel}")
        try:
            parcel_scene = trimesh.load(args.parcel)
            
            # If the user uploaded a complex scene with multiple meshes, merge them into one bounding volume
            if isinstance(parcel_scene, trimesh.Scene):
                parcel_mesh = trimesh.util.concatenate([geom for geom in parcel_scene.geometry.values()])
            else:
                parcel_mesh = parcel_scene
                
            print(f" -> Custom Parcel loaded (Volume: {round(parcel_mesh.volume, 2)} m3)")
            
            calc = SpatialClaimCalc(parcel_mesh=parcel_mesh, utilities=util_data, parcel_id=args.parcel)
            result = calc.calculate_congestion(buffer_radius=args.buffer)
            
            print("\n--- UUDM Spatial Claim Report ---")
            print(json.dumps(result, indent=4))
            
        except Exception as e:
            print(f"Error loading custom parcel: {e}")
            sys.exit(1)
            
    else:
        # Fallback to the default demonstration scenarios if no custom parcel is provided
        print("\nNo custom --parcel provided. Running demonstration scenarios...")
        
        # --- SCENARIO 1: Micro-Easement (High Congestion) ---
        print("\n--- SCENARIO 1: Shared Utility Trench (High Congestion) ---")
        print("Defining 3D Spatial Plan Unit (Parcel)...")
        # Tightly wrap the 4 shallow utilities (Telecom, Power, Water, District Cooling)
        # which now perfectly share a single congested 2D path at X=25.
        parcel_A = trimesh.creation.box(extents=[5, 8, 5])
        parcel_A.apply_translation([25, -4, 0]) 
        print(f" -> Parcel 'MarinaBay_Shared_Trench' generated (Volume: {parcel_A.volume} m3)")
        
        calc_A = SpatialClaimCalc(parcel_mesh=parcel_A, utilities=util_data, parcel_id="MarinaBay_Shared_Trench")
        result_A = calc_A.calculate_congestion(buffer_radius=1.5)
        print(json.dumps(result_A, indent=4))
        
        # --- SCENARIO 2: Deep Infrastructure Easement ---
        print("\n--- SCENARIO 2: Deep Infrastructure Easement ---")
        print("Defining 3D Spatial Plan Unit (Parcel)...")
        # Target the massive MRT Tunnel
        parcel_B = trimesh.creation.box(extents=[20, 20, 15])
        parcel_B.apply_translation([25, -10, -22.5]) 
        print(f" -> Parcel 'MarinaBay_MRT_Easement' generated (Volume: {parcel_B.volume} m3)")
        
        calc_B = SpatialClaimCalc(parcel_mesh=parcel_B, utilities=util_data, parcel_id="MarinaBay_MRT_Easement")
        result_B = calc_B.calculate_congestion(buffer_radius=1.5)
        print(json.dumps(result_B, indent=4))
