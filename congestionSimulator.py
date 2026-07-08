import numpy as np
import trimesh
import heapq
import sys
import random
try:
    from scipy.ndimage import binary_dilation
except ImportError:
    print("Please install scipy: pip install scipy")
    sys.exit(1)

# Import the synthetic data generator
try:
    from SyntheticSGUtilities import generate_marina_bay_synthetic_data
except ImportError:
    print("Error: Could not find SyntheticSGUtilities.py in the current directory.")
    sys.exit(1)

# ==========================================
# LADM / UUDM Class Definitions
# ==========================================

class LA_SpatialUnit:
    """
    Represents the 3D cadastral parcel bounding volume.
    LADM ISO 19152: LA_SpatialUnit
    """
    def __init__(self, bounds_min, bounds_max, resolution=0.25):
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.extents = self.bounds_max - self.bounds_min
        self.volume = np.prod(self.extents)
        self.resolution = resolution
        
        # Determine voxel grid shape
        self.shape = np.ceil(self.extents / self.resolution).astype(int)
        # 3D Boolean array representing physical and legal space occupancy
        self.grid = np.zeros(self.shape, dtype=bool)

class PhysicalUtilityNetwork:
    """
    Represents the physical pipe geometry.
    UUDM: PhysicalUtilityNetwork
    """
    def __init__(self, p1, p2, radius):
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.radius = radius

class LA_LegalSpaceUtilityNetwork:
    """
    Represents the statutory clearance / legal space of the utility.
    UUDM: LA_LegalSpaceUtilityNetwork
    """
    def __init__(self, physical_utility, clearance=0.5):
        self.physical = physical_utility
        self.clearance = clearance
        # The legal space encompasses the physical space plus the statutory clearance buffer
        self.legal_radius = physical_utility.radius + clearance


# ==========================================
# Voxelization Engine
# ==========================================

def voxelize_legal_space(spatial_unit: LA_SpatialUnit, legal_space: LA_LegalSpaceUtilityNetwork):
    """
    Efficiently masks voxels in the LA_SpatialUnit that fall within the legal space cylinder.
    """
    p1 = legal_space.physical.p1
    p2 = legal_space.physical.p2
    radius = legal_space.legal_radius
    res = spatial_unit.resolution
    min_bound = spatial_unit.bounds_min
    
    # Determine the bounding box of the cylinder to limit voxel checks
    min_pt = np.minimum(p1, p2) - radius
    max_pt = np.maximum(p1, p2) + radius
    
    min_idx = np.floor((min_pt - min_bound) / res).astype(int)
    max_idx = np.ceil((max_pt - min_bound) / res).astype(int)
    
    # Clip to grid boundaries
    min_idx = np.maximum(min_idx, 0)
    max_idx = np.minimum(max_idx, np.array(spatial_unit.shape) - 1)
    
    if np.any(min_idx > max_idx):
        return # Completely outside the parcel
        
    x = np.arange(min_idx[0], max_idx[0] + 1)
    y = np.arange(min_idx[1], max_idx[1] + 1)
    z = np.arange(min_idx[2], max_idx[2] + 1)
    
    # Create coordinate grid for the subregion
    xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack([xx, yy, zz], axis=-1) * res + min_bound + (res / 2.0)
    
    v = np.array(p1)
    w = np.array(p2)
    l2 = np.sum((w - v)**2)
    
    # Compute point-to-line-segment distance
    if l2 == 0.0:
        dist = np.linalg.norm(points - v, axis=-1)
    else:
        t = np.sum((points - v) * (w - v), axis=-1) / l2
        t = np.clip(t, 0.0, 1.0)
        projection = v + t[..., np.newaxis] * (w - v)
        dist = np.linalg.norm(points - projection, axis=-1)
    
    # Mask voxels within the legal radius
    mask = dist <= radius
    
    spatial_unit.grid[min_idx[0]:max_idx[0]+1, 
                      min_idx[1]:max_idx[1]+1, 
                      min_idx[2]:max_idx[2]+1][mask] = True

def calculate_sui(spatial_unit: LA_SpatialUnit):
    """
    Calculates the Spatial Utilization Index (SUI).
    SUI = Volume of all LA_LegalSpaceUtilityNetwork / Volume of LA_SpatialUnit
    We use the boolean grid to natively handle overlapping statutory clearances without double-counting.
    """
    occupied_voxels = np.sum(spatial_unit.grid)
    total_voxels = spatial_unit.grid.size
    return (occupied_voxels / total_voxels) * 100.0


# ==========================================
# 3D A* Pathfinding (Custom)
# ==========================================

def check_routability(spatial_unit: LA_SpatialUnit, new_pipe_legal_radius=0.5):
    """
    Attempts to route a new utility corridor along the X-axis using 3D A*.
    Returns (status, path_length, direct_distance)
    status: 'STRAIGHT', 'DEVIATED', 'FAILED'
    """
    grid = spatial_unit.grid
    shape = spatial_unit.shape
    
    # The new utility requires a 0.5m legal space. We dilate the existing obstacles by this radius so we can perform A* on a single point (the center of the new pipe).
    dilation_voxels = int(np.ceil(new_pipe_legal_radius / spatial_unit.resolution))
    
    # Create a structural element (sphere) for dilation
    z_g, y_g, x_g = np.ogrid[-dilation_voxels:dilation_voxels+1, 
                             -dilation_voxels:dilation_voxels+1, 
                             -dilation_voxels:dilation_voxels+1]
    struct = x_g**2 + y_g**2 + z_g**2 <= dilation_voxels**2
    
    dilated_grid = binary_dilation(grid, structure=struct)
    
    # Apply Minimum Surface Cover (Traffic Loads)
    # The top 1.5m is reserved for traffic loads and cannot be used for routing.
    cover_depth_voxels = int(np.ceil(1.5 / spatial_unit.resolution))
    dilated_grid[:, :, -cover_depth_voxels:] = True
    
    # 1. Check if ANY perfectly straight line is possible (Low Congestion)
    target_x = shape[0] - 1
    straight_lines = ~np.any(dilated_grid, axis=0)
    if np.any(straight_lines):
        return 'STRAIGHT', target_x, target_x
        
    # 2. Run A* from all valid start points on the X=0 plane
    valid_starts = np.argwhere(~dilated_grid[0, :, :])
    if len(valid_starts) == 0:
        return 'FAILED', 0, 0
        
    open_set = []
    g_score = {}
    
    def heuristic(pos):
        # Using a weighted heuristic (1.5x) to speed up A* in the massive 3D grid
        return 1.5 * (target_x - pos[0]) # Manhattan distance along X axis
        
    for start_yz in valid_starts:
        start_pos = (0, start_yz[0], start_yz[1])
        initial_dir = (1, 0, 0)
        g_score[(start_pos, initial_dir)] = 0
        heapq.heappush(open_set, (heuristic(start_pos), start_pos, initial_dir))
        
    visited = set()
    
    # Precompute directions and costs.
    # CRITICAL FIX for realistic rigid utilities:
    # Utilities follow precise orthogonal patterns (Manhattan routing) 
    # with 90-degree elbows. We restrict movement to strictly orthogonal 
    # directions (+X, +Y, -Y, +Z, -Z) to prevent unrealistic diagonal "snaking".
    directions = [
        (1, 0, 0, 1.0),
        (0, 1, 0, 1.0),
        (0, -1, 0, 1.0),
        (0, 0, 1, 1.0),
        (0, 0, -1, 1.0)
    ]
    
    TURN_PENALTY = 5.0
                
    while open_set:
        _, current_pos, current_dir = heapq.heappop(open_set)
        
        if current_pos[0] == target_x:
            return 'DEVIATED', g_score[(current_pos, current_dir)], target_x
            
        state = (current_pos, current_dir)
        if state in visited: continue
        visited.add(state)
        
        for dx, dy, dz, cost in directions:
            nx, ny, nz = current_pos[0]+dx, current_pos[1]+dy, current_pos[2]+dz
            new_dir = (dx, dy, dz)
            
            # Add turn penalty if direction changes
            move_cost = cost
            if new_dir != current_dir:
                move_cost += TURN_PENALTY
                
            if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                if not dilated_grid[nx, ny, nz]:
                    tentative_g = g_score[state] + move_cost
                    neighbor_state = ((nx, ny, nz), new_dir)
                    if neighbor_state not in g_score or tentative_g < g_score[neighbor_state]:
                        g_score[neighbor_state] = tentative_g
                        f_score = tentative_g + heuristic((nx, ny, nz))
                        heapq.heappush(open_set, (f_score, (nx, ny, nz), new_dir))
                        
    return 'FAILED', 0, 0


# ==========================================
# Simulation Execution
# ==========================================

def run_simulation():
    print("--- 3D UUDM Congestion Simulator ---")
    
    # 1. Load the spatial unit bounds (from the OBJ file)
    try:
        parcel_mesh = trimesh.load("full_network_parcel.obj")
        if isinstance(parcel_mesh, trimesh.Scene):
            parcel_mesh = trimesh.util.concatenate(list(parcel_mesh.geometry.values()))
    except Exception as e:
        print(f"Failed to load full_network_parcel.obj: {e}")
        sys.exit(1)
        
    # The full network parcel is 2.2 million m3. 
    # Centered at (20, 0, -12.5) with dimensions 50x50x25
    bounds_min = [-5.0, -25.0, -25.0]
    bounds_max = [45.0, 25.0, 0.0]
    
    print(f"LA_SpatialUnit Bounds: {bounds_min} to {bounds_max}")
    spatial_unit = LA_SpatialUnit(bounds_min, bounds_max, resolution=0.5) # Using 0.5m for speed in simulation
    print(f"Parcel Volume: {spatial_unit.volume:.2f} m3")
    print(f"Voxel Grid Shape: {spatial_unit.shape}")
    
    # 2. Initialize with base network from Marina Bay Synthetic Data
    print("\nLoading base utilities...")
    utils_3d = generate_marina_bay_synthetic_data()
    for u in utils_3d:
        rec = u['record']
        path = rec['path_2d']
        r = rec['radius']
        z1 = rec['depth_start']
        z2 = rec['depth_end']
        
        for i in range(len(path) - 1):
            p1 = np.array([path[i][0], path[i][1], z1])
            p2 = np.array([path[i+1][0], path[i+1][1], z2])
            
            phys = PhysicalUtilityNetwork(p1, p2, r)
            legal = LA_LegalSpaceUtilityNetwork(phys, clearance=0.5)
            voxelize_legal_space(spatial_unit, legal)
            
    base_sui = calculate_sui(spatial_unit)
    print(f"Base Network SUI: {base_sui:.2f}%")
    
    status, _, _ = check_routability(spatial_unit)
    print(f"Base Network Routability: {status}")
    
    # 3. Incremental Simulation Loop
    print("\nStarting Incremental Congestion Simulation...")
    
    # We will log the thresholds
    thresholds = {
        'LOW_MAX': None,
        'MEDIUM_MAX': None,
    }
    
    current_status = status
    iteration = 0
    max_iterations = 200 # Safety limit
    
    while current_status != 'FAILED' and iteration < max_iterations:
        iteration += 1
        
        # Generate a random pipe (50% X-aligned, 50% Y-aligned) to simulate a realistic crossing network
        if random.random() < 0.5:
            # X-aligned
            y_start = random.uniform(bounds_min[1], bounds_max[1])
            z_start = random.uniform(bounds_min[2], bounds_max[2] - 1.5)
            y_end = y_start + random.uniform(-2, 2)
            z_end = z_start + random.uniform(-2, 2)
            p1 = [bounds_min[0] - 5, y_start, z_start]
            p2 = [bounds_max[0] + 5, y_end, z_end]
        else:
            # Y-aligned
            x_start = random.uniform(bounds_min[0], bounds_max[0])
            z_start = random.uniform(bounds_min[2], bounds_max[2] - 1.5)
            x_end = x_start + random.uniform(-2, 2)
            z_end = z_start + random.uniform(-2, 2)
            p1 = [x_start, bounds_min[1] - 5, z_start]
            p2 = [x_end, bounds_max[1] + 5, z_end]
        radius = random.uniform(0.1, 0.4) # Random physical radius
        
        phys = PhysicalUtilityNetwork(p1, p2, radius)
        legal = LA_LegalSpaceUtilityNetwork(phys, clearance=0.5)
        voxelize_legal_space(spatial_unit, legal)
        
        sui = calculate_sui(spatial_unit)
        status, path_len, direct_len = check_routability(spatial_unit)
        
        # Calculate Path Deviation Ratio (PDR)
        # If straight, PDR = 1.0. If FAILED, PDR = infinity
        if status == 'STRAIGHT':
            pdr = 1.0
        elif status == 'DEVIATED':
            pdr = path_len / direct_len
        else:
            pdr = float('inf')
            
        # State machine for thresholds based on Engineering Path Deviation
        # Low: PDR <= 1.05 (Almost straight)
        # Medium: 1.05 < PDR <= 1.20 (Moderate bending required)
        # High: PDR > 1.20 (Severe bending or impenetrable)
        
        if pdr > 1.05 and thresholds['LOW_MAX'] is None:
            thresholds['LOW_MAX'] = sui
            print(f"-> Transitioned to MEDIUM Congestion at SUI: {sui:.2f}% (Path Deviation > 5%)")
            
        if pdr > 1.20 and thresholds['MEDIUM_MAX'] is None:
            thresholds['MEDIUM_MAX'] = sui
            print(f"-> Transitioned to HIGH Congestion at SUI: {sui:.2f}% (Path Deviation > 20%)")
            
        current_status = status
        
        if iteration % 20 == 0:
            print(f" Iteration {iteration} | SUI: {sui:.2f}% | Status: {status}")

    print("\n--- Simulation Complete ---")
    low = thresholds.get('LOW_MAX')
    med = thresholds.get('MEDIUM_MAX')
    
    if low is not None:
        print(f"Low Congestion (Deviation <= 5%): SUI < {low:.2f}%")
    else:
        print("Low Congestion (Deviation <= 5%): SUI < None")
        
    if med is not None:
        if low is not None:
            print(f"Medium Congestion (5% < Deviation <= 20%): {low:.2f}% <= SUI < {med:.2f}%")
        else:
            print(f"Medium Congestion (5% < Deviation <= 20%): SUI < {med:.2f}%")
        print(f"High Congestion (Deviation > 20% or Impenetrable): SUI >= {med:.2f}%")
    else:
        print("Medium Congestion (5% < Deviation <= 20%): SUI < None")
        print("High Congestion (Deviation > 20% or Impenetrable): SUI >= None")


if __name__ == "__main__":
    run_simulation()
