## LADM Climate Adaptation Profile: Spatial Claim Calculation Scripts

First introduced in 2024, the Land Administration Domain Model (LADM) climate adaptation profile integrates subsurface and climate-related data into spatial planning. By abstracting complex 3D Underground Utility Data Models (UUDM) into simplified spatial claims, the profile transforms conventional utility cadastres into actionable, climate-informed spatial constraints. This facilitates the design and exchange of climate-resilient underground spatial plans.

This repository contains the scripts used in the paper:
**"From Geological Models to 3D Utility Cadastres: Advancing the LADM Climate Adaptation Profile for Spatial Planning"** *(Published in the Land Administration Special Edition of the journal Survey Review, 2026).*

**Paper Authors:** Maria Luisa Tarozzo Kawasaki, Peter van Oosterom, and Rob van der Krogt
**Script Author:** Maria Luisa Tarozzo Kawasaki

<img width="1920" height="1080" alt="Screenshot 2026-07-08 at 16 51 49" src="https://github.com/user-attachments/assets/092e9ac0-96bd-434f-beb7-c816d78116d7" />

---

## Quick Start

**Requirements:** Python 3.9+

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the sample GLB files and preview image
python generate_sample_assets.py

# 3. Run the Spatial Claim Calculator (built-in demonstration scenarios)
python SpatialClaimCalc.py

# 4. Run with the sample LADM parcel (reads CLIMA-LADM metadata automatically)
python SpatialClaimCalc.py --parcel sample_ladm_parcel.glb

# 5. Use your own parcel and/or utility database
python SpatialClaimCalc.py --parcel your_parcel.glb --utilities your_utilities.glb

# 6. Run the congestion simulator (single seed, ~2 min)
python congestionSimulator.py --seed 42

# 7. Run the full N=30 multi-run simulation for threshold derivation (~60 min)
python congestionSimulator.py --multi --n-runs 30
```

### CLI Reference — SpatialClaimCalc.py

| Argument | Default | Description |
|---|---|---|
| `-p` / `--parcel` | *(demo mode)* | Path to a parcel file (`.glb`, `.obj`, `.stl`). GLB files are auto-inspected for CLIMA-LADM metadata. |
| `-u` / `--utilities` | `sample_uudm_utilities.glb` | Path to a UUDM spatial database (GLB). |
| `-b` / `--buffer` | `1.5` | Legal safety buffer radius in metres (LA_LegalSpaceUtilityNetworkElement). |
| `-r` / `--resolution` | `0.25` | Voxel resolution in metres. Lower = more accurate but slower. |
| `--sensitivity` | off | Run resolution sensitivity analysis at 0.10, 0.25, and 0.50 m. |
| `-v` / `--verbose` | off | Enable DEBUG-level logging. |

> **Note on performance:** The default resolution of 0.25 m is accurate but slow for large parcels. Use `--resolution 1.0` for quick exploration and `--resolution 0.25` for final results.

### Running the Tests

```bash
pytest test_spatial_claim_calc.py -v
```

---

## Repository Contents

* **`cesium_sandcastle_snippet.js`** — 3D Visualization Script (JavaScript): Designed to be imported into Cesium Sandcastle, this script creates and visualizes UUDM-standardized 3D utilities, 3D buildings, and underground parcel geometries. Replace `<YOUR ACCESS TOKEN (FROM CESIUM)>` with your own [Cesium Ion token](https://cesium.com/ion/). The script can be adapted to represent other UUDM databases or parcels globally.
* **`congestionSimulator.py`** — 3D Congestion Simulator (Python): A sophisticated 3D voxel-based pathfinding script that simulates routing new utilities through dense underground networks. By applying real-world engineering constraints—such as orthogonal turn penalties and surface cover depth limits—it calculates precise, empirical Space Utilization Index (SUI) thresholds (e.g., proving that geometric locking creates "High" congestion at just 16.3% SUI).
* **`SpatialClaimCalc.py`** — Spatial Claim Calculator (Python): A script that evaluates the level of existing spatial congestion within a specific parcel (categorized as High, Medium, or Low). It calculates this by comparing the legal volume occupied by existing utilities against the total 3D volume of the underground parcel.
* **`generate_sample_assets.py`** — Sample Asset Generator (Python): Generates `sample_uudm_utilities.glb` (6 utility pipes with UUDM metadata) and `sample_ladm_parcel.glb` (a LADM parcel with CLIMA-LADM metadata) from the geometries defined in `cesium_sandcastle_snippet.js`, plus a combined preview image.
* **`SyntheticSGUtilities.py`** — Synthetic Utility Generator (Python): Generates the Marina Bay synthetic underground network used internally by `congestionSimulator.py`.
* **`test_spatial_claim_calc.py`** — Unit tests for `SpatialClaimCalc.py`.

### Sample Assets

| File | Description |
|---|---|
| `sample_uudm_utilities.glb` | UUDM utility database — 6 pipes (DCS, MRT, PWR, TEL, WAT, DTSS) with Singapore UUDM metadata in glTF extras. |
| `sample_ladm_parcel.glb` | LADM parcel — 50×30×25 m box with CLIMA-LADM metadata (ISO 19152-5:2024) in glTF extras. |
| `sample_geometry_preview.png` | Combined 3D preview of the utilities and parcel. |

---

### Example Scenarios

**Scenario 1: Shared Utility Trench (High Congestion)**
In this scenario, a single tight bounding box is generated around a bundle of 4 distinct shallow utility pipes (Cooling, Power, Telecom, and Water).

<img width="1000" height="600" alt="scenario1" src="https://github.com/user-attachments/assets/785e11cc-a84c-4799-842e-513e9348fcf0" />

- Calculated SUI: 68.91%
- Status: HIGH Congestion
- Analysis: Because the bounding box hugs these 4 parallel pipes tightly with only a 1.5m buffer, the vast majority of the volume inside this box is consumed by the statutory legal space of the pipes. This is highly congested space where routing a new pipe would be nearly impossible without a path deviation >20%.

**Scenario 2: Deep Infrastructure Easement (Low Congestion)**
In this scenario, a bounding box is generated solely for the massive MRT Tunnel running deep underground.

<img width="1000" height="600" alt="scenario2" src="https://github.com/user-attachments/assets/4e9f245c-712a-4fd9-b232-df3e948e8f70" />

- Calculated SUI: 4.48%
- Status: LOW Congestion
- Analysis: The MRT tunnel is huge, but it is the only object in this generated parcel. The 1.5m legal space buffer around the cylinder leaves a lot of empty corners within the rectangular bounding box, resulting in a low SUI. There is no other intersecting utility preventing routing parallel to this tunnel.
