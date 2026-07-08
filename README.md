## LADM Climate Adaptation Profile: Spatial Claim Calculation Scripts

First introduced in 2024, the Land Administration Domain Model (LADM) climate adaptation profile integrates subsurface and climate-related data into spatial planning. By abstracting complex 3D Underground Utility Data Models (UUDM) into simplified spatial claims, the profile transforms conventional utility cadastres into actionable, climate-informed spatial constraints. This facilitates the design and exchange of climate-resilient underground spatial plans.

This repository contains the scripts used in the paper: 
**"From Geological Models to 3D Utility Cadastres: Advancing the LADM Climate Adaptation Profile for Spatial Planning"** *(Published in the Land Administration Special Edition of the journal Survey Review, 2026).*

**Paper Authors:** Maria Luisa Tarozzo Kawasaki, Peter van Oosterom, and Rob van der Krogt  
**Script Author:** Maria Luisa Tarozzo Kawasaki

<img width="1920" height="1080" alt="Screenshot 2026-07-08 at 16 51 49" src="https://github.com/user-attachments/assets/092e9ac0-96bd-434f-beb7-c816d78116d7" />

## Repository Contents

* **3D Visualization Script (JavaScript):** Designed to be imported into Cesium Sandcastle, this script creates and visualizes UUDM-standardized 3D utilities, 3D buildings, and underground parcel geometries. The script can be adapted to represent other UUDM databases or parcels globally.
* **Spatial Claim Calculator (Python):** A script that evaluates the level of existing spatial congestion within a specific parcel (categorized as High, Medium, or Low). It calculates this by comparing the legal volume occupied by existing utilities against the total 3D volume of the underground parcel. Congestion is measured as a percentage (e.g., if utilities occupy 70% of the underground parcel, it is classified as high congestion).
* **3D Congestion Simulator (Python):** A sophisticated 3D voxel-based pathfinding script that simulates routing new utilities through dense underground networks. By applying real-world engineering constraints—such as orthogonal turn penalties and surface cover depth limits—it calculates precise, empirical Space Utilization Index (SUI) thresholds (e.g., proving that geometric locking creates "High" congestion at just 16.3% SUI).

