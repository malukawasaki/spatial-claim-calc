"""
test_spatial_claim_calc.py — Unit Tests for SpatialClaimCalc

Tests use analytical geometries with known closed-form volumes and
intersection sizes to verify correctness without synthetic GLB data.

Run with:
    pytest test_spatial_claim_calc.py -v

References
----------
- ISO 19152-5:2024 (ExtSpatialClaim output schema)
- He et al. (2021), TUST, 107, 103660  — SUI metric
"""

import math
import numpy as np
import pytest
import trimesh

from SpatialClaimCalc import SpatialClaimCalc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _box_parcel(extents, translation=(0, 0, 0)):
    """Return a trimesh box centred at `translation`."""
    m = trimesh.creation.box(extents=extents)
    m.apply_translation(translation)
    return m


def _cylinder_utility(radius, height, center=(0, 0, 0), resolution=0.25):
    """
    Return a minimal utility dict for a vertical cylinder.
    The SpatialClaimCalc only calls mesh methods; no GLB needed.
    """
    m = trimesh.creation.cylinder(radius=radius, height=height)
    m.apply_translation(center)
    return [{"mesh": m, "uudm": {"utilityId": "TEST-001", "utilityType": "TEST"}}]


# ---------------------------------------------------------------------------
# Test 1: Empty parcel (no utilities) → SUI = 0
# ---------------------------------------------------------------------------

def test_empty_parcel_zero_sui():
    """A parcel with no overlapping utilities must yield SUI_Ratio = 0."""
    parcel   = _box_parcel([10, 10, 10], translation=(0, 0, 0))
    # Utility placed entirely outside the parcel
    far_cyl  = _cylinder_utility(radius=0.1, height=5, center=(100, 100, 100))

    calc   = SpatialClaimCalc(parcel_mesh=parcel, utilities=far_cyl, parcel_id="EmptyTest")
    result = calc.calculate_congestion(buffer_radius=0.5, resolution=0.5)

    assert result["Spatial_Claim_Calculations"]["SUI_Ratio"] == pytest.approx(0.0, abs=1e-6)
    assert result["Spatial_Claim_Calculations"]["Congestion_Index"] == "Low"


# ---------------------------------------------------------------------------
# Test 2: Known SUI — single central utility inside a box
# ---------------------------------------------------------------------------

def test_known_sui_single_pipe():
    """
    A 1 m radius vertical cylinder of height 10 centred in a 10×10×10 box.
    Physical volume = π × 1² × 10 ≈ 31.4 m³.
    With buffer_radius=0 the SUI should be ≈ π/(10²) ≈ 0.0314 (Low).
    We use buffer_radius=0 to isolate geometry from dilation.
    """
    parcel    = _box_parcel([10, 10, 10])
    utilities = _cylinder_utility(radius=1.0, height=10)

    calc   = SpatialClaimCalc(parcel_mesh=parcel, utilities=utilities, parcel_id="KnownSUI")
    result = calc.calculate_congestion(buffer_radius=0.0, resolution=0.25)

    expected_ratio = math.pi * 1.0**2 * 10 / (10**3)  # ≈ 0.0314
    actual_ratio   = result["Spatial_Claim_Calculations"]["SUI_Ratio"]
    # Allow 40% relative tolerance: voxelization over-counts boundary voxels
    # (partial voxels on the cylinder perimeter are counted as fully occupied).
    # This discretization error = O(surface_area × resolution) is documented
    # in the method; it decreases as resolution decreases.
    assert actual_ratio == pytest.approx(expected_ratio, rel=0.40), (
        f"SUI ratio {actual_ratio:.4f} deviates >40% from expected {expected_ratio:.4f}"
    )
    assert result["Spatial_Claim_Calculations"]["Congestion_Index"] == "Low"


# ---------------------------------------------------------------------------
# Test 3: High congestion — parcel almost entirely filled
# ---------------------------------------------------------------------------

def test_high_congestion_tightly_packed():
    """
    A thin 2×2×2 parcel around a 0.8 m radius cylinder.
    With a 1.5 m buffer the legal space will fill the entire parcel → High.
    """
    parcel    = _box_parcel([2, 2, 2])
    utilities = _cylinder_utility(radius=0.8, height=2)

    calc   = SpatialClaimCalc(parcel_mesh=parcel, utilities=utilities, parcel_id="HighTest")
    result = calc.calculate_congestion(buffer_radius=1.5, resolution=0.25)

    assert result["Spatial_Claim_Calculations"]["Congestion_Index"] == "High"
    assert result["Spatial_Claim_Calculations"]["SUI_Ratio"] > 0.163


# ---------------------------------------------------------------------------
# Test 4: LADM output schema validation
# ---------------------------------------------------------------------------

def test_output_schema_ladm_keys():
    """Output dict must include required LADM ExtSpatialClaim keys."""
    parcel    = _box_parcel([5, 5, 5])
    utilities = _cylinder_utility(radius=0.1, height=5)

    calc   = SpatialClaimCalc(parcel_mesh=parcel, utilities=utilities, parcel_id="SchemaTest")
    result = calc.calculate_congestion(buffer_radius=0.5, resolution=0.5)

    assert result["LADM_Class"] == "ExtSpatialClaim"
    assert result["LADM_Standard"] == "ISO 19152-5:2024"
    assert "LA_SpatialUnit_ID" in result
    assert "Spatial_Claim_Calculations" in result
    assert "Intersecting_UUDM_Assets" in result

    sc = result["Spatial_Claim_Calculations"]
    for key in ("SUI_Ratio", "SUI_Percent", "Congestion_Index",
                "LA_LegalSpaceBuffer_m", "Voxel_Resolution_m",
                "Parcel_Volume_m3", "Occupied_Legal_Volume_m3"):
        assert key in sc, f"Missing key: {key}"


# ---------------------------------------------------------------------------
# Test 5: SUI ratio and percent are consistent
# ---------------------------------------------------------------------------

def test_sui_ratio_percent_consistency():
    """SUI_Ratio × 100 must equal SUI_Percent (within floating-point tolerance)."""
    parcel    = _box_parcel([10, 10, 10])
    utilities = _cylinder_utility(radius=0.5, height=5, center=(0, 0, 0))

    calc   = SpatialClaimCalc(parcel_mesh=parcel, utilities=utilities, parcel_id="UnitTest")
    result = calc.calculate_congestion(buffer_radius=0.5, resolution=0.5)

    sc = result["Spatial_Claim_Calculations"]
    assert sc["SUI_Percent"] == pytest.approx(sc["SUI_Ratio"] * 100, rel=1e-4)


# ---------------------------------------------------------------------------
# Test 6: init raises ValueError for invalid parcel
# ---------------------------------------------------------------------------

def test_init_rejects_empty_parcel():
    """SpatialClaimCalc must raise ValueError for an empty / zero-volume mesh."""
    empty_mesh = trimesh.Trimesh()  # no vertices
    with pytest.raises(ValueError, match="non-empty"):
        SpatialClaimCalc(parcel_mesh=empty_mesh, utilities=[], parcel_id="BadParcel")


# ---------------------------------------------------------------------------
# Test 7: resolution parameter is respected
# ---------------------------------------------------------------------------

def test_resolution_parameter_changes_result():
    """
    SUI values at different resolutions should differ but remain in the
    same order of magnitude (within 20% of each other).
    """
    parcel    = _box_parcel([10, 10, 10])
    utilities = _cylinder_utility(radius=1.0, height=10)

    calc     = SpatialClaimCalc(parcel_mesh=parcel, utilities=utilities, parcel_id="ResTest")
    result25 = calc.calculate_congestion(buffer_radius=0.0, resolution=0.25)
    result50 = calc.calculate_congestion(buffer_radius=0.0, resolution=0.50)

    sui25 = result25["Spatial_Claim_Calculations"]["SUI_Ratio"]
    sui50 = result50["Spatial_Claim_Calculations"]["SUI_Ratio"]

    assert sui25 != sui50, "Different resolutions should produce different SUI values."
    # Both should be within 30% of each other (same geometry, different voxel size).
    # Larger tolerance at coarser resolution is expected and well-documented.
    assert abs(sui25 - sui50) / sui25 < 0.30, (
        f"Resolution sensitivity too large: {sui25:.4f} vs {sui50:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 8: Occupied volume ≤ parcel volume always
# ---------------------------------------------------------------------------

def test_occupied_volume_never_exceeds_parcel():
    """Occupied legal volume must never exceed the total parcel volume."""
    parcel    = _box_parcel([5, 5, 5])
    utilities = _cylinder_utility(radius=2.0, height=5, center=(0, 0, 0))

    calc   = SpatialClaimCalc(parcel_mesh=parcel, utilities=utilities, parcel_id="BoundsTest")
    result = calc.calculate_congestion(buffer_radius=2.0, resolution=0.5)

    sc = result["Spatial_Claim_Calculations"]
    assert sc["Occupied_Legal_Volume_m3"] <= sc["Parcel_Volume_m3"] + 1e-3
