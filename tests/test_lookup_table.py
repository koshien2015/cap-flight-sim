import numpy as np
import pandas as pd
import pytest

from cap_flight_sim.aerodynamics.lookup_table import (
    LookupTableAerodynamicModel,
    OutOfTableRangeError,
    load_coefficient_table,
)


@pytest.fixture
def table_path(tmp_path):
    rows = []
    for aoa in (-10.0, 0.0, 10.0):
        for s in (0.0, 0.2):
            rows.append({"angle_of_attack_deg": aoa, "spin_parameter": s, "cd": 1.0 + 0.01 * aoa**2 / 10,
                         "cl": 0.05 * aoa, "cy": 2.0 * s, "source": "assumed"})
    path = tmp_path / "coeffs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_grid_points_match(table_path):
    table = load_coefficient_table(table_path)
    coeffs, out = table.lookup({"angle_of_attack_deg": 10.0, "spin_parameter": 0.2}, "error")
    assert not out
    assert coeffs["cl"] == pytest.approx(0.5)
    assert coeffs["cy"] == pytest.approx(0.4)
    assert coeffs["cm_pitch"] == 0.0  # 欠損係数は 0


def test_interpolates_between_grid_points(table_path):
    table = load_coefficient_table(table_path)
    coeffs, _ = table.lookup({"angle_of_attack_deg": 5.0, "spin_parameter": 0.1}, "error")
    assert coeffs["cl"] == pytest.approx(0.25)
    assert coeffs["cy"] == pytest.approx(0.2)


def test_out_of_range_policies(table_path):
    table = load_coefficient_table(table_path)
    point = {"angle_of_attack_deg": 30.0, "spin_parameter": 0.1}
    with pytest.raises(OutOfTableRangeError):
        table.lookup(point, "error")
    clamped, out = table.lookup(point, "clamp")
    assert out and clamped["cl"] == pytest.approx(0.5) and clamped["cy"] == pytest.approx(0.2)
    nearest, out = table.lookup(point, "nearest")
    assert out and nearest["cl"] == pytest.approx(0.5) and nearest["cy"] in (pytest.approx(0.0), pytest.approx(0.4))


def test_rejects_incomplete_grid(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame([{"angle_of_attack_deg": 0, "spin_parameter": 0, "cd": 1},
                  {"angle_of_attack_deg": 10, "spin_parameter": 0, "cd": 1},
                  {"angle_of_attack_deg": 0, "spin_parameter": 0.2, "cd": 1}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="rectangular"):
        load_coefficient_table(path)


def test_rejects_mixed_sources(tmp_path):
    path = tmp_path / "mixed.csv"
    pd.DataFrame([{"angle_of_attack_deg": 0, "cd": 1, "source": "measured"},
                  {"angle_of_attack_deg": 10, "cd": 1, "source": "assumed"}]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="mixed"):
        load_coefficient_table(path)


def test_surface_orientation_filter(tmp_path):
    path = tmp_path / "surf.csv"
    rows = [{"angle_of_attack_deg": a, "surface_orientation": s, "cl": (1.0 if s == "body_z_is_top" else -1.0)}
            for a in (0, 10) for s in ("body_z_is_top", "body_z_is_cavity")]
    pd.DataFrame(rows).to_csv(path, index=False)
    top = load_coefficient_table(path, "body_z_is_top")
    cav = load_coefficient_table(path, "body_z_is_cavity")
    assert top.lookup({"angle_of_attack_deg": 5}, "error")[0]["cl"] == pytest.approx(1.0)
    assert cav.lookup({"angle_of_attack_deg": 5}, "error")[0]["cl"] == pytest.approx(-1.0)


def test_lookup_model_in_simulation_counts_out_of_range(table_path, base_raw):
    from cap_flight_sim.config import config_from_dict
    from cap_flight_sim.simulation import run_simulation

    base_raw["aerodynamics"] = {"model": "lookup_table", "lookup_table": {"path": str(table_path), "out_of_range": "clamp"}}
    base_raw["initial_state"]["angular_velocity"] = {"rpm": 3000, "axis_body": [0, 0, 1]}
    result = run_simulation(config_from_dict(base_raw))
    assert result.model_description["coefficient_source"] == "assumed"
    assert all(np.all(np.isfinite(f.state.position_world_m)) for f in result.frames)
    assert any("範囲外" in w for w in result.warnings)  # spin_parameter が 0.2 を超える
    assert isinstance(LookupTableAerodynamicModel, type)
