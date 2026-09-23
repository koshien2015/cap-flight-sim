"""CLI の統合テスト（simulate / sweep / compare / synthesize / tabulate / fit）。"""

import json
from pathlib import Path

import pandas as pd
import yaml

from cap_flight_sim.cli import main

ROOT = Path(__file__).resolve().parents[1]


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def test_simulate_writes_bundle_and_plots(tmp_path):
    out = tmp_path / "run"
    assert main(["simulate", "--config", str(ROOT / "examples/ballistic_no_aero.yaml"), "--output", str(out)]) == 0
    doc = json.loads((out / "result.json").read_text())
    assert doc["summary"]["terminationReason"] == "catcher_plane"
    assert len(list((out / "plots").glob("*.png"))) == 8


def test_sweep_and_compare(tmp_path):
    exp = _write_yaml(tmp_path / "exp.yaml", {
        "name": "mini",
        "base_config": str(ROOT / "examples/flick_flat.yaml"),
        "base_overrides": {"simulation.duration_sec": 0.2},
        "variants": [
            {"name": "ccw", "overrides": {"initial_state.angular_velocity": {"rpm": 1500, "axis_world": [0, 0, 1]}}},
            {"name": "cw", "overrides": {"initial_state.angular_velocity": {"rpm": 1500, "axis_world": [0, 0, -1]}}},
        ],
    })
    out = tmp_path / "sweep"
    assert main(["sweep", "--config", str(exp), "--output", str(out)]) == 0
    table = pd.read_csv(out / "comparison.csv")
    ccw, cw = table.set_index("name").loc[["ccw", "cw"], "lateralBreakFromLaunchLineM"]
    assert ccw > 0 > cw
    assert main(["compare", "--runs", str(out / "ccw/result.json"), str(out / "cw/result.json"),
                 "--output", str(tmp_path / "cmp")]) == 0
    assert (tmp_path / "cmp/plots/01_top_xy.png").exists()


def test_tabulate_then_lookup_simulation(tmp_path):
    table = tmp_path / "coeffs.csv"
    assert main(["tabulate", "--config", str(ROOT / "examples/overhand_drop.yaml"), "--output", str(table)]) == 0
    cfg = _write_yaml(tmp_path / "lut.yaml", {
        "extends": str(ROOT / "examples/overhand_drop.yaml"),
        "aerodynamics": {"model": "lookup_table", "lookup_table": {"path": str(table), "out_of_range": "clamp"}},
    })
    assert main(["simulate", "--config", str(cfg), "--output", str(tmp_path / "lut"), "--no-plots"]) == 0
    simple = tmp_path / "simple"
    assert main(["simulate", "--config", str(ROOT / "examples/overhand_drop.yaml"), "--output", str(simple), "--no-plots"]) == 0
    a = json.loads((tmp_path / "lut/result.json").read_text())["summary"]["finalPositionWorldM"]
    b = json.loads((simple / "result.json").read_text())["summary"]["finalPositionWorldM"]
    assert abs(a[0] - b[0]) < 0.3  # 表の補間誤差の範囲で簡易モデルと整合


def test_synthesize_and_fit(tmp_path):
    truth = _write_yaml(tmp_path / "truth.yaml", {
        "extends": str(ROOT / "examples/flick_flat.yaml"),
        "simulation": {"duration_sec": 0.3},
        "aerodynamics": {"drag": {"cd_projected": 0.7}},
    })
    obs = tmp_path / "obs.csv"
    assert main(["synthesize", "--config", str(truth), "--output", str(obs), "--fps", "30", "--noise-m", "0.001"]) == 0
    fit_cfg = _write_yaml(tmp_path / "fit.yaml", {
        "base_config": str(ROOT / "examples/flick_flat.yaml"),
        "max_nfev": 15,
        "parameters": [{"path": "aerodynamics.drag.cd_projected", "initial": 1.0, "bounds": [0.1, 2.0]}],
    })
    out = tmp_path / "fit"
    assert main(["fit", "--trajectory", str(obs), "--config", str(fit_cfg), "--output", str(out)]) == 0
    report = json.loads((out / "fit_report.json").read_text())
    assert abs(report["fittedParameters"][0]["value"] - 0.7) < 0.05
    fitted = yaml.safe_load((out / "fitted_config.yaml").read_text())
    assert fitted["aerodynamics"]["drag"]["cd_projected"]["source"] == "fitted"


def test_cli_reports_config_error(tmp_path, capsys):
    bad = _write_yaml(tmp_path / "bad.yaml", {"cap": {"mass_kg": 0.002, "outer_diameter_m": 0.03, "height_m": 0.015}})
    assert main(["simulate", "--config", str(bad), "--output", str(tmp_path / "x")]) == 2
    assert "initial_state" in capsys.readouterr().err
