"""パラメータスイープと複数結果の比較。

実験設定（YAML）:
    name: reverse_spin
    description: ...
    base_config: ../examples/flick_flat.yaml   # 実験ファイルからの相対パス
    base_overrides: {simulation.duration_sec: 2.0}  # 全条件共通（任意）
    variants:                                   # 個別条件（任意）
      - name: ccw
        label: virtual
        overrides: {initial_state.angular_velocity.rotation_sense: counterclockwise}
    grid:                                       # 直積スイープ（任意）
      initial_state.angular_velocity.rpm: [0, 600, 1200]
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any

import pandas as pd

from .config import apply_overrides, config_from_dict, load_raw_config, load_yaml
from .outputs.plots import plot_runs
from .outputs.serializers import build_result_document, load_result_document, write_json, write_result_bundle
from .simulation import run_simulation

COMPARISON_METRICS = (
    "terminationReason",
    "reachedCatcherPlane",
    "flightTimeSec",
    "initialLateralVelocityMS",
    "meanAeroLateralAccelMS2",
    "meanAeroVerticalAccelMS2",
    "maxAbsLateralDisplacementM",
    "lateralBreakFromLaunchLineM",
    "maxAbsLateralBreakM",
    "dropFromLaunchLineM",
    "verticalAeroEffectVsGravityOnlyM",
    "faceNormalChangeDeg",
    "maxFaceNormalChangeDeg",
    "rpmInitial",
    "rpmFinal",
    "meanDragN",
    "meanLiftN",
    "meanMagnusN",
)


def expand_variants(experiment: dict[str, Any]) -> list[dict[str, Any]]:
    variants = [dict(v) for v in experiment.get("variants", []) or []]
    grid = experiment.get("grid") or {}
    if grid:
        keys = list(grid)
        for combo in itertools.product(*(grid[k] for k in keys)):
            overrides = dict(zip(keys, combo))
            name = "__".join(f"{k.split('.')[-1]}={v}" for k, v in overrides.items())
            variants.append({"name": name, "overrides": overrides})
    if not variants:
        raise ValueError("experiment needs variants or grid")
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate variant names: {names}")
    return variants


def run_sweep(experiment_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    exp_path = Path(experiment_path)
    experiment = load_yaml(exp_path)
    base_raw = load_raw_config((exp_path.parent / experiment["base_config"]).resolve())
    base_raw = apply_overrides(base_raw, experiment.get("base_overrides", {}) or {})
    out = Path(output_dir)
    docs = []
    for variant in expand_variants(experiment):
        raw = apply_overrides(base_raw, variant.get("overrides", {}) or {})
        raw = {**raw, "name": variant["name"], "label": variant.get("label", raw.get("label", "virtual"))}
        result = run_simulation(config_from_dict(raw, name=variant["name"]))
        write_result_bundle(result, out / variant["name"])
        docs.append(build_result_document(result))
    table = comparison_table(docs)
    table.to_csv(out / "comparison.csv", index=False)
    plot_runs(docs, out / "plots")
    write_json(
        out / "experiment.json",
        {"experiment": experiment, "runs": [d["name"] for d in docs], "comparison": table.to_dict(orient="records")},
    )
    return table


def comparison_table(docs: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for d in docs:
        s = d["summary"]
        cp = s.get("catcherPlane") or {}
        rows.append(
            {
                "name": d["name"],
                "label": d.get("label"),
                **{m: s.get(m) for m in COMPARISON_METRICS},
                "catcherY_m": cp.get("yM"),
                "catcherZ_m": cp.get("zM"),
                "catcherTimeSec": cp.get("timeSec"),
                "warnings": " | ".join(d.get("warnings", [])),
            }
        )
    return pd.DataFrame(rows)


def compare_runs(result_paths: list[str | Path], output_dir: str | Path) -> pd.DataFrame:
    docs = [load_result_document(p) for p in result_paths]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = comparison_table(docs)
    table.to_csv(out / "comparison.csv", index=False)
    plot_runs(docs, out / "plots")
    return table
