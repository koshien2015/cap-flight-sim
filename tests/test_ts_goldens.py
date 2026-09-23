"""ブラウザ版（TS 移植）との整合を Python 側から守るテスト。

- ゴールデンの古さ検出: Python の物理を変えたのに scripts/generate_ts_goldens.py を再実行していないと失敗する。
  失敗したら再生成し、viewer 側で `pnpm test` を通すこと（TS 側の追従が必要）。
- UI 書き出し JSON の再現: ブラウザが書き出した条件を Python CLI で計算し、TS と同じ終了位置になることを確認する。
"""

import importlib.util
import json
from pathlib import Path

import numpy as np

from cap_flight_sim.config import config_from_dict
from cap_flight_sim.simulation import run_simulation

ROOT = Path(__file__).resolve().parents[1]


def _generator():
    spec = importlib.util.spec_from_file_location("generate_ts_goldens", ROOT / "scripts/generate_ts_goldens.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ts_goldens_are_up_to_date():
    gen = _generator()
    stale = [
        str(path.relative_to(ROOT))
        for path, data in gen.build_all().items()
        if not path.exists() or path.read_text(encoding="utf-8") != gen.render(data)
    ]
    assert not stale, f"ゴールデンが古い: {stale} → uv run python scripts/generate_ts_goldens.py を実行し、viewer で pnpm test"


def test_browser_exported_config_reproduces_in_python():
    fixture = json.loads((ROOT / "viewer/tests/fixtures/exported-config.json").read_text(encoding="utf-8"))
    result = run_simulation(config_from_dict(fixture["config"]))
    expected = fixture["expected"]
    assert result.termination_reason == expected["terminationReason"]
    final = result.frames[-1].state
    assert abs(final.time_sec - expected["finalTime"]) < 1e-9
    assert np.allclose(final.position_world_m, expected["finalPosition"], atol=1e-8)
