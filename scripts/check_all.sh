#!/usr/bin/env bash
# Python（正本）とブラウザ版（TS 移植）の両方のテストを実行する。
# cap/ は git 管理外で CI が無いため、物理を変えたら必ずこれを手元で実行すること。
set -euo pipefail
cd "$(dirname "$0")/.."
echo "== Python: pytest（ゴールデンの古さ検出・書き出しJSONの再現を含む）"
uv run pytest -q
echo "== Viewer: typecheck + vitest（Python ゴールデンとの照合を含む）"
(cd viewer && pnpm -s typecheck && pnpm -s test)
echo "OK: Python と TS の両方が通過"
