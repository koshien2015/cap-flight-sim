# cap-flight-sim

キャップ野球（ペットボトルキャップを投げる野球）の投球を計算する、6自由度（並進3＋回転3）のフライトシミュレーターです。研究・仮説検証用で、three.js 製の再生ビューア（`viewer/`）が付属します。

> **結果は仮定した空力係数モデルに依存する計算値です。** 現時点の係数はすべて `assumed`（仮定値）で、実測・CFD による裏付けはありません。出力された軌道を物理的事実として扱わないでください。

- 物理計算の正本は Python 側です。ビューアには、ブラウザ内で条件を変えて即時計算するための **TypeScript 移植版**があります（§10）。移植版は Python と同じ条件で照合テストし、浮動小数点誤差の範囲で一致することを確認しています。
- 物理計算は UI・動画解析・YOLO 推論から独立しています（依存は NumPy / SciPy だけ）。

---

## 1. セットアップと実行

```bash
cd cap-flight-sim
uv sync                      # Python 3.12 + 依存を .venv に導入（requires-python >= 3.11）
uv run pytest                # 単体・統合テスト

# 1球を計算（result.json / timeseries.csv / viewer.json / plots/*.png）
uv run python -m cap_flight_sim simulate --config examples/overhand_drop.yaml --output output/overhand_drop

# 比較実験（パラメータスイープ）
uv run python -m cap_flight_sim sweep --config experiments/A_reverse_spin.yaml --output output/A_reverse_spin

# 既存の結果を重ねて比較
uv run python -m cap_flight_sim compare \
  --runs output/A_reverse_spin/flat_ccw_from_above/result.json output/A_reverse_spin/flat_cw_from_above/result.json \
  --output output/comparison

# 係数フィッティング（実測が無い間は合成データで検証）
uv run python -m cap_flight_sim synthesize --config fitting/synthetic_truth_flick.yaml --output observed/synthetic_flick_train.csv --seed 1
uv run python -m cap_flight_sim fit --trajectory observed/synthetic_flick_train.csv \
  --test-trajectory observed/synthetic_flick_test.csv --config fitting/base.yaml --output output/fit_synthetic_flick

# 簡易モデルを係数表スキーマの CSV へ書き出す（CFD 交換形式の見本）
uv run python -m cap_flight_sim tabulate --config examples/overhand_drop.yaml --output examples/tables/example_coefficients.csv
```

ビューア:

```bash
cd viewer
pnpm install
pnpm dev          # http://localhost:5173 。public/samples/ のサンプルが読める
pnpm test         # Vitest（座標変換・SLERP・同期モード）
```

`simulate` / `sweep` の出力にある `viewer.json` は、ビューアの「viewer.json を追加」から読み込めます。複数読み込むと重ねて表示します。

---

## 2. 座標系と回転方向の定義

### ワールド座標（右手系）

| 軸 | 向き |
|---|---|
| `+X` | 投手 → 捕手 |
| `+Y` | **投手から見て左** |
| `+Z` | 鉛直上 |

- 地面は `Z = 0`、重力は `[0, 0, -g]`、捕手面は `X = 9.22 m` です（設定で変更できます）。
- 指示書原文は「`+Y` = 右」でしたが、この組み合わせ（前・右・上）は左手系になり、外積 `ω × v`（マグヌス力の向き）が左右逆に出ます。そのため右手系を保つよう `+Y` = 左に確定しました。
- **右へ曲がる変化は `-Y` の値として現れます。** CSV・グラフの `Y` も同じ規約です。
- 方位角 `launch_azimuth_deg` は `+Y`（左）向きを正とします。

### ボディ座標とクォータニオン

- `body +Z` はキャップ円形面の法線、`body X-Y` 面は円形面です。
- `surface_orientation` は `body +Z` が向く側を表し、`body_z_is_top`（天面＝閉じた側）か `body_z_is_cavity`（空洞側）のどちらかです。表裏非対称の係数（`cl0` や係数表の `surface_orientation` 列）はこの区別に従います。
- クォータニオンは `[w, x, y, z]`（スカラー先頭）で、ボディ座標からワールド座標への回転です（`v_world = R(q) v_body`）。
- scipy や three.js の `[x, y, z, w]` との並べ替えは `coordinates.wxyz_to_xyzw` / `xyzw_to_wxyz`（Python）と `viewer/src/domain/coordinate-transform.ts`（TS）だけで行います。
- オイラー角は入力と表示専用で、積分には使いません。
  - 並びは `[roll, pitch, yaw]` [deg]、内在的 Z-Y'-X''、つまり `R = Rz(yaw) Ry(pitch) Rx(roll)` です。
  - `+pitch` は `body +X` を下へ向ける回転です。例: `pitch = -5°` は前縁を上げる。

### 回転方向の入力

「時計回り」「反時計回り」という文字列だけでは入力できません。回転は次のいずれか1つの書式で指定します。

```yaml
angular_velocity: {omega_world_rad_s: [wx, wy, wz]}
angular_velocity: {omega_body_rad_s: [wx, wy, wz]}
angular_velocity: {rpm: 1200, axis_world: [0, 1, 0]}   # 右手の法則で軸方向が正
angular_velocity: {rpm: 1200, axis_body: [0, 0, 1]}
angular_velocity: {rpm: 1500, rotation_sense: clockwise, viewed_from_world: [0, 0, 1]}
#   viewed_from_world = キャップから観測者への方向（必須）。例は「真上から見て時計回り」= ω が -Z
```

ワールド座標で与えた角速度は、初期姿勢を使って `ω_body = Rᵀ ω_world` でボディ座標へ変換します。

| 回転 | FLU 右手系での ω（進行方向 `+X`） | マグヌス力 |
|---|---|---|
| バックスピン | `-Y` | 上（`+Z`） |
| トップスピン | `+Y` | 下（`-Z`） |
| 真上から見て反時計回り | `+Z` | 左（`+Y`） |

---

## 3. 物理モデル

### 運動方程式

```text
m dv/dt = F_gravity + F_drag + F_lift + F_magnus      （ワールド座標）
dr/dt   = v
I dω/dt + ω × (Iω) = M_aero                           （ボディ座標、オイラーの剛体方程式）
dq/dt   = ½ q ⊗ [0, ω_body] + k (1 − |q|²) q            （k = quaternion_drift_gain）
```

- 状態量は 13 要素で、`[r(3), v(3), q(4), ω_body(3)]` です。`r` は重心位置です。
- `solve_ivp` はステップの途中でクォータニオンを正規化できません。そのため次の3つを併用しています。
  1. 状態微分の中で `q` を正規化してから力を評価する
  2. ノルムを 1 へ引き戻す拘束安定化項 `k (1 − |q|²) q` を加える
  3. 出力時に `q` を正規化する

### 相対風・迎角の符号規約

- `u = v_cap − wind` は空気に対するキャップの速度で、`relative_air_velocity = wind − v_cap` の符号を反転したものです。動圧は `q = ½ ρ |u|²` です。
- **迎角** `α = atan2(−u_z, √(u_x² + u_y²))`（ボディ座標）、範囲は [−90°, 90°] です。
  - `α > 0` は、相対風が `body −Z` 側（法線と反対側の面）に当たる状態です。
  - 例: 面法線を上に向けて水平に投げたフリスビーが、やや下向きに進むとき `α > 0` です。
- **横滑り角** `β = atan2(u_y, u_x)` は、面内のどの方位へ進んでいるかを表します。軸対称のキャップでは β は自転とともに回るだけで、簡易モデルの空気力には効きません。
- **法線角**は面法線と相対風のなす角で、90° がエッジオン（キャップの縁から風を受ける状態）です。
- 速度がほぼゼロ（|u| < 1e-6 m/s）のときは空気力を 0 とし、NaN を出しません。

### 簡易係数モデル（`aerodynamics.model: simple`）

| 項 | 式 | 備考 |
|---|---|---|
| 抗力 | `F_D = −q A_ref Cd(α) û` | `projected_area`: `Cd = cd_projected · A_proj(α)/A_ref`、`A_proj = πR²|sin α| + D h |cos α|`。`constant` / `quadratic_aoa` も選択可 |
| 揚力 | `F_L = q A_ref Cl(α) ê_L`、`ê_L = unit(n − (n·û)û)` | 相対風に垂直で、面法線を含む平面内。`Cl = cl0 + sign(α)·f(|α|)`（奇関数部＋非対称項）。失速角以降は `decay`（90° で 0）か `hold` |
| マグヌス | `F_M = q A_ref C_M(S) unit(ω⊥ × u)`、`S = R|ω⊥|/|u|` | ω は速度方向成分を除いた ω⊥ だけを使う。`linear_spin_parameter` / `constant` |
| 空力中心のずれ | `(r_cp − r_com) × F` | `center_of_pressure_body_m: null` なら重心と一致（0） |
| ピッチ（姿勢安定化） | `q A_ref D Cm(α) ê_pitch`、`Cm = cm_pitch0 + cm_pitch_alpha·α` | `ê_pitch = ê_inplane × n`。正のモーメントで α が増える |
| 回転抗力 | `−½ρ|u| A_ref D R (c_spin ω_n + c_tumble ω_t)` | 面法線まわり（自転）と面内軸まわり（タンブリング）で別々の係数 |

- 面が相対風に正対して揚力方向が決まらない場合（`n ∥ û`）は、揚力を 0 として扱い、警告を出します。
- 抗力・揚力・マグヌス・モーメントは `aerodynamics.enable` で個別に ON/OFF できます。

**注意（簡易モデルの限界）**
- **キャップは球でも円柱でもありません。** マグヌス式は球・円柱の式を流用した暫定近似です。
- `projected_area` の抗力は、円筒の幾何投影面積による検証前の近似です。
- `reference_area_m2` を省略すると円形面積 πR² を使います。これは**エッジオン時の実効投影面積（D·h）とは別物**です。
- 慣性テンソルを省略すると `thin_walled_cup` 近似（天板の薄円盤＋側壁の薄肉円筒、一様シェル、質量は面積比で配分）で計算します。
  - この近似では重心が天板側へ約 2.5 mm ずれます。
  - 近似方法は出力 JSON の `assumptions` に記録されます。

### 係数表モデル（`aerodynamics.model: lookup_table`）

CFD・風洞・実測の結果は、この CSV / Parquet スキーマに変換して読み込みます。OpenFOAM などの固有形式はコアに持ち込みません（詳細は `cap_flight_sim/aerodynamics/lookup_table.py` の docstring）。

| 列の種類 | 列 |
|---|---|
| 軸（使うものだけ、完全な直交格子） | `speed_m_s`, `angle_of_attack_deg`, `sideslip_deg`, `spin_parameter` |
| カテゴリ（任意） | `surface_orientation` |
| 係数（無い列は 0 として警告） | `cd`, `cl`, `cy`, `cm_pitch`, `cm_roll`, `cm_yaw` |
| 出典（任意） | `source`（表全体で1値。混在はエラー） |

- 範囲内は多次元線形補間します。範囲外の扱いは `out_of_range` で `clamp` / `nearest` / `error` から選びます。
- 無警告の外挿はしません。範囲外に出たフレーム数は `warnings` に記録します。

---

## 4. 仮定した係数と出典区分

- 数値パラメータは `{value, source, note}` の形で書けます。`source` は `measured` / `fitted` / `literature` / `assumed` のいずれかです。
- source を省略した数値は `assumed` として記録されます。
- 全パラメータの出典は `result.json` の `parameter_source` に出ます。

`examples/_base_cap.yaml` の既定値（**すべて assumed**）:

| パラメータ | 値 | 根拠 |
|---|---|---|
| 質量 / 外径 / 高さ | 2.2 g / 30 mm / 15 mm | 一般的な PET キャップの例。**要実測** |
| `drag.cd_projected` | 1.0 | 平板・短円柱で Cd ≈ 1 程度という桁感 |
| `lift.cl_alpha_per_rad` | 1.4 /rad、失速 40° | フリスビーの文献値の桁を借用。キャップでは未検証 |
| `lift.cl0` | 0 | 表裏非対称の効果は未知 |
| `magnus.coefficient` | 1.0（C_M = S）、上限 0.5 | 球の低スピン域の桁を借用した暫定値 |
| モーメント係数・回転減衰 | 0 | 未知のため 0 から開始 |

---

## 5. 設定例

```yaml
extends: _base_cap.yaml            # 継承（姿勢指定・angular_velocity は丸ごと置換）
name: overhand_drop
label: virtual                     # good / bad / virtual（ビューアの表示切替）
initial_state:
  position_world_m: [0.0, 0.0, 1.45]
  speed_kmh: 58.0                  # または speed_m_s / velocity_world_m_s
  launch_azimuth_deg: 0.0          # + = 左
  launch_elevation_deg: 2.0
  face_normal_world: [0.0, 1.0, 0.0]   # または orientation_euler_deg / orientation_quaternion_wxyz
  angular_velocity: {rpm: 1200, axis_world: [0.0, 1.0, 0.0]}
simulation:
  max_step_sec: null               # null → min(回転周期/20, 1/(4·video_fps), 2 ms)
```

入力では km/h・rpm・度を使えます。読み込み時に SI 単位へ変換し、元の単位は `config.inputUnits` に残します。

| プリセット | 内容 |
|---|---|
| `ballistic_no_aero` | 空気力なしの放物運動（参照条件） |
| `flick_flat` | 弾き投げの例: 面をほぼ水平にし、真上から見て反時計回り 1500 rpm |
| `sidearm_flat` | サイドスローの例: 弾き投げと回転方向が逆という仮説を置いた条件 |
| `overhand_drop` | オーバースローの例: キャップを立ててエッジオン、トップスピン 1200 rpm |
| `spec_example_tumble` | 指示書 §12 の例をそのまま再現。面法線が捕手方向を向き、直径まわりのタンブリングになる |
| `lookup_table_example` | 係数表モデルの例 |

プリセットは編集用の例であり、物理的な正解値ではありません。

### 終了条件

- 地面到達（重心の `Z ≤ 0`。キャップの半径は無視）
- 捕手面到達（`X ≥ 9.22`）
- 最大計算時間
- 速度・角速度の異常
- NaN/Inf の発生（状態微分で検出し、直前までの結果を保持して `abnormal_non_finite` として記録）

捕手面到達時には次を `summary.catcherPlane` に記録します: 時刻、`Y, Z`、速度、姿勢、回転数と回転軸、各空気力、終了理由。

---

## 6. 数値積分

- 既定は `solver: solve_ivp`、`method: DOP853`、`rtol 1e-7`、`atol 1e-9` です。
  - scipy の OdeSolver クラス（solve_ivp が内部で使うもの）を直接ステップ実行しています。NaN が出たときに直前までの結果を残すためです。
  - イベント時刻は、各ステップの密出力を `brentq` で根探索して求めます。
- 最大ステップ幅の既定は `min(回転周期/20, 1/(4·video_fps), 2 ms)` です。1200 rpm・240 fps なら 1.04 ms になります。
- `solver: rk4` は固定刻みの古典 RK4 で、比較・デバッグ用です。同じ状態微分関数を使い、テストで solve_ivp と 1e-5 m 以内で一致することを確認しています。
- CSV・JSON の力の列は、出力時刻の状態で空力関数を**再評価**した値です。積分中の試行点の値ではありません。

---

## 7. 出力とグラフの読み方

| ファイル | 内容 |
|---|---|
| `result.json` | 設定（入力値と SI 値）、`parameter_source`、`aerodynamic_model`、`assumptions`、`warnings`、`simulation_version`、`coordinate_system`、`summary`、`events`、時系列 |
| `timeseries.csv` | 1時刻1行: 位置・速度・クォータニオン・roll/pitch/yaw（表示用）・ω（body と world）・rpm・面法線・迎角・横滑り角・抗力/揚力/マグヌス/合力・モーメント・cd/cl/cy |
| `viewer.json` | ビューア用（物理座標のまま保存） |
| `plots/01〜08` | 上面 X-Y、側面 X-Z、捕手面 Y-Z、速度、回転数、面法線、力の大きさ、迎角・横滑り角 |

比較指標（`comparison.csv`）は着弾点だけに頼らないように、次の値を出します。

- `initialLateralVelocityMS`: 初期の横速度（投射方向による横移動）
- `lateralBreakFromLaunchLineM`: **初速方向の直線からの横ずれ**（飛行中の横加速度による変化）。初期投射方向と混同しないよう分けています
- `meanAeroLateralAccelMS2`: 空気力による平均横加速度
- `dropFromLaunchLineM` / `verticalAeroEffectVsGravityOnlyM`: 落下量と、重力だけの場合との差
- `maxFaceNormalChangeDeg`: 姿勢変化

捕手面 Y-Z 図は捕手側から投手方向を見た図です。横軸を反転しているので、`+Y`（投手の左）が画面右に来ます。

---

## 8. 比較実験（`experiments/`）

| 実験 | 内容 |
|---|---|
| `A_reverse_spin` | 角速度ベクトルの符号だけを反転（面水平／エッジオン） |
| `B_face_orientation` | 面水平、前後に 5°・10° 傾斜、エッジオン、左右に 5°・10° 傾斜 |
| `C_spin_axis` | 上下・左右・進行方向の回転軸と、それぞれ 10°・20° 傾斜（揚力 OFF でマグヌス力を分離） |
| `D_spin_rate` | 0 / 300 / 1500 / 3000 rpm。ピッチモーメントを仮定した条件でジャイロ安定を確認 |
| `E_drag_lift_sensitivity` | Cd と cl0 の感度。捕手面へ届くのに必要な係数の桁を見る |

---

## 9. 実測値による検証方法（フィッティング）

観測 CSV の形式は `time_sec,x_m,y_m,z_m,confidence` です（ワールド座標・SI、リリース時刻 = 0）。

- 損失は信頼度で重み付けした位置誤差で、`Σ confidence_i ||p_sim(t_i) − p_obs_i||²` です。`scipy.optimize.least_squares` で境界付きに最小化します。
- 推定するパラメータは、ドット区切りのパスで任意に指定できます（`aerodynamics.drag.cd_projected`、`initial_state.speed_kmh` など）。
- 出力（`fit_report.json`）には推定値、標準誤差、相関行列、ヤコビアンの条件数、境界到達の有無、学習・評価それぞれの RMSE が含まれます。
- `fitted_config.yaml` には推定値が `source: fitted` として書き込まれます。
- 学習用（`--trajectory`）と評価用（`--test-trajectory`）の軌道を分けて指定できます。
- 次の場合は警告を出します: 1球だけのフィット、評価用データなし、合成データの使用、パラメータ間の高い相関（|r| > 0.95）、境界到達。

**限界**
- **2D 動画の軌道だけでは係数は一意に決まりません。** 抗力・揚力・マグヌスは同じ軌道を別の組み合わせで説明できることがあり、2D 動画から完全な 3D 姿勢も得られません。
- 1球にフィットした係数を一般則として扱わないでください。
- YOLO の画像座標から 3D ワールド座標への変換（カメラ校正）は、このパッケージの範囲外です。

実測データが無い現段階では、`synthesize` で作った**合成（仮想）データ**から真値を回復できるかを検証しています（`observed/` の CSV は1行目のコメントで synthetic と明示）。

---

## 10. three.js ビューアとブラウザ内計算（`viewer/`）

### 再生
- 構成は Vite + TypeScript + three.js（素の three.js）です。
- 物理座標から表示座標への変換は `src/domain/coordinate-transform.ts` の1か所だけで行います。
  - 変換は `three = (x, z, −y)` で、X 軸まわり −90° の真正回転（det = +1）です。
  - 姿勢は `q_three = q_M ⊗ q_phys ⊗ q_M⁻¹` で変換します。成分の並べ替えだけでは変換にならないことをテストで確認しています。
- シーク時、位置は線形補間、姿勢は SLERP で補間します。オイラー角の補間は使いません。
- 主な機能:
  - 再生・停止・シーク・再生速度・1フレーム送り
  - カメラ切替（上面・側面・捕手側・自由）
  - HUD（時刻・速度・回転数・迎角・横滑り角）
  - 色分けしたベクトル（速度 黄 / 抗力 赤 / 揚力 青 / マグヌス 紫 / 回転軸 緑 / 面法線 シアン）
  - 複数投球の重ね合わせと、良い球・抜け球・仮想条件の表示切替
  - 4つの同期モード（絶対時刻 / リリース基準 / X進行率 / 捕手面到達=100%）
- 表示上の誇張は右上に常時表示します（キャップの拡大率、矢印の倍率）。物理値は変えません。

### ブラウザ内計算（指示書 §14A からの方針変更）
指示書 §14A は「three.js 側で Python と別の空力式を実装しない」としていました。ブラウザだけで完結させるため、利用者の判断でこれを変更し、`viewer/src/physics/` に TypeScript 移植版を置いています。

- 範囲は簡易係数モデル（`aerodynamics.model: simple`）と固定刻み RK4 だけです。係数表モデル、スイープ、フィッティングは Python CLI で行います。
- 計算時間は 1球 25〜40 ms です。Web Worker で動かすので、スライダーを動かすと即座に再計算されます。
- 左パネル「条件を調整して計算」で次を変えられます。
  - 初速・仰角・方位・リリース高さ
  - 傾き（右傾き / 前縁上げ / 向き。定義は実験 B と同じ）
  - 回転数と回転軸（キャップ自身の軸 / world の各軸、逆回転）
  - 力の ON/OFF と主要係数（すべて assumed として書き出し）
- 「比較用に固定」で結果を残して重ねられます。ブラウザで計算した結果は、一覧に「ブラウザ計算」と表示されます。
- 「条件をJSONで保存」で出る JSON は、そのまま Python CLI で再現できます（`solver: rk4` なので結果は一致します）。

  ```bash
  uv run python -m cap_flight_sim simulate --config browser_xxx.json --output output/browser_xxx
  ```

### Python と TS の照合（物理を変えるときの手順）
1. Python 側を変更する。
2. `uv run python scripts/generate_ts_goldens.py` を実行し、ゴールデンを再生成する。
3. `viewer/src/physics/` を追従させる。
4. **`./scripts/check_all.sh` を実行し、両方のテストを通す。**

| テスト | 内容 |
|---|---|
| `viewer/tests/physics-parity.test.ts` | 状態微分・空気力の内訳（7バリエーション × 12状態）と RK4 軌道6ケースを照合。実測の差は位置 ~2e-15 m、到達時刻 ~3e-16 s |
| `tests/test_ts_goldens.py` | ゴールデンやプリセットの古さを検出する。UI が書き出した JSON を Python で再計算して同じ終了位置になることも確認 |
| `viewer/tests/export-roundtrip.test.ts` | UI の書き出し形式と、傾き定義が実験 B と一致すること |


## 11. 将来の CFD 連携

```text
実物キャップの3D形状取得
    ↓
OpenFOAM等の仮想風洞で条件スイープ
    ↓
力・モーメント係数をCSV/Parquet化（§3 の係数表スキーマ、source 列付き）
    ↓
LookupTableAerodynamicModelへ読み込み
    ↓
6DoF軌道シミュレーション
    ↓
実測軌道との比較・係数補正（fit）
```

CFD を実施していない係数を CFD 由来と表示しないでください。係数表の `source` 列には、実際の出典区分（`literature` / `measured` / `fitted` / `assumed`）だけを書きます。

---

## 12. ディレクトリ構成

```text
cap_flight_sim/
  cli.py  config.py  models.py  coordinates.py  rigid_body.py  integrators.py
  simulation.py  events.py  experiments.py
  aerodynamics/  base.py  simple.py  lookup_table.py  factory.py  tabulate.py
  fitting/       trajectory_loader.py  optimizer.py  losses.py
  outputs/       serializers.py  plots.py  summaries.py
examples/  experiments/  fitting/  observed/（合成データ）  tests/
viewer/src/  domain/  scene/  playback/  ui/
```
