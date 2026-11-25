# NeurIPS 2026 Smart Buildings — experiment log

Target: Kaggle LB <= 0.0088 (top 10). Score = mean over scenarios of -(episode return / 8640); lower is better.

## Findings about the problem (2026-11-25)

- **Scale check.** `kaggle_evaluation/gateway.py` writes one row per scenario holding the mean per-step
  reward, scored by MAE against 0. That is the same number `score_leaderboard.py` prints, so local and
  LB scores are directly comparable. The simulator shipped in the competition files is identical to
  the starter kit.
- **Why leaders are ~6x better than the baselines.**
  1. Comfort regret is occupancy-weighted and normalised by total occupancy; with nobody in the building
     it is exactly zero. Occupants are only present on workdays between 06:00 and 19:00, so the plant can
     be fully off for ~61% of all steps. The baselines condition the building all night and all weekend.
  2. The baselines run fans at high static pressure (fan power = flow x pressure) and reheat at the same
     time as they over-ventilate. `pid_water` on building_train_1/fairbanks_1 averages ~70 kW fan and
     ~480 kW gas; the physically required load is a small fraction of that.
  3. The buildings have large thermal inertia (plant off: 0.3-0.5 K/h drift in the worst zones).
- **Reward economics** (per AHU of normalisation): 1 kW electricity costs ~2.1x 1 kW of gas.
  Fan power ~ flow x pressure x (2 - recirc) / 0.9; cooling = air-side load / 3.5; gas = coil heat.
- **Plant facts used by the controller.** AHU can only cool. Reheat is water-limited:
  q = valve x 125.4 x sqrt(dP_bar) x (T_water - T_supply) W, capped by a 35 C discharge limit.
  Zone airflow = damper x 2.0 x sqrt(P/2000) m3/s, damper floor 0.1, pressure floor 200 Pa, recirc <= 0.7.
- **Zone thermal model.** Each zone has a fast air node (~12 W/K per m2 of capacity per 5-min step)
  coupled to a slow mass node (5 W/K per m2, time constant ~5 h) and to wall cells (~2.8 W/K per wall
  face). One-step model: T+ = (Cdt*T + W + Q_reheat + Cs*Ts) / (K0 + Cs), with W a slow disturbance.
  Fitted from traces: Cdt ~= 11.8-12.0 x area, K0 - Cdt ~= 4.5 x area + 2.5-3.0 x faces.
  Zone areas come from connected components of `zone_map` (matches the simulator's room numbering).
- **Sensor lag.** The zone temperature in the observation is latched before the thermal solve, so it is
  the true temperature one step ago. A deadbeat controller that ignores this oscillates by +-1.5 K.
- Holidays: the comfort schedule has no holidays but the occupancy model does (US federal holidays),
  so on a holiday the band is "comfort" but nobody arrives. Detected from zero occupancy after 06:30.

- **Gateway serialization scrambles dict order.** `ahu_zone_assignments` arrives with its keys in a
  different order than locally (e.g. ahu_4, ahu_3, ...), while `temp_indices` etc. stay positional in
  AHU-number order. A policy that maps AHU index to dict order silently drives the wrong zones
  (600-step test: -1.41 through the gateway vs -0.31 direct). Fixed by sorting AHU ids by number;
  the local gateway test (`kaggle/submission_local.py`) now reproduces the direct runner bit-for-bit.
- Output file of the notebook is `submission.parquet`. Daily limit: 5 submissions.
- Kaggle runtime: 8641 steps of building_train_1/atlanta_2 in 760 s (88 ms/step), policy ~1 ms/step.

## Local tooling

- `tools/sbrun.py` one scenario, reward/energy breakdown, optional trace. `tools/run_batch.py` runs a
  scenario set 4-wide and caches per-scenario JSON under `results/<tag>/`.
- `tools/make_val.py` builds the held-out validation set: 12 new buildings (seeds 601-613, sized like
  the shipped six) x {one Minneapolis strip, one strip unused by the training set} = 24 scenarios.
- `tools/make_notebook.py` embeds `policy/sb_policy.py` into `kaggle/<slug>.ipynb`.
  Local gateway test: `SB_COMPETITION_DATA_FOLDER=<competition files>/ python kaggle/submission_local.py`.

## Experiment log

| version | description | local score | LB |
|---|---|---|---|
| baselines | pid_water 0.0541, fixed_setpoint 0.0659, pid 0.0674, bang_bang 0.225 (30 train scenarios) | — | — |
| v2 | per-zone one-step model + observer, per-AHU cost optimiser, fixed 3-step pre-start (sensor lag handled) | first 4 train scenarios: atlanta_2 0.00286, denver_7 0.00177, fairbanks_1 0.06278, phoenix_5 0.01444 | not submitted |
| v3 | + optimal start from online slow-state model (theta = W/Kss), pre-start pressure cap 400 Pa, finer economiser candidates | same 4: 0.00281 / 0.00172 / 0.04410 / 0.01413 (full table below when the run completes) | submitted 2026-11-25 01:17 UTC (kernel v1, ref 58859225), pending |
