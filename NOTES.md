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

## Experiment log

| version | description | local score | LB |
|---|---|---|---|
| baselines | pid_water 0.0541, fixed_setpoint 0.0659, pid 0.0674, bang_bang 0.225 (30 train scenarios) | — | — |
