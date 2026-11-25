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
| v3 | + optimal start from online slow-state model (theta = W/Kss), pre-start pressure cap 400 Pa, finer economiser candidates | **0.00825** (30 train scenarios, table below) | submitted 2026-11-25 01:17 UTC (kernel v1, ref 58859225), pending |

### v3 on the 30 training scenarios (mean 0.00825)

| scenario | score | comfort | cost | carbon | fan kW | cooling kW | gas kW |
|---|---|---|---|---|---|---|---|
| building_train_1/atlanta_2 | 0.00281 | -0.00000 | 0.00131 | 0.00149 | 0.75 | 1.19 | 6.3 |
| building_train_1/denver_7 | 0.00172 | -0.00000 | 0.00072 | 0.00100 | 0.51 | 0.42 | 4.4 |
| building_train_1/fairbanks_1 | 0.04410 | 0.00014 | 0.01257 | 0.03139 | 2.09 | 0.00 | 153.8 |
| building_train_1/phoenix_5 | 0.01413 | -0.00000 | 0.01149 | 0.00264 | 1.33 | 22.86 | 0.0 |
| building_train_1/singapore_8 | 0.00617 | -0.00000 | 0.00495 | 0.00123 | 0.69 | 10.45 | 0.0 |
| building_train_2/atlanta_3 | 0.00280 | 0.00000 | 0.00223 | 0.00057 | 1.66 | 8.81 | 0.6 |
| building_train_2/denver_8 | 0.00371 | 0.00001 | 0.00119 | 0.00252 | 1.30 | 0.40 | 27.0 |
| building_train_2/fairbanks_2 | 0.01059 | 0.00000 | 0.00298 | 0.00760 | 1.66 | 0.16 | 83.3 |
| building_train_2/phoenix_6 | 0.00703 | -0.00000 | 0.00571 | 0.00132 | 1.57 | 25.67 | 0.0 |
| building_train_2/singapore_1 | 0.00464 | -0.00000 | 0.00373 | 0.00091 | 1.20 | 17.40 | 0.0 |
| building_train_4/atlanta_5 | 0.00633 | -0.00000 | 0.00514 | 0.00119 | 1.02 | 15.17 | 0.0 |
| building_train_4/denver_2 | 0.00495 | -0.00000 | 0.00144 | 0.00351 | 0.78 | 0.15 | 25.5 |
| building_train_4/fairbanks_4 | 0.00123 | -0.00000 | 0.00058 | 0.00065 | 1.03 | 0.25 | 4.1 |
| building_train_4/phoenix_8 | 0.00134 | -0.00000 | 0.00075 | 0.00059 | 1.14 | 0.79 | 3.3 |
| building_train_4/singapore_3 | 0.00658 | -0.00000 | 0.00535 | 0.00123 | 1.02 | 15.82 | 0.0 |
| building_train_5/atlanta_6 | 0.00407 | -0.00000 | 0.00330 | 0.00077 | 1.10 | 12.81 | 0.0 |
| building_train_5/denver_3 | 0.00347 | 0.00001 | 0.00107 | 0.00240 | 0.95 | 0.53 | 22.9 |
| building_train_5/fairbanks_5 | 0.00140 | -0.00000 | 0.00108 | 0.00032 | 1.54 | 2.87 | 0.8 |
| building_train_5/phoenix_1 | 0.00207 | 0.00000 | 0.00080 | 0.00126 | 1.07 | 0.64 | 11.6 |
| building_train_5/singapore_4 | 0.00526 | -0.00000 | 0.00427 | 0.00099 | 1.07 | 16.94 | 0.0 |
| building_train_7/atlanta_8 | 0.01657 | 0.01120 | 0.00158 | 0.00380 | 0.52 | 0.09 | 18.4 |
| building_train_7/denver_5 | 0.01339 | 0.00943 | 0.00318 | 0.00079 | 0.52 | 6.09 | 0.3 |
| building_train_7/fairbanks_7 | 0.01015 | 0.00309 | 0.00185 | 0.00520 | 0.52 | 0.06 | 25.4 |
| building_train_7/phoenix_3 | 0.01486 | 0.00967 | 0.00421 | 0.00098 | 0.62 | 8.20 | 0.0 |
| building_train_7/singapore_6 | 0.01504 | 0.00981 | 0.00424 | 0.00099 | 0.54 | 8.44 | 0.0 |
| building_train_8/atlanta_1 | 0.00415 | -0.00000 | 0.00128 | 0.00287 | 0.39 | 0.08 | 10.3 |
| building_train_8/denver_6 | 0.00310 | -0.00000 | 0.00245 | 0.00065 | 0.48 | 3.33 | 0.3 |
| building_train_8/fairbanks_8 | 0.02111 | 0.00000 | 0.00563 | 0.01548 | 0.83 | 0.00 | 56.8 |
| building_train_8/phoenix_4 | 0.00940 | -0.00000 | 0.00764 | 0.00176 | 0.68 | 11.42 | 0.0 |
| building_train_8/singapore_7 | 0.00523 | -0.00000 | 0.00423 | 0.00100 | 0.43 | 6.41 | 0.0 |

By station: atlanta 0.0061, denver 0.0051, fairbanks 0.0148, phoenix 0.0081, singapore 0.0072.

**Finding: false holiday detection (building_train_7, all five climates, comfort ~0.01 each).**
The observed `num_occupants` is the reward's occupancy lagged one step. v3 declared a holiday after 6
comfort steps with zero occupancy and never revoked it. With one occupant per zone and 36 zones, the first
arrival is occasionally later than 06:30 (probability ~e^-6 per day); the occupancy RNG seed is fixed, so
the same workday went wrong in every building_train_7 scenario and the plant stayed off all day (one day
costs ~85 reward units = 0.0098 on the scenario score). Fix (v4): revoke the holiday flag as soon as any
occupant is seen, detect after 4 steps. The pending v3 submission carries this risk for small buildings.

**Finding: night flush with outside air is a net loss.** Tested on building_train_7/denver_5 (summer,
cool nights): compressor energy fell 20% but fan energy doubled and morning reheat appeared; the comfort
band already gives free storage and the building cools through the envelope at night anyway. Also triggered
wrongly in winter Atlanta. Left in the code, disabled (`flush_on=0`).

**Finding: deep-winter heating is ventilation-bound.** In Fairbanks at -35 C about 57% of the gas heats
the mandatory 30% outside air: net zone heat must be delivered at <= 35 C discharge, so
gas = net heat x (308.15 - T_supply) / (308.15 - T_zone). Nothing in the action space avoids this, so
building_train_1/fairbanks_1 (0.044) is close to its physical floor.

**Finding: per-zone occupancy is observable from heat signatures (v5).**
`num_occupants` in the observation equals the reward's total occupancy one step earlier. Each occupant adds
70 W of sensible heat to their zone, and the zone model's one-step disturbance measurement resolves that
cleanly: afternoon innovation noise is 1-10 W per zone against a 65-67 W step, appearing exactly one policy
call after the building-wide count changes. Arrivals stop at 12:59 and nobody returns after leaving, so a
zone whose occupants have all been seen leaving is empty for the rest of the day and its comfort no longer
counts. v5 counts departures per zone (only unambiguous signatures; any unexplained jump disqualifies the
zone for the day) and widens the band of fully vacated zones by up to 3 K, ramped at 0.25 K/step.
- Offline tracker on 8-workday traces: 0-4 false released zone-steps out of ~5,500-8,000 released.
- First closed-loop test had one false release: a neighbour that had just been released cooled 1.6 K in one
  step and pulled ~90 W through the doorway (door cells = air in `floor_plan` but wall in `zone_map`).
  Fixed by ramping the release, compensating doorway exchange (15 W/K per door cell) and refusing to
  attribute when there are more signatures than departures.
- Short closed-loop tests after the fix: building_train_8/phoenix_4 (5 days) 0.009826 -> 0.009391 (-4.4%),
  building_train_8/fairbanks_8 (7 days) 0.014177 -> 0.013837 (-2.4%), no false releases, ~70% of empty
  zone-time captured.
- Bug caught on the way: release flags must be cleared before the 05:55 action of the next day.

**Finding: banking (pulsing an air handler so it can rest on alternate steps) is worth <1%** and adds
model-excitation noise that would hurt the occupancy tracker; disabled (`bank_max_n=0`).

| v4 | v3 + holiday recovery + banking | first 5 train scenarios within 1% of v3 (run stopped early) | not submitted |
| v5 | v3 + holiday recovery (4 steps, revoked when anyone is seen) + release of vacated zones + margins 0.10 K + max lead 12 h | **0.00667** (30 train; every scenario better than v3; 2 false released zone-steps in total) | see below |

### v5 on the 30 training scenarios (mean 0.00667; v3 was 0.00825)

| scenario | v5 | v3 | v5 comfort | fan kW | cooling kW | gas kW |
|---|---|---|---|---|---|---|
| building_train_1/atlanta_2 | 0.00268 | 0.00281 | -0.00000 | 0.73 | 1.09 | 6.1 |
| building_train_1/denver_7 | 0.00162 | 0.00172 | -0.00000 | 0.49 | 0.40 | 4.2 |
| building_train_1/fairbanks_1 | 0.04357 | 0.04410 | -0.00000 | 1.96 | 0.00 | 152.5 |
| building_train_1/phoenix_5 | 0.01367 | 0.01413 | -0.00000 | 1.31 | 22.13 | 0.0 |
| building_train_1/singapore_8 | 0.00605 | 0.00617 | 0.00000 | 0.72 | 10.20 | 0.0 |
| building_train_2/atlanta_3 | 0.00271 | 0.00280 | 0.00000 | 1.72 | 8.42 | 0.6 |
| building_train_2/denver_8 | 0.00356 | 0.00371 | 0.00001 | 1.27 | 0.36 | 25.8 |
| building_train_2/fairbanks_2 | 0.01040 | 0.01059 | 0.00000 | 1.66 | 0.13 | 81.7 |
| building_train_2/phoenix_6 | 0.00685 | 0.00703 | -0.00000 | 1.57 | 25.02 | 0.0 |
| building_train_2/singapore_1 | 0.00456 | 0.00464 | -0.00000 | 1.27 | 17.05 | 0.0 |
| building_train_4/atlanta_5 | 0.00609 | 0.00633 | -0.00000 | 1.09 | 14.52 | 0.0 |
| building_train_4/denver_2 | 0.00474 | 0.00495 | -0.00000 | 0.75 | 0.13 | 24.4 |
| building_train_4/fairbanks_4 | 0.00113 | 0.00123 | -0.00000 | 0.95 | 0.24 | 3.8 |
| building_train_4/phoenix_8 | 0.00121 | 0.00134 | -0.00000 | 1.06 | 0.65 | 3.1 |
| building_train_4/singapore_3 | 0.00639 | 0.00658 | -0.00000 | 1.08 | 15.30 | 0.0 |
| building_train_5/atlanta_6 | 0.00399 | 0.00407 | 0.00000 | 1.16 | 12.49 | 0.0 |
| building_train_5/denver_3 | 0.00334 | 0.00347 | 0.00001 | 0.91 | 0.47 | 22.1 |
| building_train_5/fairbanks_5 | 0.00133 | 0.00140 | 0.00000 | 1.52 | 2.66 | 0.7 |
| building_train_5/phoenix_1 | 0.00198 | 0.00207 | 0.00000 | 1.01 | 0.57 | 11.2 |
| building_train_5/singapore_4 | 0.00517 | 0.00526 | 0.00000 | 1.12 | 16.60 | 0.0 |
| building_train_7/atlanta_8 | 0.00580 | 0.01657 | 0.00014 | 0.53 | 0.07 | 19.4 |
| building_train_7/denver_5 | 0.00401 | 0.01339 | 0.00008 | 0.57 | 6.01 | 0.2 |
| building_train_7/fairbanks_7 | 0.00696 | 0.01015 | 0.00005 | 0.52 | 0.04 | 24.8 |
| building_train_7/phoenix_3 | 0.00514 | 0.01486 | 0.00005 | 0.65 | 7.99 | 0.0 |
| building_train_7/singapore_6 | 0.00529 | 0.01504 | 0.00010 | 0.59 | 8.33 | 0.0 |
| building_train_8/atlanta_1 | 0.00399 | 0.00415 | 0.00000 | 0.37 | 0.06 | 10.0 |
| building_train_8/denver_6 | 0.00294 | 0.00310 | 0.00000 | 0.49 | 3.13 | 0.3 |
| building_train_8/fairbanks_8 | 0.02075 | 0.02111 | 0.00000 | 0.83 | 0.00 | 55.7 |
| building_train_8/phoenix_4 | 0.00906 | 0.00940 | 0.00000 | 0.67 | 11.01 | 0.0 |
| building_train_8/singapore_7 | 0.00512 | 0.00523 | -0.00000 | 0.46 | 6.25 | 0.0 |

By station: atlanta 0.0042, denver 0.0034, fairbanks 0.0140, phoenix 0.0063, singapore 0.0054.
The building_train_7 scenarios drop by 31-70% (holiday fix); all others by 1.2-9.6% (zone release, margins).

Submissions so far:
- 2026-11-25 01:17 UTC, kernel v1 = v3 (ref 58859225): pending.
- 2026-11-25 04:23 UTC, kernel v2 = v5 (ref 58866701): pending. Kaggle commit run reproduced the local
  building_train_1/atlanta_2 score (0.00268). 3 submissions left today.

### v5 on the held-out validation set (24 scenarios, mean 0.00806)

12 new buildings (25-78 zones, 3-8 AHUs) x one Minneapolis strip and one strip unused in training.
Comfort part: mean 0.00002, max 0.00035 (building_val_610/minneapolis_1). False released zone-steps: 0.
The set is colder than the training set (12 of 24 scenarios are Minneapolis), hence the higher mean.

| scenario | score | comfort | AHUs | zones | fan kW | cooling kW | gas kW |
|---|---|---|---|---|---|---|---|
| building_val_601/atlanta_4 | 0.00584 | -0.00000 | 8 | 75 | 1.44 | 18.55 | 0.0 |
| building_val_601/minneapolis_1 | 0.01547 | 0.00000 | 8 | 75 | 1.66 | 0.08 | 108.5 |
| building_val_602/atlanta_7 | 0.00334 | 0.00001 | 6 | 59 | 1.14 | 4.13 | 7.2 |
| building_val_602/minneapolis_2 | 0.02163 | 0.00000 | 6 | 59 | 1.97 | 0.07 | 114.8 |
| building_val_604/denver_1 | 0.01467 | 0.00001 | 6 | 47 | 1.34 | 0.03 | 76.8 |
| building_val_604/minneapolis_3 | 0.00704 | 0.00001 | 6 | 47 | 0.80 | 1.05 | 35.7 |
| building_val_605/denver_4 | 0.00255 | -0.00000 | 8 | 78 | 1.52 | 6.37 | 1.8 |
| building_val_605/minneapolis_4 | 0.00332 | 0.00000 | 8 | 78 | 1.68 | 8.10 | 3.4 |
| building_val_606/fairbanks_3 | 0.02155 | 0.00001 | 6 | 58 | 1.96 | 0.09 | 116.0 |
| building_val_606/minneapolis_5 | 0.00392 | 0.00000 | 6 | 58 | 1.22 | 8.58 | 0.5 |
| building_val_607/fairbanks_6 | 0.00078 | -0.00000 | 7 | 62 | 1.17 | 0.80 | 0.8 |
| building_val_607/minneapolis_6 | 0.00248 | -0.00000 | 7 | 62 | 1.37 | 6.01 | 0.1 |
| building_val_608/minneapolis_7 | 0.00137 | -0.00000 | 6 | 25 | 0.53 | 1.34 | 3.6 |
| building_val_608/phoenix_2 | 0.00194 | -0.00000 | 6 | 25 | 0.72 | 4.21 | 0.2 |
| building_val_609/minneapolis_8 | 0.00432 | -0.00000 | 5 | 29 | 0.41 | 0.02 | 18.9 |
| building_val_609/phoenix_7 | 0.00293 | -0.00000 | 5 | 29 | 0.81 | 5.49 | 0.1 |
| building_val_610/minneapolis_1 | 0.03475 | 0.00035 | 8 | 73 | 4.87 | 0.01 | 237.4 |
| building_val_610/singapore_2 | 0.00598 | 0.00003 | 8 | 73 | 1.36 | 19.44 | 0.0 |
| building_val_611/minneapolis_2 | 0.01603 | 0.00000 | 5 | 47 | 1.29 | 0.16 | 70.4 |
| building_val_611/singapore_5 | 0.00783 | -0.00000 | 5 | 47 | 1.06 | 15.71 | 0.0 |
| building_val_612/atlanta_4 | 0.00545 | -0.00000 | 7 | 63 | 1.24 | 15.12 | 0.0 |
| building_val_612/minneapolis_3 | 0.00357 | -0.00000 | 7 | 63 | 0.83 | 1.59 | 18.2 |
| building_val_613/atlanta_7 | 0.00304 | -0.00000 | 3 | 28 | 0.59 | 2.14 | 2.6 |
| building_val_613/minneapolis_4 | 0.00351 | -0.00000 | 3 | 28 | 0.60 | 2.92 | 2.2 |


### Parameter sensitivity (4 small scenarios x 10 days, base = v5 + adaptive holiday wait, mean 0.00921)

| change | mean | note |
|---|---|---|
| rel_delta 5 | 0.00919 | release depth is not the limit |
| rel_delta 5, rel_ramp 0.4 | 0.00918 | |
| p_pre 300 / 700 | 0.00925 / 0.00922 | 400 Pa is about right |
| m_heat = m_cool = 0.05 | 0.00916 | |
| pre_margin 0.1, pre_extra_steps 0 | 0.00916 | small 06:00 misses appear |
| comfort_scale 10 / 100 | 0.00921 / 0.00921 | no effect |
| force-start within 2 steps of occupancy | 0.00921, comfort 0 | removes the 06:00 misses |
| force-start + margins 0.05 + pre_margin 0.1 + extra 0 | 0.00914 | adopted as v6 |

The controller is on a plateau with respect to its tuning knobs (each worth <1%); remaining gains have to be
structural.

**Evaluated and rejected**
- Tracking morning arrivals to release zones where fewer people arrived than the per-zone maximum: only
  15-40% of zones keep a clean record through the noisy first two hours, so the extra capture is ~3-5% of
  empty zone-time for a real risk of releasing an occupied zone.
- Relaxing not-yet-occupied zones in the morning: arrivals are unpredictable and detected two calls late;
  the comfort cost of each late detection outweighs the saving until ~08:15, when few zones are still empty.
- Pre-cooling before 06:00 on cheaper electricity, night setback floors, pre-heating at reduced air
  temperature: each worth well under 1% on paper, some negative.

**Deliberately not used: the occupancy RNG.** `RandomizedArrivalDepartureOccupancy` is seeded with a fixed
default (17321) and no scenario overrides it, so arrival and departure times are a deterministic function of
zone count and workday index and could in principle be replayed inside the policy to condition each zone
just in time. That is exploiting a harness artefact rather than controlling the building, it is fragile, and
it sits too close to the "no hard-coded signatures" rule, so the controller only uses what it can infer
from live observations.

| v6 | v5 + adaptive holiday wait (exp(-7) false-positive rate) + force-start within 2 steps of occupancy + margins 0.05 K, pre_margin 0.1 K | **0.00660** (30 train; no scenario worse than v5; comfort part mean 0.000007) | submitted 2026-11-25 ~07:35 UTC (kernel v3, ref 58873131), pending |

v6 full training set: mean 0.00660 (v5 0.00667, v3 0.00825). One false release event in
building_train_4/fairbanks_4 (24 zone-steps, no measurable comfort cost) and the 2 zone-steps in
building_train_2/fairbanks_2 seen in v5. Kaggle commit run of kernel v3 reproduced building_train_1/atlanta_2
(-22.814 over 8641 steps = 0.00264). 2 submissions left today.

**LB result 1 (2026-11-25): v3 = 0.01003 public** (local train 0.00825 including the holiday bug, ~0.0067
without it). The hidden set is therefore roughly 1.2-1.5x harsher than the shipped training set, in line
with what public notebooks report (board 5-30% above local). Current board: #1 0.00870, #10 0.00896.

**Tested and rejected: pre-charging the thermal mass at a reduced air temperature.** Reheat is more
efficient while the zone air is cold (gas per unit of net heat is 1.4-1.6 at 280-285 K against ~2.2 at
294 K with -35 C outside), so holding the air a few K below the band for hours before occupancy looked
attractive. Measured on four cold/mild scenarios x 10 days (base 0.00845): hold 2 K below for 2 h 0.00898,
5 K below for 4 h 0.00879, 3 K below for 6 h 0.00955. The extra envelope loss and air-handler run time
outweigh the efficiency gain; the latest possible start stays.

v6 on the validation set (24 scenarios): mean 0.00796 (v5 0.00806); no scenario worse than v5, no false
releases, comfort part mean 0.00002.

**Harsh-cold profile (building_val_610/minneapolis_1, the worst local scenario, 0.034).** Day-by-day: on
-20 C days one or two large zones are capacity-limited (2 m3/s per VAV regardless of zone size), so they
start the day 2-6 K below the band even after an all-night pre-start, the fan runs at up to 2000 Pa and the
day costs ~5% in fan and ~5% in comfort on top of gas. A holiday Monday wastes a full pre-start (2% of the
episode). Tested on the first 17 days (base 0.03212):
- pre-start pressure boost for zones predicted late (700-1500 Pa in the last 2-6 h): 0.0331-0.0339, worse.
- lower pre-start cap (250 Pa): 0.03328, worse.
- cap on occupied-hours pressure: 1200 Pa 0.03203, 800 Pa 0.03189, 500 Pa 0.03195 -> adopt 800 Pa with a
  guard rail (cap lifted for an air handler when any of its zones is more than 3 K outside the band).

**Upper bound on occupancy awareness (oracle experiment, analysis only).** Feeding the policy the true
per-zone occupancy from a recorded trace (zones conditioned only from 2 steps before someone is present,
all day including mornings, others relaxed by 3 K) improves on the heat-signature tracker by only 3.1% on
building_train_8/fairbanks_8 and 2.3% on building_train_8/phoenix_4 (10 days). So the tracker already
captures more than half of what perfect knowledge could give, and no occupancy trick (including replaying
the occupancy RNG) is worth more than ~3% from here.

**LB result 2 (2026-11-25): v5 = 0.00964 public** (v3 0.01003, -3.9%). The local v3->v5 gain excluding
the building_train_7 holiday days is about the same, so the hidden set did not contain a holiday-bug
building and it is ~1.45x harsher than the training set. Board: #1 0.00870, #10 0.00896.

## 2026-11-25 (afternoon): where could another 8% come from? Audits and bounds

Context: v5 is at 0.00964, the top-20 pack sits at 0.0087-0.0091. Public analysis notebooks report the same
board/local ratio for a different controller family (public "DemandSized325" lineage: local 0.00687, board
0.0099 in round 1), and say the hidden set has 48 scenarios. The board was reset on 2026-11-18 (round 2:
energy-balanced tank, flow-weighted return air); the best public round-2 notebook claims 0.01002. So the
ratio board/local ~= 1.44 is a property of the hidden set, and the pack corresponds to ~0.0061 on the 30
training scenarios against 0.0066 for v6/v7.

**Score concentration.** Train + validation (54 scenarios): ten deep-winter scenarios carry 51% of the total
(building_train_1/fairbanks_1 0.043, building_val_610/minneapolis_1 0.034, ...). 

**Deep-cold audit (building_train_1/fairbanks_1, -25..-33 C; `tools/sbrun.py` traces now record per-VAV flow,
true supply temperature and coil heat).**
- Coil heat = billed gas within 1%. 96-98% of the airflow is discharged at >= 34 C during occupied hours,
  so heat is delivered at the best efficiency the plant allows; 54% of the gas is the mandatory 30% outside
  air (gas = zone heat + 0.3 x flow x rho cp x (T_return - T_out)).
- One workday at -32 C: pre-heat 02:30-06:00 13% of the day's gas, 06:00-13:00 61% (of which ~24% is
  re-charging the thermal mass that cooled 4.4 K overnight), 13:00-19:00 26%. Holding 21 C all night would
  cost ~21% more, so the setback is right; charging the mass earlier at lower air temperature was measured
  worse (see above).
- Releasing vacated zones deeper (rel_delta 30) or with perfect per-zone occupancy in the afternoon changes
  the 10-day score by only -0.1..-0.4%: the afternoon saving (gas at 18:00 falls from 184 to 71 kW with an
  oracle) is paid back the next morning because the released zones' mass has to be re-heated.
- "Pulsing" potential (all net zone heat delivered at 35 C discharge, no throttled minimum-flow reheat):
  0.3% of gas in deep cold, 1-3% in mild cold. Banking with a stored-heat credit and banking with a price on
  heat flushed from satisfied zones were both implemented and measured on four mild-cold scenarios x 10 days:
  0.00392 (base) vs 0.00393-0.00399. Rejected again; `bank_max_n=0` stays.

**Hot audit (Phoenix/Singapore).** Supply air at the 285 K minimum, recirculation 0.7, 200 Pa: compressor
load = flow x rho cp x (T_mix - T_supply) / 3.5 with the flow already minimal. Nothing left at plant level.

**Mild-cold audit (building_train_4/denver_2).** 80% of the gas burns 05:00-10:00 (mass recharge after the
night). Afternoon air handlers run because single interior zones sit at the top of the band and need free
cooling; the minimum ventilation flow then flushes ~2 kW per air handler out of mid-band zones, but pulsing
cannot avoid it profitably (measured). Tank standing loss is 6.25 W/K (0.45-0.67 kW), 3-6% of gas in mild
scenarios and 0.4% in deep cold.

**Upper bound with perfect foresight of per-zone occupancy (analysis hook `oracle_pre`, never shipped).**
Zones are pre-conditioned only if someone arrives within the lead time; others float (10 days):

| scenario | v7 | foresight, 3 h lead | foresight, 1 h lead |
|---|---|---|---|
| building_train_1/fairbanks_1 | 0.04953 | 0.04881 (-1.5%) | 0.05574 (comfort misses) |
| building_train_8/fairbanks_8 | 0.01963 | 0.01907 (-2.9%) | 0.01940 |
| building_train_8/phoenix_4 | 0.00911 | 0.00885 (-2.9%) | 0.00888 |
| building_train_4/denver_2 | 0.00419 | 0.00379 (-9.5%) | 0.00346 (-17%) |

In mild weather most of the morning gas warms zones nobody is in yet; waiting lets the 10 W/m2 clock-driven
internal gains and the rising outdoor temperature do part of the work. This is the only lever of the right
size (mild scenarios are light individually but numerous) found anywhere, and it needs *future* arrivals.
Arrivals are memoryless (1/36 per step per person), so nothing observable predicts them; an honest
approximation (hold empty zones slightly low and react to the 70 W signature two calls later) costs more
comfort than it saves because early arrivals carry weight 1/N_present (re-derived: sum over arrivals of
0.12 x delta x 2 / k).
The only way to have that foresight is to replay the simulator's occupancy generator inside the policy
(shared `RandomState(17321)`, fixed draw order). **Still deliberately not done** (see "Deliberately not used"
above): it is an information leak of the harness, organisers have been patching such artefacts between
rounds, a hard-coded seed plus a re-implemented occupancy model is exactly what a signature scan would
flag, and the decision whether to take that risk with the account belongs to the owner. My estimate is
that it is worth 5-8% on the board, i.e. roughly the gap between this controller and the top-10.

Also noticed, not used: the hot-water "supervisor off" command zeroes the pump's electricity bill without
stopping the water flow (0.3-0.7% of the score); an accounting artefact, the policy only switches the loop
off when no reheat is requested.

**Release coverage.** 66-74% of empty afternoon zone-steps are released. The rest: departures counted but
the zone flagged ambiguous/noisy 10-20%, one of two departures missed 6-8%, zones nobody came to all day up
to 12% (one-occupant buildings). Worth ~1% if all were captured.
