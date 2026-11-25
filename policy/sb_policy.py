"""Structure-aware HVAC controller for the Smart Buildings competition.

Self-contained (numpy only, scipy optional). The same file is embedded in the
Kaggle notebook.

Core ideas
----------
* Comfort is only scored while people are in the building (workdays, comfort
  hours), so the plant is off outside that window apart from a model-based
  optimal start.
* Each zone's air node is modelled one step ahead as

      T+ = (Cdt*T + W + Q_reheat + Cs*Ts) / (K0 + Cs)

  where Cs is the supply-air capacity rate, Ts the air-handler supply
  temperature, and W a slowly varying disturbance (thermal mass, walls,
  envelope, gains) tracked by an observer.  The zone sensor lags the true
  temperature by one step, so the current state is predicted first.
* theta = W / (K0 - Cdt) is the temperature the zone air would settle at with
  the plant off; its slow dynamics are identified online and used to find the
  latest time pre-conditioning has to start.
* Every step, each air handler picks the cheapest feasible combination of
  recirculation, supply-air setpoint, static pressure, dampers and reheat
  valves that keeps its zones inside the comfort band at the next step; if all
  zones stay in band unaided the air handler is switched off.
"""

from __future__ import annotations

import math

import numpy as np

RHOCP = 1.2 * 1006.0          # J/m3/K of air
WATER_C = 4180.0 * 1000.0     # J/m3/K of water
BIG = 1.0e9

DEFAULT_PARAMS = dict(
    # comfort margins inside the band (K)
    m_heat=0.05,
    m_cool=0.05,
    # zone model
    cdt_per_m2=12.0,           # air heat capacity / step, W/K per m2
    g_per_m2=4.8,              # air <-> mass conductance, W/K per m2
    k_face=2.8,                # W/K per wall-adjacent cell face
    w_alpha=0.6,               # observer smoothing
    gain_step_w_m2=8.0,        # scheduled internal gains above standby
    # slow model priors (per step)
    slow_a0=0.02,
    slow_b0=0.001,
    slow_lam_a=50.0,
    slow_lam_b=2.0e4,
    slow_lam_c=1.0e3,
    # optimal start
    max_lead_steps=144,
    p_pre=400.0,               # static pressure cap while pre-conditioning
    pre_margin=0.1,            # K of safety on the predicted start temperature
    pre_extra_steps=0,         # start this many steps earlier than predicted
    pre_hold_steps=0,          # hold air at (band floor - pre_hold_k) this many steps before occupancy
    pre_hold_k=5.0,
    pre_force_steps=2,         # within this many steps of occupancy, start every zone that needs it
    # night flush (free cooling with outside air while unoccupied)
    flush_on=0,
    flush_window_steps=120,    # only in the last hours before occupancy
    flush_delta=0.6,           # stop when the free-float temperature is this far above the band floor
    flush_hyst=0.3,
    flush_dt_min=2.5,          # zone must be this much warmer than outside
    flush_net_w_m2=1.5,        # zone counts as cooling-prone if it net-cooled by this much last day
    flush_mech_w=300.0,        # mean compressor W per air handler last day for flushing to pay
    flush_ratio=0.3,           # flush zones needed per heating-prone zone on the same air handler
    # banking: over-deliver inside the band so the air handler can rest for n steps
    bank_max_n=0,
    bank_rel_frac=0.5,         # only bank on air handlers where at least this share of zones is released
    bank_kappa=1.0,            # accept if banked cost < kappa * (n + 1) * steady cost
    bank_eps=0.03,
    # release of zones whose occupants have left for the day (afternoon only)
    rel_on=1,
    rel_delta=3.0,             # K the band is widened by for released zones
    rel_ramp=0.25,             # K per step the widening grows by (keeps neighbours quiet)
    rel_door_w=15.0,           # W/K per doorway cell between adjacent zones
    rel_heat_w=70.0,           # sensible heat per occupant
    rel_tol=22.0,              # W: unexplained disturbance jump that disqualifies a zone
    rel_sig_max=15.0,          # W: zones noisier than this are never released
    rel_lag=1,                 # count change that matches the current disturbance innovation
    rel_start_hour=13.0,
    # holiday / end-of-day detection
    holiday_steps=3,           # minimum steps of an empty comfort period before assuming a holiday
    holiday_exp=7.0,           # wait until exp(-holiday_exp) is the chance of a normal day looking empty
    # hot water
    tw_set=350.0,
    dp_min=0.3,
    dp_max=20.0,
    # penalties
    comfort_scale=30.0,
    # plant constants (generator constants)
    kv=0.108,
    d_min=0.1,
    p_min=200.0,
    p_des=2000.0,
    r_max=0.7,
    t_dis_max=308.15,
    cop=3.5,
    fan_eff=0.9,
)


def _label_rooms(zone_map):
    """4-connected components of the air cells, in raster order (as the sim)."""
    try:
        from scipy import ndimage
        lab, n = ndimage.label(np.asarray(zone_map) == 0)
        return lab, int(n)
    except Exception:  # pragma: no cover - scipy missing
        return None, 0


class SBPolicy:
    name = "sb"

    def __init__(self, building_meta_data, action_spec=None, **overrides):
        md = building_meta_data
        self.p = dict(DEFAULT_PARAMS)
        self.p.update(overrides)
        p = self.p
        self.n_disc = int(md["n_discrete_action"])
        self.n_cont = int(md["n_continuous_action"])
        names = list(md["observation_field_names"])
        self.name_idx = {n: i for i, n in enumerate(names)}
        self.norm = md.get("observation_normalization_metadata", {}) or {}

        # ---- air handlers and their zones -------------------------------
        # dict order is not preserved across the gateway's serialization, and the
        # positional index lists are in air-handler number order: sort by number.
        def _num(name):
            digits = "".join(ch for ch in str(name) if ch.isdigit())
            return (int(digits) if digits else 10 ** 9, str(name))

        self.ahu_ids = sorted(md["ahu_zone_assignments"].keys(), key=_num)
        self.K = len(md["temp_indices"])
        self.temp_idx = list(md["temp_indices"])
        self.recirc_idx = list(md["recirculation_indices"])
        pres = list(md["pressure_indices"])
        water = list(md["water_indices"])
        self.ahu_pres_idx = pres[: self.K]
        extra = pres[self.K:]
        self.water_temp_idx = water[0] if water else None
        self.water_dp_idx = water[1] if len(water) > 1 else (extra[0] if extra else None)

        zai = md["zone_action_indices"]
        self.zone_ids = []
        zone_ahu = []
        for a, aid in enumerate(self.ahu_ids[: self.K]):
            for room in md["ahu_zone_assignments"][aid]:
                vid = room.replace("room_", "VAV_")
                if vid in zai and vid not in self.zone_ids:
                    self.zone_ids.append(vid)
                    zone_ahu.append(a)
        for vid in zai:
            if vid not in self.zone_ids:
                self.zone_ids.append(vid)
                zone_ahu.append(0)
        self.zone_ahu = np.asarray(zone_ahu, dtype=int)
        self.N = len(self.zone_ids)
        self.damper_idx = np.array(
            [zai[v]["supply_air_damper_percentage_command"] for v in self.zone_ids])
        self.valve_idx = np.array(
            [zai[v]["reheat_valve_percentage_command"] for v in self.zone_ids])
        self.ahu_zones = [np.flatnonzero(self.zone_ahu == a) for a in range(self.K)]

        # ---- observation indices ----------------------------------------
        def idx(name):
            return self.name_idx.get(name, -1)

        self.o_ztemp = np.array([idx(f"{v}_zone_air_temperature_sensor") for v in self.zone_ids])
        self.o_fmax = np.array([idx(f"{v}_supply_air_flowrate_setpoint") for v in self.zone_ids])
        self.o_oat = [i for n, i in self.name_idx.items() if n.endswith("outside_air_temperature_sensor")]
        self.o_tw = idx("hws_supply_water_temperature_sensor")
        if self.o_tw < 0:
            c = [i for n, i in self.name_idx.items() if n.endswith("supply_water_temperature_sensor")]
            self.o_tw = c[0] if c else -1
        self.o_hod = (idx("hod_cos_000"), idx("hod_sin_000"))
        self.o_dow = (idx("dow_cos_000"), idx("dow_sin_000"))
        self.o_cnow = idx("comfort_mode_now")
        self.o_occ = idx("num_occupants")

        zt_norm = self.norm.get("zone_air_temperature_sensor", None)
        self.zt_mean = float(zt_norm["sample_mean"]) if zt_norm else 295.0
        self.zt_std = float(zt_norm["sample_std"]) if zt_norm else 5.0
        oat_norm = self.norm.get("outside_air_temperature_sensor", None)
        self.oat_mean = float(oat_norm["sample_mean"]) if oat_norm else 295.0
        self.oat_std = float(oat_norm["sample_std"]) if oat_norm else 10.0

        # ---- schedule ------------------------------------------------------
        self.h_start = float(md.get("morning_start_hour", 6))
        self.h_end = float(md.get("evening_start_hour", 19))
        cw = md.get("comfort_temp_window", (294.0, 297.0))
        self.band_lo = float(cw[0])
        self.band_hi = float(cw[1])

        # ---- action ranges -----------------------------------------------
        an = md.get("action_normalization_metadata", {}) or {}

        def rng(key_sub, default):
            for k, v in an.items():
                if key_sub in k:
                    return float(v["min_native_value"]), float(v["max_native_value"])
            return default

        self.sat_rng = rng("supply_air_temperature_setpoint", (285.0, 305.0))
        self.sp_rng = rng("static_pressure_setpoint", (0.0, 2000.0))
        self.rc_rng = rng("recirculation_percentage_command", (0.0, 1.0))
        self.tw_rng = rng("supply_water_temperature_setpoint", (310.0, 350.0))
        self.dp_rng = rng("differential_pressure_setpoint", (0.0, 20.0))
        self.dmp_rng = rng("supply_air_damper_percentage_command", (0.0, 1.0))
        self.vlv_rng = rng("reheat_valve_percentage_command", (0.0, 1.0))

        # ---- geometry -------------------------------------------------------
        area = np.full(self.N, 100.0)
        faces = np.full(self.N, 250.0)
        try:
            zm = md.get("zone_map", None)
            if zm is not None:
                lab, n = _label_rooms(zm)
                if lab is not None and n >= self.N:
                    cell_m2 = 0.04
                    counts = np.bincount(lab.ravel(), minlength=n + 1)
                    fcount = np.zeros(n + 1)
                    for sl_a, sl_b in (
                        ((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                        ((slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                    ):
                        a_, b_ = lab[sl_a], lab[sl_b]
                        m1 = (a_ > 0) & (b_ == 0)
                        m2 = (b_ > 0) & (a_ == 0)
                        fcount += np.bincount(a_[m1], minlength=n + 1)
                        fcount += np.bincount(b_[m2], minlength=n + 1)
                    for i, vid in enumerate(self.zone_ids):
                        rid = int(vid.split("_")[-1])
                        if 1 <= rid <= n and counts[rid] > 0:
                            area[i] = counts[rid] * cell_m2
                            faces[i] = fcount[rid]
        except Exception:
            pass
        # doorway cells: air in the floor plan but a zone boundary in the zone map
        self.door = np.zeros((self.N, self.N))
        try:
            fpl = md.get("floor_plan", None)
            zm = md.get("zone_map", None)
            if fpl is not None and zm is not None:
                lab, n = _label_rooms(zm)
                fpl = np.asarray(fpl)
                zm = np.asarray(zm)
                if lab is not None and n >= self.N and fpl.shape == zm.shape:
                    zi = {int(v.split("_")[-1]): i for i, v in enumerate(self.zone_ids)}
                    rows, cols = np.nonzero((fpl == 0) & (zm != 0))
                    h, w = lab.shape
                    for r, c in zip(rows.tolist(), cols.tolist()):
                        adj = set()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w and lab[rr, cc] > 0:
                                adj.add(int(lab[rr, cc]))
                        adj = [zi[a] for a in adj if a in zi]
                        for i in range(len(adj)):
                            for j in range(i + 1, len(adj)):
                                self.door[adj[i], adj[j]] += 1.0
                                self.door[adj[j], adj[i]] += 1.0
        except Exception:
            self.door = np.zeros((self.N, self.N))
        self.door_sum = self.door.sum(1)
        self.area = area
        self.faces = faces
        self.cdt = p["cdt_per_m2"] * area
        self.kss = p["g_per_m2"] * area + p["k_face"] * faces
        self.k0 = self.cdt + self.kss
        self.g_sched = p["gain_step_w_m2"] * area

        # ---- reward normalisation -------------------------------------------
        self.e_max = 80000.0 * self.K
        self.g_max = 205000.0 * self.K

        # ---- state ---------------------------------------------------------
        self.Wb = None                # disturbance without scheduled gains (W)
        self.p1 = None                # plan applied last step
        self.p2 = None                # plan applied the step before
        self.y_prev = None
        self.xhat_prev = None
        self.theta_prev = None
        self.step = 0
        self.occ_seen_today = False
        self.comfort_steps_today = 0
        self.holiday = False
        self.day_done = False
        self.prev_cnow = None
        self.start_heat = np.zeros(self.N, dtype=bool)
        self.start_cool = np.zeros(self.N, dtype=bool)
        self.day_net = np.zeros(self.N)       # net HVAC heat into each zone over the occupied day (W-steps)
        self.day_cnt = 0
        self.last_net = None                  # mean of the above for the last occupied day (W)
        self.flushing = np.zeros(self.N, dtype=bool)
        self.m_occ_est = 1                    # occupants per zone, learnt from the daily peak count
        self.wm_last = None                   # last raw disturbance measurement
        self.dwm = None                       # its latest one-step change
        self.occ_hist = [0.0, 0.0, 0.0, 0.0]  # recent observed occupant counts, newest first
        self.occ_max_today = 0.0
        self.dep = np.zeros(self.N)           # departures attributed to each zone today
        self.dirty = np.zeros(self.N, dtype=bool)
        self.drift = np.zeros(self.N)
        self.sig2 = np.full(self.N, 25.0)
        self.released = np.zeros(self.N, dtype=bool)
        self.rel_off = np.zeros(self.N)       # current band widening of released zones (K)
        self.dy = np.zeros(self.N)            # last one-step change of the zone sensors
        self.day_mech = np.zeros(self.K)      # compressor electricity per air handler over the day
        self.last_mech = None
        self.was_window = False
        # slow-model normal equations per zone
        lam = np.array([p["slow_lam_a"], p["slow_lam_b"], p["slow_lam_c"]])
        prior = np.array([p["slow_a0"], p["slow_b0"], 0.0])
        self.sXX = np.tile(np.diag(lam), (self.N, 1, 1))
        self.sXy = np.tile(lam * prior, (self.N, 1))
        self.slow = np.tile(prior, (self.N, 1))
        self._cand = None
        self.dbg = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_norm(x, lo, hi):
        if hi <= lo:
            return 0.0
        return float(np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    @staticmethod
    def _kelvin(v, mean, std):
        v = float(v)
        return v if v > 150.0 else v * std + mean

    def _weights(self, hour, workday):
        """Reward units per W for electricity and gas at this hour."""
        h = int(hour) % 24
        if workday:
            pe = 0.16 if (h < 6 or h >= 19) else (0.18 if h < 12 else 0.20)
        else:
            pe = 0.16
        ce_tab = (88.2, 87.8, 87.9, 87.8, 88.0, 88.2, 89.7, 94.0, 98.9, 100.8, 101.4,
                  101.8, 102.6, 103.4, 104.1, 104.7, 102.1, 97.0, 93.6, 92.5, 91.7,
                  90.7, 89.8, 89.0)
        ce = ce_tab[h] / 1000.0          # kg/kWh
        pg = 7.6 / 293.07                # $/kWh (typical month)
        cg = 53.12 / 293.07              # kg/kWh
        cost_den = pe * self.e_max + pg * self.g_max
        carb_den = ce * self.e_max + cg * self.g_max
        we = 0.2 * (pe / cost_den + ce / carb_den)
        wg = 0.2 * (pg / cost_den + cg / carb_den)
        return we, wg

    # ------------------------------------------------------------- main entry
    def get_action(self, obs):
        p = self.p
        obs = np.asarray(obs, dtype=np.float64)
        N, K = self.N, self.K

        # ---- decode observation ------------------------------------------
        y = obs[self.o_ztemp]
        y = np.where(y > 150.0, y, y * self.zt_std + self.zt_mean)
        fmax = np.where(self.o_fmax >= 0, obs[self.o_fmax], 2.0)
        fmax = np.where(fmax > 0.01, fmax, 2.0)
        t_out = self._kelvin(obs[self.o_oat[0]], self.oat_mean, self.oat_std) if self.o_oat else 293.0
        t_w = float(obs[self.o_tw]) if self.o_tw >= 0 else 340.0
        if t_w < 150.0:
            t_w = 340.0
        hour = (math.atan2(obs[self.o_hod[1]], obs[self.o_hod[0]]) / (2 * math.pi) * 24.0) % 24.0
        dow = int(round((math.atan2(obs[self.o_dow[1]], obs[self.o_dow[0]]) / (2 * math.pi) * 7.0))) % 7
        cnow = bool(obs[self.o_cnow] > 0.5) if self.o_cnow >= 0 else False
        occ = float(obs[self.o_occ]) if self.o_occ >= 0 else 0.0

        # ---- observer update / state prediction ---------------------------
        # `y` is the true temperature one step ago.  `p1` is the plan applied
        # to that state (outcome not visible yet); `p2` the one before it,
        # whose outcome is exactly `y`.
        if self.Wb is None:
            self.Wb = self.kss * y - (self.g_sched if cnow else 0.0)
        if self.p2 is not None and self.y_prev is not None:
            q2, ts2 = self._realize(self.p2, self.y_prev)
            cs2 = self.p2["cs"]
            w_meas = y * (self.k0 + cs2) - self.cdt * self.y_prev - q2 - cs2 * ts2
            w_meas = w_meas - (self.g_sched if self.p2["cnow"] else 0.0)
            al = p["w_alpha"]
            self.Wb = (1 - al) * self.Wb + al * w_meas
            self.dwm = (w_meas - self.wm_last) if self.wm_last is not None else None
            self.wm_last = w_meas
        else:
            self.dwm = None
        self.dy = (y - self.y_prev) if self.y_prev is not None else np.zeros(N)
        theta = self.Wb / self.kss
        if self.theta_prev is not None:
            self._slow_update(theta, self.theta_prev, y, t_out)
        self.theta_prev = theta
        if self.p1 is not None:
            q1, ts1 = self._realize(self.p1, y)
            cs1 = self.p1["cs"]
            w1 = self.Wb + (self.g_sched if self.p1["cnow"] else 0.0)
            zt = (self.cdt * y + w1 + q1 + cs1 * ts1) / (self.k0 + cs1)
            if self.xhat_prev is not None:
                self.dbg["pred_err"] = y - self.xhat_prev
        else:
            zt = y.copy()
        self.y_prev = y
        self.xhat_prev = zt
        W = self.Wb + (self.g_sched if cnow else 0.0)

        # ---- schedule state -------------------------------------------------
        # the action taken now is judged at t+5min
        hj = hour + 5.0 / 60.0
        dj = dow
        if hj >= 24.0 - 1e-9:
            hj -= 24.0
            dj = (dow + 1) % 7
        in_window = (dj < 5) and (self.h_start - 1e-6 <= hj < self.h_end - 1e-6)

        self.occ_hist = [occ] + self.occ_hist[:3]
        if cnow and not self.prev_cnow:
            self.occ_seen_today = False
            self.comfort_steps_today = 0
            self.holiday = False
            self.day_done = False
        if cnow:
            self.comfort_steps_today += 1
            if occ > 0:
                self.occ_seen_today = True
            # wait long enough that an ordinary late first arrival is unlikely:
            # P(nobody yet after k steps) = exp(-occupants * k / 36)
            n_occ = max(self.m_occ_est, 1) * N
            k_hol = min(max(int(math.ceil(p["holiday_exp"] * 36.0 / n_occ)), int(p["holiday_steps"])), 14)
            if (not self.occ_seen_today) and self.comfort_steps_today > k_hol:
                self.holiday = True
            if self.occ_seen_today:
                self.holiday = False
            if hour >= 13.0 + 1e-6 and occ <= 0 and self.occ_seen_today:
                self.day_done = True
        else:
            self.holiday = False
            self.day_done = False
        self.prev_cnow = cnow

        steps_to_start = None
        if not in_window:
            h, d = hj, dj
            if d < 5 and h < self.h_start:
                wait = self.h_start - h
            else:
                wait = 24.0 - h + self.h_start
                d = (d + 1) % 7
                while d >= 5:
                    wait += 24.0
                    d = (d + 1) % 7
            steps_to_start = int(round(wait * 12.0))

        steps_left = int(round((self.h_end - hj) * 12.0)) if in_window else 0
        lo = self.band_lo + p["m_heat"]
        hi = self.band_hi - p["m_cool"]
        lo_z = np.full(N, -BIG)
        hi_z = np.full(N, BIG)
        p_cap = self.sp_rng[1]
        if cnow and in_window:
            self.occ_max_today = max(self.occ_max_today, occ)
            if p["rel_on"]:
                self._track_departures(hour)
        else:
            self.dep[:] = 0.0
            self.dirty[:] = False
            self.released[:] = False
            self.rel_off[:] = 0.0
            self.drift[:] = 0.0
            if not cnow:
                self.occ_max_today = 0.0
        mode = "off"
        if in_window:
            self.start_heat[:] = False
            self.start_cool[:] = False
            if not self.holiday and not self.day_done:
                mode = "occ"
                lo_z[:] = lo
                hi_z[:] = hi
                if p["rel_on"] and self.released.any():
                    self.rel_off = np.where(
                        self.released, np.minimum(self.rel_off + p["rel_ramp"], p["rel_delta"]), 0.0)
                    lo_z = lo_z - self.rel_off
                    hi_z = hi_z + self.rel_off
        elif steps_to_start is not None and steps_to_start <= p["max_lead_steps"]:
            mode = "pre"
            self._prestart(zt, theta, steps_to_start, lo, hi, t_out, fmax)
            lo_z[self.start_heat] = lo
            hi_z[self.start_cool] = hi
            if p["pre_hold_steps"] > 0 and steps_to_start <= p["pre_hold_steps"]:
                # charge the thermal mass early at a reduced air temperature
                lo_z = np.maximum(lo_z, lo - p["pre_hold_k"])
            p_cap = p["p_pre"]
        else:
            self.start_heat[:] = False
            self.start_cool[:] = False

        we, wg = self._weights(hour, dow < 5)

        d_cmd = np.full(N, p["d_min"])
        v_cmd = np.zeros(N)
        cs_plan = np.zeros(N)
        cw_need = np.zeros(N)
        ahu_on = np.zeros(K, dtype=bool)
        ahu_set = np.full(K, self.sat_rng[1])
        ahu_p = np.full(K, p["p_min"])
        ahu_r = np.full(K, p["r_max"])

        if mode != "off":
            u_lo = self.k0 * lo_z - self.cdt * zt - W
            u_hi = self.k0 * hi_z - self.cdt * zt - W
            u_lo = np.where(lo_z < -1e8, -BIG, u_lo)
            u_hi = np.where(hi_z > 1e8, BIG, u_hi)
            for a in range(K):
                z = self.ahu_zones[a]
                if z.size == 0:
                    continue
                res = self._plan_ahu(zt[z], u_lo[z], u_hi[z], fmax[z], self.k0[z],
                                     lo_z[z], hi_z[z], t_out, t_w, we, wg, p_cap)
                if res is None:
                    continue
                if (mode == "occ" and p["bank_max_n"] > 0
                        and self.released[z].mean() >= p["bank_rel_frac"]):
                    res = self._bank(res, z, zt, W, lo_z[z], hi_z[z], fmax, t_out, t_w, we, wg,
                                     p_cap, steps_left)
                ahu_on[a] = True
                if mode == "occ":
                    self.day_mech[a] += res["cool"]
                ahu_set[a], ahu_p[a], ahu_r[a] = res["tset"], res["p"], res["r"]
                d_cmd[z] = res["d"]
                cs_plan[z] = res["c"]
                cw_need[z] = res["cw"]

        # ---- night flush with outside air -----------------------------------
        if (p["flush_on"] and mode != "occ" and steps_to_start is not None
                and 2 <= steps_to_start <= p["flush_window_steps"]):
            th_tgt = lo + p["flush_delta"]
            if self.last_net is None:
                prone = np.zeros(N, dtype=bool)
                heat_prone = np.zeros(N, dtype=bool)
            else:
                # only worth it where yesterday's cooling needed the compressor
                mech = (self.last_mech > p["flush_mech_w"])[self.zone_ahu]
                prone = mech & (self.last_net < -p["flush_net_w_m2"] * self.area)
                heat_prone = self.last_net > p["flush_net_w_m2"] * self.area
            want = prone & (zt - t_out > p["flush_dt_min"]) & (
                theta > th_tgt + np.where(self.flushing, 0.0, p["flush_hyst"]))
            self.flushing[:] = False
            sq = math.sqrt(p["p_min"] / p["p_des"])
            for a in range(K):
                z = self.ahu_zones[a]
                if ahu_on[a] or z.size == 0:
                    continue
                wz = want[z]
                if wz.sum() == 0 or wz.sum() < p["flush_ratio"] * heat_prone[z].sum():
                    continue
                ahu_on[a] = True
                ahu_set[a] = self.sat_rng[1]
                ahu_p[a] = p["p_min"]
                ahu_r[a] = 0.0
                d_cmd[z] = np.where(wz, 1.0, p["d_min"])
                cs_plan[z] = d_cmd[z] * fmax[z] * sq * RHOCP
                cw_need[z] = 0.0
                self.flushing[z] = wz
        else:
            self.flushing[:] = False

        # ---- hot water plant ---------------------------------------------
        kw = p["kv"] / 3600.0 * WATER_C        # W/K per sqrt(bar) at valve = 1
        heat_on = bool(np.any(cw_need > 1e-6))
        dp = p["dp_min"]
        if heat_on:
            dp = float(np.clip((cw_need.max() / kw) ** 2, p["dp_min"], p["dp_max"]))
            v_cmd = np.clip(cw_need / (kw * math.sqrt(dp)), 0.0, 1.0)
            cw_need = v_cmd * kw * math.sqrt(dp)

        # ---- write actions ---------------------------------------------------
        cont = np.zeros(self.n_cont, dtype=np.float32)
        disc = np.zeros(self.n_disc, dtype=np.int32)
        for a in range(K):
            disc[a] = 1 if ahu_on[a] else 0
            cont[self.temp_idx[a]] = self._to_norm(ahu_set[a], *self.sat_rng)
            cont[self.ahu_pres_idx[a]] = self._to_norm(ahu_p[a], *self.sp_rng)
            cont[self.recirc_idx[a]] = self._to_norm(ahu_r[a], *self.rc_rng)
        if self.n_disc > K:
            disc[K:] = 1 if heat_on else 0
        if self.water_temp_idx is not None:
            tw_sp = p["tw_set"] if heat_on else self.tw_rng[0]
            cont[self.water_temp_idx] = self._to_norm(tw_sp, *self.tw_rng)
        if self.water_dp_idx is not None:
            cont[self.water_dp_idx] = self._to_norm(dp, *self.dp_rng)
        dn = 2.0 * (d_cmd - self.dmp_rng[0]) / (self.dmp_rng[1] - self.dmp_rng[0]) - 1.0
        vn = 2.0 * (v_cmd - self.vlv_rng[0]) / (self.vlv_rng[1] - self.vlv_rng[0]) - 1.0
        cont[self.damper_idx] = np.clip(dn, -1.0, 1.0)
        cont[self.valve_idx] = np.clip(vn, -1.0, 1.0)

        # ---- remember the plan for the observer ---------------------------
        self.p2 = self.p1
        self.p1 = dict(cs=cs_plan, cw=cw_need, r=ahu_r.copy(), tset=ahu_set.copy(),
                       on=ahu_on.copy(), t_out=t_out, t_w=t_w, cnow=cnow)
        if mode == "occ":
            qn, tsn = self._realize(self.p1, zt)
            self.day_net += qn + cs_plan * (tsn - zt)
            self.day_cnt += 1
        if self.was_window and not in_window:
            if self.occ_max_today > 0:
                self.m_occ_est = max(1, int(round(self.occ_max_today / (0.9 * N))))
            if self.day_cnt > 24:
                self.last_net = self.day_net / self.day_cnt
                self.last_mech = self.day_mech / self.day_cnt
            self.day_mech = np.zeros(K)
            self.day_net = np.zeros(N)
            self.day_cnt = 0
        self.was_window = in_window
        self.dbg.update(hour=hour, dow=dow, active=bool(ahu_on.any()), mode=mode, ahu_on=ahu_on.copy(),
                        ahu_set=ahu_set.copy(), ahu_p=ahu_p.copy(), ahu_r=ahu_r.copy(),
                        cs=cs_plan.copy(), W=W.copy(), theta=theta.copy(), dp=dp,
                        start=(self.start_heat | self.start_cool).astype(float),
                        released=self.released.astype(float), dep=self.dep.copy(),
                        dwm=(self.dwm.copy() if self.dwm is not None else np.zeros(N)),
                        occ_obs=occ,
                        slow_a=self.slow[:, 0].copy(), holiday=self.holiday, day_done=self.day_done)
        self.step += 1
        return {"continuous_action": cont, "discrete_action": disc}

    # ------------------------------------------------------ occupancy tracking
    def _track_departures(self, hour):
        """Attribute each observed departure to a zone via its heat signature.

        After the arrival window closes nobody comes back, so a zone whose
        occupants have all been seen leaving needs no conditioning.  A
        departure removes `rel_heat_w` from the zone's disturbance one step
        after the building-wide count drops.  Only unambiguous signatures are
        counted, and a zone with any unexplained jump is never released.
        """
        p = self.p
        if self.dwm is None or hour < p["rel_start_hour"] + 0.16:
            return
        lag = int(p["rel_lag"])
        d = int(round(self.occ_hist[lag] - self.occ_hist[lag + 1]))
        heat = p["rel_heat_w"]
        tol = p["rel_tol"]
        m_occ = max(1, int(round(self.occ_max_today / (0.9 * self.N))))
        # heat exchanged with neighbours through doorways when their temperature moves
        comp = p["rel_door_w"] * (self.door @ self.dy - self.door_sum * self.dy)
        x = self.dwm - np.median(self.dwm) - self.drift - comp
        active = ~self.released
        cap = np.maximum(m_occ - self.dep, 0.0)
        nz = np.clip(np.round(-x / heat), 0.0, cap)
        match = (np.abs(x + heat * nz) <= tol) & active
        cand = match & (nz >= 1)
        units = int(nz[cand].sum())
        chosen = np.zeros(self.N)
        if d < 0 and 0 < units <= -d:
            chosen = np.where(cand, nz, 0.0)
            self.dep += chosen
        elif units > 0:
            # more signatures than departures: cannot tell which are real
            self.dirty |= cand
        resid = x + heat * chosen
        bad = (np.abs(resid) > tol) & active
        self.dirty |= bad
        quiet = (chosen == 0) & ~bad & active
        self.drift = np.where(quiet, self.drift + 0.3 * x, self.drift)
        self.sig2 = np.where(quiet, 0.95 * self.sig2 + 0.05 * resid * resid, self.sig2)
        self.released = (self.released | ((self.dep >= m_occ) & ~self.dirty
                                          & (self.sig2 < p["rel_sig_max"] ** 2)))

    # ------------------------------------------------------------- slow model
    def _slow_update(self, theta, theta_prev, t_air, t_out):
        """Accumulate  d(theta) = a*(T_air - theta) + b*(T_out - theta) + c."""
        yv = theta - theta_prev
        ok = np.abs(yv) < 0.5
        x = np.stack([t_air - theta_prev, t_out - theta_prev, np.ones(self.N)], 1)
        x = x * ok[:, None]
        self.sXX += x[:, :, None] * x[:, None, :]
        self.sXy += x * yv[:, None]
        if self.step % 12 == 0:
            try:
                sol = np.linalg.solve(self.sXX, self.sXy[:, :, None])[:, :, 0]
                sol[:, 0] = np.clip(sol[:, 0], 0.004, 0.08)
                sol[:, 1] = np.clip(sol[:, 1], 0.0, 0.01)
                sol[:, 2] = np.clip(sol[:, 2], -0.05, 0.05)
                self.slow = sol
            except Exception:
                pass

    def _prestart(self, zt, theta, n, lo, hi, t_out, fmax):
        """Latch zones whose pre-conditioning has to begin now.

        Simulates: plant off for one more step, then full capacity (at the
        pre-start pressure cap) until occupancy.  If that no longer reaches
        the band, the zone starts now.
        """
        p = self.p
        todo_h = ~self.start_heat
        todo_c = ~self.start_cool
        if not (todo_h.any() or todo_c.any()):
            return
        a, b, c = self.slow[:, 0], self.slow[:, 1], self.slow[:, 2]
        cdt, kss, k0 = self.cdt, self.kss, self.k0
        cs = fmax * math.sqrt(p["p_pre"] / p["p_des"]) * RHOCP
        t_heat = p["t_dis_max"]
        t_cool = self.sat_rng[0]
        marg = p["pre_margin"]

        def off_step(ta, th):
            ta = (cdt * ta + kss * th) / k0
            th = th + a * (ta - th) + b * (t_out - th) + c
            return ta, th

        # free-running trajectory to the first occupied step
        ta_o, th_o = zt.copy(), theta.copy()
        for _ in range(n + 1):
            ta_o, th_o = off_step(ta_o, th_o)
        want_h = todo_h & (ta_o < lo)
        want_c = todo_c & (ta_o > hi)
        if not (want_h.any() or want_c.any()):
            return
        if n <= p["pre_force_steps"]:
            # close to occupancy: start everything that would otherwise begin the day out of band
            self.start_heat |= want_h
            self.start_cool |= want_c
            return
        nb = max(n - p["pre_extra_steps"], 0)
        ta1, th1 = off_step(zt.copy(), theta.copy())
        if want_h.any():
            ta, th = ta1.copy(), th1.copy()
            for _ in range(nb):
                ta = np.minimum((cdt * ta + kss * th + cs * t_heat) / (k0 + cs), lo + marg)
                th = th + a * (ta - th) + b * (t_out - th) + c
            self.start_heat |= want_h & (ta < lo + marg - 1e-6)
        if want_c.any():
            ta, th = ta1.copy(), th1.copy()
            for _ in range(nb):
                ta = np.maximum((cdt * ta + kss * th + cs * t_cool) / (k0 + cs), hi - marg)
                th = th + a * (ta - th) + b * (t_out - th) + c
            self.start_cool |= want_c & (ta > hi - marg + 1e-6)

    def _bank(self, res, z, zt, W, lo, hi, fmax, t_out, t_w, we, wg, p_cap, steps_left):
        """Try landing the zones far enough inside the band that the air handler
        can stay off for the next n steps; keep it if cheaper per step."""
        p = self.p
        k0, cdt, w = self.k0[z], self.cdt[z], W[z]
        best, best_rate = res, res["cost"]
        need_lo = np.array(lo, dtype=float).copy()
        need_hi = np.array(hi, dtype=float).copy()
        eps = p["bank_eps"]
        for n in range(1, int(p["bank_max_n"]) + 1):
            if steps_left < n + 1:
                break
            # temperature needed now so that one more plant-off step still ends in band
            need_lo = (k0 * (need_lo + (eps if n == 1 else 0.0)) - w) / cdt
            need_hi = (k0 * (need_hi - (eps if n == 1 else 0.0)) - w) / cdt
            lo_b = np.maximum(need_lo, lo)
            hi_b = np.minimum(need_hi, hi)
            if np.any(lo_b > hi_b - 0.05):
                break
            u_lo = k0 * lo_b - cdt * zt[z] - w
            u_hi = k0 * hi_b - cdt * zt[z] - w
            r2 = self._plan_ahu(zt[z], u_lo, u_hi, fmax[z], k0, lo_b, hi_b, t_out, t_w, we, wg,
                                p_cap, force=True)
            if r2 is None or r2["dev"] > 1e-6:
                break
            rate = r2["cost"] / (n + 1)
            if rate < p["bank_kappa"] * best_rate:
                best, best_rate = r2, rate
        return best

    def _realize(self, plan, t_true):
        """Reheat power and supply temperature a stored plan produced on `t_true`."""
        p = self.p
        cs = plan["cs"]
        q = np.zeros(self.N)
        ts = np.zeros(self.N)
        for a in range(self.K):
            if not plan["on"][a]:
                continue
            z = self.ahu_zones[a]
            c = cs[z]
            tot = c.sum()
            if tot <= 0:
                continue
            tret = float((c * t_true[z]).sum() / tot)
            tmix = plan["r"][a] * tret + (1.0 - plan["r"][a]) * plan["t_out"]
            t_s = min(tmix, plan["tset"][a])
            ts[z] = t_s
            qz = plan["cw"][z] * max(plan["t_w"] - t_s, 0.0)
            q[z] = np.minimum(qz, c * max(p["t_dis_max"] - t_s, 0.0))
        return q, ts

    # ------------------------------------------------------------ AHU planner
    def _candidates(self):
        if self._cand is None:
            rmax = self.p["r_max"]
            lo_s, hi_s = self.sat_rng
            r_free = np.linspace(0.0, rmax, 15)
            t_mech = np.arange(lo_s, min(lo_s + 13.0, hi_s) + 1e-9, 1.0)
            R = np.concatenate([r_free, np.zeros_like(t_mech), np.full_like(t_mech, rmax)])
            T = np.concatenate([np.full_like(r_free, hi_s), t_mech, t_mech])
            self._cand = (R[:, None], T[:, None])
        return self._cand

    def _plan_ahu(self, zt, u_lo, u_hi, fmax, k0, lo, hi, t_out, t_w, we, wg, p_cap, force=False):
        """Cheapest way to keep these zones inside their bounds at the next step.

        `lo`/`hi` are per-zone bounds (+-BIG when unconstrained); `u_lo`/`u_hi`
        the net heat (W, evaluated at the bound) needed to land on them.
        Returns None when the air handler should stay off.
        """
        p = self.p
        c_price = p["comfort_scale"] * 0.6 / 5.0 / max(self.N, 1)   # reward per K per zone
        has_lo = lo > -1e8
        has_hi = hi < 1e8

        dev_off = (np.where(has_lo, np.maximum(u_lo, 0.0), 0.0)
                   + np.where(has_hi, np.maximum(-u_hi, 0.0), 0.0)) / k0
        cost_off = c_price * dev_off.sum()
        if cost_off <= 0.0 and not force:
            return None

        R, TSET = self._candidates()
        m = R.shape[0]
        zt_ = zt[None, :]
        ulo = np.where(has_lo, u_lo, -BIG)[None, :]
        uhi = np.where(has_hi, u_hi, BIG)[None, :]
        lo_ = np.where(has_lo, lo, 0.0)[None, :]
        hi_ = np.where(has_hi, hi, 1000.0)[None, :]
        fm = fmax[None, :]
        tdm = p["t_dis_max"]
        kw = p["kv"] / 3600.0 * WATER_C
        cw_cap = kw * math.sqrt(p["dp_max"])
        p_lo = p["p_min"]
        p_hi = max(min(p_cap, self.sp_rng[1]), p_lo)

        P = np.full((m, 1), p_lo)
        C = None
        for _ in range(3):
            sq = np.sqrt(P / p["p_des"])
            cmin = p["d_min"] * fm * sq * RHOCP
            cmax = fm * sq * RHOCP
            if C is None:
                C = np.broadcast_to(cmin, (m, zt.size)).copy()
            C = np.clip(C, cmin, cmax)
            tret = (C * zt_).sum(1, keepdims=True) / C.sum(1, keepdims=True)
            tmix = R * tret + (1.0 - R) * t_out
            ts = np.minimum(tmix, TSET)
            # cooling side: Q + C (ts - hi) <= uhi
            gap_c = hi_ - ts
            c_cool = np.where((uhi < 0) & (gap_c > 0.05), -uhi / np.maximum(gap_c, 0.05), 0.0)
            c_need = np.maximum(cmin, c_cool)
            # heating side: Q + C (ts - lo) >= ulo, discharge <= tdm
            q0 = ulo - c_need * (ts - lo_)
            c_heat = np.where(q0 > 0, ulo / np.maximum(tdm - lo_, 1.0), 0.0)
            c_new = np.maximum(c_need, c_heat)
            need_sq = (c_new / (fm * RHOCP)).max(1, keepdims=True)
            P_new = np.clip(p["p_des"] * need_sq ** 2, p_lo, p_hi)
            C = c_new
            done = np.allclose(P_new, P)
            P = P_new
            if done:
                break
        sq = np.sqrt(P / p["p_des"])
        cmin = p["d_min"] * fm * sq * RHOCP
        cmax = fm * sq * RHOCP
        C = np.clip(C, cmin, cmax)
        tret = (C * zt_).sum(1, keepdims=True) / C.sum(1, keepdims=True)
        tmix = R * tret + (1.0 - R) * t_out
        ts = np.minimum(tmix, TSET)
        q = np.maximum(ulo - C * (ts - lo_), 0.0)
        q_cap = np.minimum(C * np.maximum(tdm - ts, 0.0), cw_cap * np.maximum(t_w - ts, 0.0))
        q = np.minimum(q, q_cap)
        kk = k0[None, :] + C
        dev = (np.where(has_lo[None, :], np.maximum(ulo - (q + C * (ts - lo_)), 0.0), 0.0)
               + np.where(has_hi[None, :], np.maximum((q + C * (ts - hi_)) - uhi, 0.0), 0.0)) / kk
        flow = C.sum(1, keepdims=True) / RHOCP
        fan = flow * P / p["fan_eff"] * (2.0 - R)
        cool = flow * RHOCP * np.maximum(tmix - ts, 0.0) / p["cop"]
        gas = q.sum(1, keepdims=True)
        cost = (we * (fan + cool) + wg * gas + c_price * dev.sum(1, keepdims=True)).ravel()
        j = int(np.argmin(cost))
        if cost[j] >= cost_off and not force:
            return None
        Cj = C[j]
        tsj = float(ts[j, 0])
        qj = q[j]
        cw = np.where(qj > 0, qj / max(t_w - tsj, 1.0), 0.0)
        d = np.clip(Cj / (fmax * float(sq[j, 0]) * RHOCP), p["d_min"], 1.0)
        return dict(tset=float(TSET[j, 0]), p=float(P[j, 0]), r=float(R[j, 0]),
                    d=d, c=Cj, cw=cw, cost=float(cost[j]), dev=float(dev[j].sum()),
                    cool=float(cool[j, 0]))
