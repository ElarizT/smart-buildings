"""Structure-aware HVAC controller for the Smart Buildings competition.

Self-contained (numpy only, scipy optional). The same file is embedded in the
Kaggle notebook.

Core ideas
----------
* Comfort is only scored while people are in the building (workdays, comfort
  hours), so the plant is off outside that window apart from a short pre-start.
* Each zone's air node is modelled one step ahead as

      T+ = (Cdt*T + W + Q_reheat + Cs*Ts) / (K0 + Cs)

  where Cs is the supply-air capacity rate, Ts the air-handler supply
  temperature, and W a slowly varying disturbance (thermal mass, envelope,
  gains) tracked by an observer.
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
STEP_S = 300.0

DEFAULT_PARAMS = dict(
    # comfort margins inside the band (K)
    m_heat=0.20,
    m_cool=0.20,
    # zone model
    cdt_per_m2=12.0,          # air heat capacity / step, W/K per m2
    g_per_m2=4.8,              # air <-> mass conductance, W/K per m2
    k_face=2.8,                # W/K per wall-adjacent cell face
    w_alpha=0.6,               # observer smoothing
    gain_step_w_m2=8.0,        # feed-forward jump of internal gains at schedule edges
    # pre-start
    lead_steps=3,              # steps before occupancy at which the band starts to apply
    ramp_k=4.0,                # K per step the pre-start band is relaxed by
    # holiday / end-of-day detection
    holiday_steps=6,
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
    pump_eff=0.98,
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
        self.ahu_ids = list(md["ahu_zone_assignments"].keys())
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
        self.zone_ids = []        # VAV ids in a fixed order
        self.zone_ahu = []
        for a, aid in enumerate(self.ahu_ids):
            for room in md["ahu_zone_assignments"][aid]:
                vid = room.replace("room_", "VAV_")
                if vid in zai:
                    self.zone_ids.append(vid)
                    self.zone_ahu.append(a)
        # zones that are not assigned to an AHU (should not happen)
        for vid in zai:
            if vid not in self.zone_ids:
                self.zone_ids.append(vid)
                self.zone_ahu.append(0)
        self.zone_ahu = np.asarray(self.zone_ahu, dtype=int)
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
        self.o_csoon = idx("comfort_mode_soon")
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
        faces = np.full(self.N, 200.0)
        try:
            zm = md.get("zone_map", None)
            if zm is not None:
                lab, n = _label_rooms(zm)
                if lab is not None and n >= self.N:
                    cell_m2 = 0.04
                    counts = np.bincount(lab.ravel(), minlength=n + 1)
                    air = lab > 0
                    # faces between a room cell and a non-room cell
                    fcount = np.zeros(n + 1)
                    for ax, sl_a, sl_b in (
                        (0, (slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                        (1, (slice(None), slice(1, None)), (slice(None), slice(None, -1))),
                    ):
                        a_, b_ = lab[sl_a], lab[sl_b]
                        m1 = (a_ > 0) & (b_ == 0)
                        m2 = (b_ > 0) & (a_ == 0)
                        fcount += np.bincount(a_[m1], minlength=n + 1)
                        fcount += np.bincount(b_[m2], minlength=n + 1)
                    for i, vid in enumerate(self.zone_ids):
                        rid = int(vid.split("_")[-1])
                        if 1 <= rid <= n:
                            area[i] = counts[rid] * cell_m2
                            faces[i] = fcount[rid]
        except Exception:
            pass
        self.area = area
        self.faces = faces
        self.cdt = p["cdt_per_m2"] * area
        self.k0 = self.cdt + p["g_per_m2"] * area + p["k_face"] * faces

        # ---- reward weights (true reward units per W) ------------------------
        self.e_max = 80000.0 * self.K
        self.g_max = 205000.0 * self.K

        # ---- state ---------------------------------------------------------
        self.W = None                 # disturbance estimate per zone (W)
        self.Wb = None                # same, without the scheduled internal gains
        self.p1 = None                # plan applied last step
        self.p2 = None                # plan applied the step before
        self.y_prev = None
        self.xhat_prev = None
        self.step = 0
        self.occ_seen_today = False
        self.comfort_steps_today = 0
        self.holiday = False
        self.day_done = False
        self.prev_cnow = None
        self.dbg = {}

    # ------------------------------------------------------------------ utils
    @staticmethod
    def _to_norm(x, lo, hi):
        if hi <= lo:
            return 0.0
        return float(np.clip(2.0 * (x - lo) / (hi - lo) - 1.0, -1.0, 1.0))

    def _kelvin(self, v, mean, std):
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
        zt = obs[self.o_ztemp]
        zt = np.where(zt > 150.0, zt, zt * self.zt_std + self.zt_mean)
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

        # ---- observer update / state prediction -------------------------------
        # The zone sensor is latched before the thermal solve, so `zt` is the
        # true temperature one step ago.  `p1` is the plan applied to that
        # state (its outcome is not visible yet); `p2` is the plan before it,
        # whose outcome is exactly `zt`.
        g_sched = p["gain_step_w_m2"] * self.area
        if self.Wb is None:
            self.Wb = self.k0 * zt - self.cdt * zt - (g_sched if cnow else 0.0)
        if self.p2 is not None and self.y_prev is not None:
            q2, ts2 = self._realize(self.p2, self.y_prev)
            cs2 = self.p2["cs"]
            w_meas = zt * (self.k0 + cs2) - self.cdt * self.y_prev - q2 - cs2 * ts2
            w_meas = w_meas - (g_sched if self.p2["cnow"] else 0.0)
            al = p["w_alpha"]
            self.Wb = (1 - al) * self.Wb + al * w_meas
        if self.p1 is not None:
            q1, ts1 = self._realize(self.p1, zt)
            cs1 = self.p1["cs"]
            w1 = self.Wb + (g_sched if self.p1["cnow"] else 0.0)
            xhat = (self.cdt * zt + w1 + q1 + cs1 * ts1) / (self.k0 + cs1)
            if self.xhat_prev is not None:
                self.dbg["pred_err"] = zt - self.xhat_prev
        else:
            xhat = zt.copy()
        self.y_prev = zt
        self.xhat_prev = xhat
        self.W = self.Wb + (g_sched if cnow else 0.0)
        zt = xhat

        # ---- schedule state ---------------------------------------------------
        # the action taken now is judged at t+5min
        hj = hour + 5.0 / 60.0
        dj = dow
        if hj >= 24.0:
            hj -= 24.0
            dj = (dow + 1) % 7
        workday_j = dj < 5
        in_window = workday_j and (self.h_start <= hj < self.h_end)

        if cnow and not self.prev_cnow:
            self.occ_seen_today = False
            self.comfort_steps_today = 0
            self.holiday = False
            self.day_done = False
        if cnow:
            self.comfort_steps_today += 1
            if occ > 0:
                self.occ_seen_today = True
            if (not self.occ_seen_today) and self.comfort_steps_today > p["holiday_steps"]:
                self.holiday = True
            if hour >= 13.0 + 1e-6 and occ <= 0 and self.occ_seen_today:
                self.day_done = True
        else:
            self.holiday = False
            self.day_done = False
        self.prev_cnow = cnow

        # steps until the first judged step of the next comfort window
        steps_to_start = None
        if not in_window:
            # hours from judged time to next workday start
            h = hj
            d = dj
            wait = 0.0
            if d < 5 and h < self.h_start:
                wait = self.h_start - h
            else:
                wait = 24.0 - h + self.h_start
                d = (d + 1) % 7
                while d >= 5:
                    wait += 24.0
                    d = (d + 1) % 7
            steps_to_start = int(round(wait * 12.0))

        lo = self.band_lo + p["m_heat"]
        hi = self.band_hi - p["m_cool"]
        active = False
        if in_window and not self.holiday and not self.day_done:
            active = True
        elif steps_to_start is not None and steps_to_start <= p["lead_steps"]:
            active = True
            lo -= p["ramp_k"] * steps_to_start
            hi += p["ramp_k"] * steps_to_start

        we, wg = self._weights(hour, dow < 5)

        cont = np.zeros(self.n_cont, dtype=np.float32)
        disc = np.zeros(self.n_disc, dtype=np.int32)
        d_cmd = np.full(N, p["d_min"])
        v_cmd = np.zeros(N)
        q_plan = np.zeros(N)
        cs_plan = np.zeros(N)
        ts_plan = np.zeros(N)
        cw_need = np.zeros(N)
        ahu_on = np.zeros(K, dtype=bool)
        ahu_set = np.full(K, self.sat_rng[1])
        ahu_p = np.full(K, p["p_min"])
        ahu_r = np.full(K, p["r_max"])

        if active:
            u_lo = self.k0 * lo - self.cdt * zt - self.W
            u_hi = self.k0 * hi - self.cdt * zt - self.W
            for a in range(K):
                z = self.ahu_zones[a]
                if z.size == 0:
                    continue
                res = self._plan_ahu(zt[z], u_lo[z], u_hi[z], fmax[z], self.k0[z],
                                     lo, hi, t_out, t_w, we, wg)
                if res is None:
                    continue
                ahu_on[a] = True
                ahu_set[a], ahu_p[a], ahu_r[a] = res["tset"], res["p"], res["r"]
                d_cmd[z] = res["d"]
                q_plan[z] = res["q"]
                cs_plan[z] = res["c"]
                ts_plan[z] = res["ts"]
                cw_need[z] = res["cw"]

        # ---- hot water plant ---------------------------------------------
        kw = p["kv"] / 3600.0 * WATER_C        # W/K per sqrt(bar) at valve = 1
        heat_on = bool(np.any(cw_need > 1e-6))
        dp = p["dp_min"]
        if heat_on:
            dp = float(np.clip((cw_need.max() / kw) ** 2, p["dp_min"], p["dp_max"]))
            v_cmd = np.clip(cw_need / (kw * math.sqrt(dp)), 0.0, 1.0)

        # ---- write actions ---------------------------------------------------
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
        for i in range(N):
            cont[self.damper_idx[i]] = self._to_norm(d_cmd[i], *self.dmp_rng)
            cont[self.valve_idx[i]] = self._to_norm(v_cmd[i], *self.vlv_rng)

        # ---- remember the plan for the observer ---------------------------
        self.p2 = self.p1
        self.p1 = dict(cs=cs_plan, cw=cw_need, r=ahu_r.copy(), tset=ahu_set.copy(),
                       on=ahu_on.copy(), t_out=t_out, t_w=t_w, cnow=cnow)
        self.dbg.update(hour=hour, dow=dow, active=active, lo=lo, hi=hi, ahu_on=ahu_on.copy(),
                        ahu_set=ahu_set.copy(), ahu_p=ahu_p.copy(), ahu_r=ahu_r.copy(),
                        q=q_plan.copy(), cs=cs_plan.copy(), ts=ts_plan.copy(), W=self.W.copy(), dp=dp,
                        holiday=self.holiday, day_done=self.day_done)
        self.step += 1
        return {"continuous_action": cont, "discrete_action": disc}

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
        if getattr(self, "_cand", None) is None:
            rs = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            rs = rs[rs <= self.p["r_max"] + 1e-9]
            lo_s, hi_s = self.sat_rng
            ts = np.array([lo_s, lo_s + 2, lo_s + 4, lo_s + 6, lo_s + 8, lo_s + 10, lo_s + 12, hi_s])
            R, T = np.meshgrid(rs, ts, indexing="ij")
            self._cand = (R.ravel()[:, None], T.ravel()[:, None])
        return self._cand

    def _plan_ahu(self, zt, u_lo, u_hi, fmax, k0, lo, hi, t_out, t_w, we, wg):
        """Cheapest way to keep these zones in [lo, hi] at the next step.

        Returns None when the air handler should stay off.
        """
        p = self.p
        n_tot = max(self.N, 1)
        c_price = p["comfort_scale"] * 0.6 / 5.0 / n_tot     # reward units per K per zone

        # cost of staying off
        dev_off = np.maximum(u_lo, 0.0) / k0 + np.maximum(-u_hi, 0.0) / k0
        cost_off = c_price * dev_off.sum()
        if cost_off <= 0.0:
            return None

        R, TSET = self._candidates()            # (m,1)
        zt_ = zt[None, :]
        ulo = u_lo[None, :]
        uhi = u_hi[None, :]
        fm = fmax[None, :]
        tdm = p["t_dis_max"]
        kw = p["kv"] / 3600.0 * WATER_C
        cw_cap = kw * math.sqrt(p["dp_max"])
        p_lo, p_hi = p["p_min"], self.sp_rng[1]

        P = np.full(R.shape, p_lo)
        C = None
        for _ in range(3):
            sq = np.sqrt(P / p["p_des"])
            cmin = p["d_min"] * fm * sq * RHOCP
            cmax = fm * sq * RHOCP
            if C is None:
                C = np.broadcast_to(cmin, (R.shape[0], zt.size)).copy()
            C = np.clip(C, cmin, cmax)
            tret = (C * zt_).sum(1, keepdims=True) / C.sum(1, keepdims=True)
            tmix = R * tret + (1.0 - R) * t_out
            ts = np.minimum(tmix, TSET)
            # cooling side: Q + C (ts - hi) <= uhi
            gap_c = hi - ts
            c_cool = np.where((uhi < 0) & (gap_c > 0.05), -uhi / np.maximum(gap_c, 0.05), cmin)
            c_need = np.maximum(cmin, c_cool)
            # heating side: Q + C (ts - lo) >= ulo
            q0 = ulo - c_need * (ts - lo)
            heat = q0 > 0
            c_heat = np.where(heat, ulo / max(tdm - lo, 1.0), 0.0)
            c_new = np.maximum(c_need, c_heat)
            # required pressure
            need_sq = (c_new / (fm * RHOCP)).max(1, keepdims=True)
            P_new = np.clip(p["p_des"] * need_sq ** 2, p_lo, p_hi)
            C = c_new
            if np.allclose(P_new, P):
                P = P_new
                break
            P = P_new
        sq = np.sqrt(P / p["p_des"])
        cmin = p["d_min"] * fm * sq * RHOCP
        cmax = fm * sq * RHOCP
        C = np.clip(C, cmin, cmax)
        tret = (C * zt_).sum(1, keepdims=True) / C.sum(1, keepdims=True)
        tmix = R * tret + (1.0 - R) * t_out
        ts = np.minimum(tmix, TSET)
        q = np.maximum(ulo - C * (ts - lo), 0.0)
        # coil limits: discharge temperature and water capacity
        q_cap = np.minimum(C * np.maximum(tdm - ts, 0.0), cw_cap * np.maximum(t_w - ts, 0.0))
        q = np.minimum(q, q_cap)
        net_lo = q + C * (ts - lo)
        net_hi = q + C * (ts - hi)
        dev = np.maximum(ulo - net_lo, 0.0) / (k0[None, :] + C) + np.maximum(net_hi - uhi, 0.0) / (k0[None, :] + C)
        flow = C.sum(1, keepdims=True) / RHOCP
        fan = flow * P / p["fan_eff"] * (2.0 - R)
        cool = flow * RHOCP * np.maximum(tmix - ts, 0.0) / p["cop"]
        gas = q.sum(1, keepdims=True)
        cost = we * (fan + cool) + wg * gas + c_price * dev.sum(1, keepdims=True)
        cost = cost.ravel()
        j = int(np.argmin(cost))
        if cost[j] >= cost_off:
            return None
        Cj = C[j]
        tsj = float(ts[j, 0])
        qj = q[j]
        cw = np.where(qj > 0, qj / max(t_w - tsj, 1.0), 0.0)
        d = np.clip(Cj / (fmax * float(sq[j, 0]) * RHOCP), p["d_min"], 1.0)
        return dict(tset=float(TSET[j, 0]), p=float(P[j, 0]), r=float(R[j, 0]),
                    d=d, q=qj, c=Cj, ts=np.full(zt.size, tsj), cw=cw, cost=float(cost[j]))
