"""Lightweight scenario runner (no charts) with reward/energy breakdown.

Usage (from the starter-kit root, so config.gin relative paths resolve):
    python /home/user/smart-buildings/tools/sbrun.py --scenario <dir> \
        --policy <module>:<Class> [--steps N] [--out summary.json] [--trace t.npz]
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

KIT = Path(os.environ.get("SB_KIT", "/root/starter-kit"))
REPO = Path(__file__).resolve().parents[1]


def load_policy_class(spec: str):
    if spec in ("all_off", "bang_bang", "fixed_setpoint", "pid", "pid_water", "transfer"):
        from smart_buildings import evaluate as ev
        return ev.POLICY_MAP[spec]
    mod, cls = spec.split(":")
    sys.path.insert(0, str(REPO / "policy"))
    return getattr(importlib.import_module(mod), cls)


def run(scenario: Path, policy_spec: str, steps: int = 8640, trace: str | None = None,
        policy_kwargs: dict | None = None, quiet: bool = True) -> dict:
    os.chdir(KIT)
    if str(KIT) not in sys.path:
        sys.path.insert(0, str(KIT))
    if quiet:
        from absl import logging as absl_logging
        absl_logging.set_verbosity(absl_logging.ERROR)
    from smart_buildings import evaluate as ev

    scenario = Path(scenario)
    if not scenario.is_absolute():
        scenario = KIT / scenario
    env = ev.load_hybrid_environment(scenario / "config.gin")
    env.metrics_path = None
    env._occupancy_normalization_constant = 125.0
    md = ev.create_building_meta_data(env, scenario)
    spec = env.action_spec()
    cls = load_policy_class(policy_spec)
    t_init = time.time()
    policy = cls(md, spec, **(policy_kwargs or {})) if policy_kwargs else ev.initialize_policy(cls, md, spec)
    t_init = time.time() - t_init

    obs, _ = env.reset()
    sim = env.building.simulator
    hvac = sim._hvac
    ahus = hvac.air_handler.ahus
    hws = hvac.hot_water_system
    vavs = hvac.vavs
    vav_ids = list(vavs.keys())

    mass_idx = None
    if trace is not None:
        try:
            solver = sim._solver
            bld = sim.building
            ncol = bld.temp.shape[1]
            pos = {int(f_): i_ for i_, f_ in enumerate(solver._solved)}
            mass_idx = []
            for v in vav_ids:
                coords = bld.room_dict[v.replace("VAV_", "room_")]
                mass_idx.append(np.array([pos[r_ * ncol + c_] for r_, c_ in coords]))
        except Exception as e:  # pragma: no cover
            print("mass idx failed", e, file=sys.stderr)
            mass_idx = None
    rel_stats = dict(true=0, false=0, false_wt=0.0, empty=0)
    rel_prev = None
    pol_zone_ids = list(getattr(policy, "zone_ids", vav_ids))
    tot = dict(reward=0.0, comfort=0.0, cost=0.0, carbon=0.0)
    e_tot = dict(fan=0.0, cool=0.0, pump=0.0, gas=0.0)
    rows = []
    t_pol = 0.0
    t0 = time.time()
    n = 0
    max_dev = 0.0
    for i in range(steps):
        tp = time.time()
        a = policy.get_action(obs)
        t_pol += time.time() - tp
        a = ev.validate_action(a, spec)
        obs, r, term, trunc, _ = env.step(a)
        info, resp = env.get_reward_info_and_response()
        n += 1
        tot["reward"] += float(r)
        tot["comfort"] += resp.normalized_productivity_regret
        tot["cost"] += resp.normalized_energy_cost
        tot["carbon"] += resp.normalized_carbon_emission
        fan = cool = pump = gas = 0.0
        for v in info.air_handler_reward_infos.values():
            fan += v.blower_electrical_energy_rate
            cool += abs(v.air_conditioning_electrical_energy_rate)
        for v in info.boiler_reward_infos.values():
            pump += v.pump_electrical_energy_rate
            gas += v.natural_gas_heating_energy_rate
        e_tot["fan"] += fan; e_tot["cool"] += cool; e_tot["pump"] += pump; e_tot["gas"] += gas
        # release accounting: flags used for the action just applied vs true occupancy now
        dbg_ = getattr(policy, "dbg", None)
        rel_prev = (np.asarray(dbg_["released"]) > 0) if (dbg_ and "released" in dbg_) else None
        if rel_prev is not None:
            zo = np.array([info.zone_reward_infos[v.replace("VAV_", "room_")].average_occupancy
                           for v in pol_zone_ids])
            if zo.sum() > 0:
                rel_stats["true"] += int((rel_prev & (zo <= 0)).sum())
                rel_stats["false"] += int((rel_prev & (zo > 0)).sum())
                rel_stats["false_wt"] += float((rel_prev & (zo > 0)).sum() / zo.sum())
                rel_stats["empty"] += int((zo <= 0).sum())
        if trace is not None:
            zobs = np.array([vavs[v].zone_air_temperature for v in vav_ids])
            _avg = sim.building.get_zone_average_temps()
            zt = np.array([_avg[v.replace("VAV_", "room_")] for v in vav_ids])
            occ = np.array([info.zone_reward_infos[v.replace("VAV_", "room_")].average_occupancy
                            for v in vav_ids])
            hs = next(iter(info.zone_reward_infos.values())).heating_setpoint_temperature
            cs = next(iter(info.zone_reward_infos.values())).cooling_setpoint_temperature
            rows.append(dict(
                r=float(r), comfort=resp.normalized_productivity_regret,
                cost=resp.normalized_energy_cost, carbon=resp.normalized_carbon_emission,
                fan=fan, cool=cool, pump=pump, gas=gas, occ=float(occ.sum()),
                oat=float(ahus[0].outside_air_temperature_sensor),
                zt=zt, zobs=zobs, zocc=occ, hs=hs, cs=cs,
                wt=float(hws.supply_water_temperature_sensor),
                flow=np.array([x.air_flow_rate for x in ahus]),
                damp=np.array([vavs[v].damper_setting for v in vav_ids]),
                rh=np.array([vavs[v].reheat_valve_setting for v in vav_ids]),
                ts=str(env.building.current_timestamp),
            ))
            if mass_idx is not None:
                mt = sim._solver._mass_temp
                rows[-1]["zmass"] = np.array([mt[ix].mean() for ix in mass_idx])
            dbg = getattr(policy, "dbg", None)
            if dbg:
                for k in ("ahu_on", "ahu_p", "ahu_r", "ahu_set", "cs", "W", "pred_err", "theta", "start", "slow_a",
                          "released", "dep", "dwm", "occ_obs"):
                    if k in dbg:
                        rows[-1]["dbg_" + k] = np.array(dbg[k], dtype=float)
                rows[-1]["dbg_active"] = float(dbg.get("active", 0))
        if term or trunc:
            break
    wall = time.time() - t0
    out = dict(
        scenario=f"{scenario.parent.name}/{scenario.name}", policy=policy_spec,
        total_steps=n, total_reward=tot["reward"], score=-tot["reward"] / max(n, 1),
        comfort_sum=tot["comfort"], cost_sum=tot["cost"], carbon_sum=tot["carbon"],
        comfort_part=-0.6 * tot["comfort"] / n, cost_part=0.2 * tot["cost"] / n,
        carbon_part=0.2 * tot["carbon"] / n,
        mean_kw={k: v / n / 1000.0 for k, v in e_tot.items()},
        release=rel_stats, wall_s=wall, policy_s=t_pol, policy_init_s=t_init, n_ahu=len(ahus), n_zone=len(vav_ids),
    )
    if trace is not None:
        keys = [k for k in rows[-1].keys() if all(k in row for row in rows[1:])]
        np.savez_compressed(trace, vav_ids=np.array(vav_ids),
                            zone_ids=np.array(getattr(policy, "zone_ids", [])),
                            **{k: np.array([row[k] for row in rows[1:]]) for k in keys})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--out", default=None)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--kwargs", default=None, help="JSON dict of policy kwargs")
    a = ap.parse_args()
    res = run(Path(a.scenario), a.policy, a.steps, a.trace,
              json.loads(a.kwargs) if a.kwargs else None)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
