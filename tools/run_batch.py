"""Run a policy over many scenarios in parallel (one process per scenario).

    python tools/run_batch.py --policy sb_policy:SBPolicy --tag v2 --set train
Results: results/<tag>/<building>__<weather>.json  (existing files are skipped)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
KIT = Path(os.environ.get("SB_KIT", "/root/starter-kit"))
PY = os.environ.get("SB_PY", str(Path.home() / "sbvenv/bin/python"))


def scenario_list(name: str) -> list[str]:
    if name == "train":
        root = KIT / "smart_buildings/training_scenarios"
    elif name == "val":
        root = KIT / "smart_buildings/val_scenarios"
    else:
        root = Path(name)
    return sorted(str(p.parent.relative_to(KIT)) if p.is_relative_to(KIT) else str(p.parent)
                  for p in root.glob("*/*/config.gin"))


def run_one(scen: str, args) -> dict:
    key = "__".join(Path(scen).parts[-2:])
    out = REPO / "results" / args.tag / f"{key}.json"
    if out.exists():
        return json.load(open(out))
    cmd = [PY, str(REPO / "tools/sbrun.py"), "--scenario", scen, "--policy", args.policy,
           "--steps", str(args.steps), "--out", str(out)]
    if args.kwargs:
        cmd += ["--kwargs", args.kwargs]
    if args.trace:
        tdir = Path(args.trace); tdir.mkdir(parents=True, exist_ok=True)
        cmd += ["--trace", str(tdir / f"{key}.npz")]
    r = subprocess.run(cmd, cwd=KIT, capture_output=True, text=True)
    if not out.exists():
        return dict(scenario=scen, error=r.stderr[-2000:])
    return json.load(open(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--set", default="train")
    ap.add_argument("--only", nargs="*", default=None, help="substring filters")
    ap.add_argument("--steps", type=int, default=8640)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--kwargs", default=None)
    ap.add_argument("--trace", default=None)
    a = ap.parse_args()
    scens = scenario_list(a.set)
    if a.only:
        scens = [s for s in scens if any(o in s for o in a.only)]
    t0 = time.time()
    with ThreadPoolExecutor(a.procs) as ex:
        res = list(ex.map(lambda s: run_one(s, a), scens))
    ok = [r for r in res if "score" in r]
    for r in res:
        if "score" in r:
            print(f"{r['scenario']:36s} score {r['score']:.5f}  comfort {r['comfort_part']:.5f} "
                  f"cost {r['cost_part']:.5f} carbon {r['carbon_part']:.5f}  "
                  f"kW fan {r['mean_kw']['fan']:.2f} cool {r['mean_kw']['cool']:.2f} gas {r['mean_kw']['gas']:.2f}")
        else:
            print("ERROR", r.get("scenario"), r.get("error", "")[-400:])
    if ok:
        m = sum(r["score"] for r in ok) / len(ok)
        print(f"== {a.tag}: mean score {m:.5f} over {len(ok)} scenarios "
              f"(comfort {sum(r['comfort_part'] for r in ok)/len(ok):.5f}) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
