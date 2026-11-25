"""Generate a held-out validation set: new buildings (sized like the shipped six) x unused weather strips.

Run from the starter-kit root:  python /home/user/smart-buildings/tools/make_val.py
"""
import hashlib, os, random, sys
from pathlib import Path
import yaml

KIT = Path(os.environ.get("SB_KIT", "/root/starter-kit"))
os.chdir(KIT); sys.path.insert(0, str(KIT))
from smart_buildings.smart_control.utils.config_generator import generate_config
from smart_buildings.smart_control.utils.materials import draw_materials
from smart_buildings.smart_control.utils.scenario_generator import (
    check_zone_floor_areas, scenario_config_from_dict, load_scenario_from_parts, generate_scenario_from_config)

RANGES = dict(width=(310, 440), height=(185, 305), num_floors=(2, 4), min_room_size=(25, 36),
              room_size_span=(11, 16), shape_ratio=(0.28, 0.40), composite_coverage=(0.65, 0.85),
              composite_style="architectural", num_ahus=(3, 7), occupants_per_zone=(1, 2))
SEEDS = list(range(600, 640))
N_BUILD = 12
UNUSED = ["atlanta_4", "atlanta_7", "denver_1", "denver_4", "fairbanks_3", "fairbanks_6",
          "phoenix_2", "phoenix_7", "singapore_2", "singapore_5"]

bdir = KIT / "smart_buildings/val_buildings"; bdir.mkdir(exist_ok=True)
sdir = Path("smart_buildings/val_scenarios")
digest = hashlib.sha256(repr(SEEDS[:N_BUILD]).encode()).digest()
mats = draw_materials(random.Random(int.from_bytes(digest[:8], "big")), N_BUILD)
built = []
for seed in SEEDS:
    if len(built) >= N_BUILD:
        break
    cfg = generate_config(name=f"building_val_{seed}", seed=seed, building_shape="composite", **RANGES)
    cfg["materials"] = mats[len(built)]
    try:
        check_zone_floor_areas(scenario_config_from_dict(cfg))
    except ValueError as e:
        print("skip", seed, e); continue
    p = bdir / f"{cfg['name']}.yaml"
    yaml.dump(cfg, open(p, "w"), default_flow_style=False, sort_keys=False)
    built.append(p)
    print("building", cfg["name"], cfg["floor_plan"], cfg["ahu"], cfg["simulation"].get("occupants_per_zone"))
for i, bp in enumerate(built):
    for w in (f"minneapolis_{i % 8 + 1}", UNUSED[i % len(UNUSED)]):
        sc = load_scenario_from_parts(str(bp), str(KIT / f"smart_buildings/weather/{w}.yaml"), output_base_dir=str(sdir))
        res = generate_scenario_from_config(sc)
        print("scenario", res["output_dir"])
