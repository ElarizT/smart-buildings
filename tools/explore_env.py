"""Dump metadata / observation layout for one scenario."""
import sys, time, collections
from pathlib import Path
import numpy as np
from smart_buildings import evaluate as ev

scen = Path(sys.argv[1])
t=time.time()
env = ev.load_hybrid_environment(scen / "config.gin")
env.metrics_path = None
env._occupancy_normalization_constant = 125.0
print("load s", time.time()-t)
md = ev.create_building_meta_data(env, scen)
for k, v in md.items():
    if k in ("observation_field_names", "floor_plan", "zone_map", "observation_normalization_metadata", "zone_action_indices","zone_temp_indices"):
        continue
    print(k, ":", v)
fn = md["observation_field_names"]
print("n obs", len(fn))
suffix = collections.Counter()
for f in fn:
    print("   ", f)
print("zone_action_indices sample:", list(md["zone_action_indices"].items())[:2])
print("obs norm:")
for k, v in md["observation_normalization_metadata"].items():
    print("   ", k, v)
print("action norm:", md["action_normalization_metadata"])
print("floor_plan", md["floor_plan"].shape, np.unique(md["floor_plan"], return_counts=True))
print("zone_map", md["zone_map"].shape, np.unique(md["zone_map"])[:50])
print(env.action_spec())
print("device action tuples", env._device_action_tuples[:12])
obs, _ = env.reset()
print("obs0", np.round(obs, 3).tolist())
print("steps per episode", env.steps_per_episode, "start", env.building.current_timestamp)
