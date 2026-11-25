import sys, numpy as np, glob
sys.path.insert(0, '/home/user/smart-buildings/policy'); sys.path.insert(0,'/root/starter-kit')
from pathlib import Path
from absl import logging; logging.set_verbosity(logging.ERROR)
from smart_buildings import evaluate as ev
from sb_policy import SBPolicy
scen = Path(sys.argv[1]); res = np.load(sys.argv[2])
env = ev.load_hybrid_environment(scen / 'config.gin')
md = ev.create_building_meta_data(env, scen)
pol = SBPolicy(md, env.action_spec())
A = pol.area; F = pol.faces
K0, Cdt = res[:, 0], res[:, 1]
print('area range', A.min(), A.max(), 'faces range', F.min(), F.max())
X = np.stack([A, F], 1)
for name, y in (('Cdt', Cdt), ('K0', K0), ('K0-Cdt', K0 - Cdt)):
    s, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ s
    print(name, 'coef area, faces', np.round(s, 3), 'rel rms err', np.sqrt(((p - y) ** 2).mean()) / y.mean())
    s1 = (A @ y) / (A @ A)
    print('   area only coef', round(s1, 3), 'rel rms', np.sqrt(((A * s1 - y) ** 2).mean()) / y.mean())
