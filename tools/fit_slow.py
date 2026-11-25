"""Check a lumped slow-state model: theta = Wb/Kss, dtheta = a*(Ta - theta) + b*(Tout - theta) + c."""
import sys, numpy as np
sys.path.insert(0, '/home/user/smart-buildings/policy'); sys.path.insert(0, '/root/starter-kit')
from pathlib import Path
from absl import logging; logging.set_verbosity(logging.ERROR)
from smart_buildings import evaluate as ev
from sb_policy import SBPolicy
d = np.load(sys.argv[1]); scen = Path(sys.argv[2])
env = ev.load_hybrid_environment(scen / 'config.gin'); md = ev.create_building_meta_data(env, scen)
pol = SBPolicy(md, None); A = pol.area; kss = pol.k0 - pol.cdt
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zobs = d['zobs'][:, perm]; W = d['dbg_W']; cn = d['hs'] > 290; oat = d['oat']
# dbg_W at row i = Wb + g*cnow(at call i+1).  Wb reflects transitions ending at y (lagged) => theta aligned with zobs row i
Wb = W - np.where(cn[:, None], 8.0 * A[None, :], 0.0)
th = Wb / kss[None, :]
Ta = zobs
n = th.shape[1]
np.set_printoptions(linewidth=200, precision=4, suppress=True)
res = []
ntr = int(sys.argv[3]) if len(sys.argv) > 3 else len(th) - 1
for z in range(n):
    y = th[1:ntr + 1, z] - th[:ntr, z]
    X = np.stack([Ta[1:ntr + 1, z] - th[:ntr, z], oat[:ntr] - th[:ntr, z], cn[:ntr].astype(float), np.ones(ntr)], 1)
    m = np.abs(y) < 0.5
    s, *_ = np.linalg.lstsq(X[m], y[m], rcond=None)
    res.append(s)
res = np.array(res)
print('alpha  median %.4f  range %.4f %.4f' % (np.median(res[:, 0]), res[:, 0].min(), res[:, 0].max()))
print('beta   median %.5f range %.5f %.5f' % (np.median(res[:, 1]), res[:, 1].min(), res[:, 1].max()))
print('gain   median %.5f, const median %.5f' % (np.median(res[:, 2]), np.median(res[:, 3])))
# multi-step open-loop simulation of theta given actual Ta trajectory, from several starts, horizon 72 steps
H = 72
errs = []
for s0 in range(ntr if ntr < len(th) - H - 1 else 300, len(th) - H - 1, 24):
    t = th[s0].copy()
    for k in range(s0, s0 + H):
        t = t + res[:, 0] * (Ta[k + 1] - t) + res[:, 1] * (oat[k] - t) + res[:, 2] * cn[k] + res[:, 3]
    errs.append(t - th[s0 + H])
errs = np.array(errs)
print('6h open-loop theta prediction error: rms %.3f K, max abs %.3f, n starts %d' % (np.sqrt((errs ** 2).mean()), np.abs(errs).max(), len(errs)))
print('theta - Ta typical at 05:50 rows:', [(d['ts'][i][5:16], np.round((th[i] - Ta[i])[[0, 1, 38 if n > 38 else n - 1]], 2)) for i in range(len(th)) if d['ts'][i][11:16] == '05:50'][:5])
