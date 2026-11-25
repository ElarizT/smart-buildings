"""Online-style check: pooled alpha, per-zone beta/const with ridge priors."""
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
Wb = W - np.where(cn[:, None], 8.0 * A[None, :], 0.0)
th = Wb / kss[None, :]; Ta = zobs; n = th.shape[1]
act = d['dbg_active'] > 0
def fit(ntr, a0=0.02, b0=0.001, lam_a=50.0, lam_b=2e4, lam_c=1e3):
    # per zone ridge: y = a*x1 + b*x2 + c
    out = np.zeros((n, 3))
    for z in range(n):
        y = th[1:ntr + 1, z] - th[:ntr, z]
        x1 = Ta[1:ntr + 1, z] - th[:ntr, z]; x2 = oat[:ntr] - th[:ntr, z]
        m = np.abs(y) < 0.5
        X = np.stack([x1, x2, np.ones(ntr)], 1)[m]; yy = y[m]
        P = np.diag([lam_a, lam_b, lam_c]); prior = np.array([a0, b0, 0.0])
        s = np.linalg.solve(X.T @ X + P, X.T @ yy + P @ prior)
        out[z] = s
    return out
H = 72
for ntr in [280, 300, 400, 580, 700, 860, 1000, 1150]:
    res = fit(ntr)
    errs = []
    for s0 in range(ntr, min(ntr + 288, len(th) - H - 1), 12):
        t = th[s0].copy()
        for k in range(s0, s0 + H):
            t = t + res[:, 0] * (Ta[k + 1] - t) + res[:, 1] * (oat[k] - t) + res[:, 2]
        errs.append(t - th[s0 + H])
    errs = np.array(errs)
    print(d['ts'][ntr][5:16], 'ntr', ntr, 'alpha med %.4f [%.4f %.4f] beta med %.5f' % (np.median(res[:, 0]), res[:, 0].min(), res[:, 0].max(), np.median(res[:, 1])),
          '| next-day 6h pred rms %.3f max %.3f' % (np.sqrt((errs ** 2).mean()), np.abs(errs).max()))
