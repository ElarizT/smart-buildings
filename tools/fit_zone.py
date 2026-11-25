"""Fit per-zone one-step model  K0*T+ - Cdt*T = W + q + cs*(ts - T+)  from a trace."""
import sys, numpy as np
sys.path.insert(0, '/home/user/smart-buildings/policy')
d = np.load(sys.argv[1])
vid = list(d['vav_ids']); zid = list(d['zone_ids'])
perm = [vid.index(z) for z in zid]
zt = d['zt'][:, perm]
q = d['dbg_q']; cs = d['dbg_cs']; ts = d['dbg_ts']; act = d['dbg_active'] > 0
# row i: dbg = plan computed from obs zt[i] (state after step i+1), result = zt[i+1]
T = zt[:-1]; Tn = zt[1:]
qq = q[:-1]; cc = cs[:-1]; tt = ts[:-1]
inp = qq + cc * (tt - Tn)            # W delivered at end-of-step temperature
n = T.shape[1]
res = []
for z in range(n):
    # use differences over consecutive rows where plant was active in both
    m = act[:-1].copy(); m[1:] &= act[:-2]; m[0] = False
    idx = np.flatnonzero(m)
    dTn = Tn[idx, z] - Tn[idx - 1, z]
    dT = T[idx, z] - T[idx - 1, z]
    dI = inp[idx, z] - inp[idx - 1, z]
    A = np.stack([dTn, -dT], 1)
    sol, *_ = np.linalg.lstsq(A, dI, rcond=None)
    pred = A @ sol
    r2 = 1 - ((dI - pred) ** 2).sum() / ((dI - dI.mean()) ** 2).sum()
    res.append((sol[0], sol[1], r2, len(idx)))
res = np.array(res)
from sb_policy import SBPolicy
print('zone  K0_fit  Cdt_fit   r2   n')
for z in range(n):
    print(zid[z], np.round(res[z], 2))
np.save(sys.argv[2], res) if len(sys.argv) > 2 else None
