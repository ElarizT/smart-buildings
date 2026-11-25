"""Offline: afternoon-only departure counting with a cleanliness gate; measure false releases."""
import sys, numpy as np
d = np.load(sys.argv[1]); alpha = 0.6; LAG = 3; HEAT = 70.0
TOL = float(sys.argv[2]) if len(sys.argv) > 2 else 22.0
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zocc = d['zocc'][:, perm]; W = d['dbg_W']; ts = d['ts']
n, N = W.shape; m = int(zocc.max())
wm = np.zeros_like(W); wm[1:] = (W[1:] - (1 - alpha) * W[:-1]) / alpha; wm[0] = W[0]
dw = np.zeros_like(wm); dw[1:] = wm[1:] - wm[:-1]
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
tot = zocc.sum(1)
dep = np.zeros(N); dirty = np.zeros(N, bool)
rel_true = rel_false = empty_total = 0; wt_false = 0.0; ev_units = ev_bad = 0; nday = 0
for i in range(1, n - LAG):
    if hours[i] < 13.0 or tot[i - 1] == 0:
        if hours[i] < 13.0:
            if dep.any() or dirty.any(): nday += 1
            dep[:] = 0; dirty[:] = False
        continue
    dN = int(round(tot[i] - tot[i - 1]))
    x = dw[i + LAG] - np.median(dw[i + LAG])
    a = -x / HEAT
    if dN < 0:
        cap = m - dep
        gains = sorted(((2 * a[z] - (2 * u + 1), z) for z in range(N) for u in range(int(cap[z]))), reverse=True)
        chosen = np.zeros(N)
        for _, z in gains[:-dN]:
            chosen[z] += 1
        dz = -(zocc[i] - zocc[i - 1])
        ev_units += -dN; ev_bad += int(np.abs(chosen - dz).sum() // 2)
        dep += chosen
        resid = x + HEAT * chosen
    else:
        resid = x
        chosen = np.zeros(N)
    bad = np.abs(resid) > TOL
    dirty |= bad & (dep < m)          # unexplained jump in a zone not yet fully departed
    if hours[i] < 19.0 and tot[i] > 0:
        released = (dep >= m) & ~dirty
        te = zocc[i] <= 0
        rel_true += int((released & te).sum()); rel_false += int((released & ~te).sum()); empty_total += int(te.sum())
        wt_false += (released & ~te).sum() / tot[i]
print(f'{sys.argv[1].split("/")[-1]}: zones {N} occ/zone {m} tol {TOL}: departures {ev_units}, misassigned {ev_bad} | released zone-steps correct {rel_true} of {empty_total} empty ({rel_true/max(empty_total,1):.2f}), FALSE {rel_false}, weighted false {wt_false:.3f}')
