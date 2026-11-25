"""Offline: strict morning arrival counting; how often is a 'valid' zone's arrival count wrong at 13:00?"""
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
arr = np.zeros(N); valid = np.ones(N, bool); drift = np.zeros(N)
stats = dict(days=0, valid=0, valid_ok=0, under=0, over=0, zero_claim=0, zero_wrong=0, one_claim=0, one_wrong=0)
for i in range(1, n - LAG):
    h = hours[i]
    if h < 6.0 or tot[i:i + 3].sum() == 0 and h < 6.2:
        arr[:] = 0; valid[:] = True; drift[:] = 0
    if 6.0 <= h < 13.0 and (tot[i] > 0 or tot[i - 1] > 0 or True):
        dN = int(round(tot[i] - tot[i - 1]))
        x = dw[i + LAG] - np.median(dw[i + LAG]) - drift
        cap = m - arr
        nz = np.clip(np.round(x / HEAT), 0, cap)
        match = np.abs(x - HEAT * nz) <= TOL
        cand = match & (nz >= 1)
        units = int(nz[cand].sum())
        chosen = np.zeros(N)
        if dN > 0 and 0 < units <= dN:
            chosen = np.where(cand, nz, 0); arr += chosen
        elif units > 0:
            valid &= ~cand
        resid = x - HEAT * chosen
        bad = np.abs(resid) > TOL
        valid &= ~bad
        quiet = (chosen == 0) & ~bad
        drift = np.where(quiet, drift + 0.3 * x, drift)
    if abs(h - 13.0) < 1e-6 and tot[i] > 0:
        truth = zocc[i]
        stats['days'] += 1; stats['valid'] += int(valid.sum()); stats['valid_ok'] += int((valid & (arr == truth)).sum())
        stats['under'] += int((valid & (arr < truth)).sum()); stats['over'] += int((valid & (arr > truth)).sum())
        stats['zero_claim'] += int((valid & (arr == 0)).sum()); stats['zero_wrong'] += int((valid & (arr == 0) & (truth > 0)).sum())
        stats['one_claim'] += int((valid & (arr == 1) & (m > 1)).sum()); stats['one_wrong'] += int((valid & (arr == 1) & (truth > 1) & (m > 1)).sum())
print(sys.argv[1].split('/')[-1], 'zones', N, 'm', m, 'tol', TOL, stats)
