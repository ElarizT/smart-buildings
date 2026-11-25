"""Offline: noise by period and single-step vs windowed assignment accuracy."""
import sys, numpy as np
d = np.load(sys.argv[1]); alpha = 0.6; LAG = 3; HEAT = 70.0
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zocc = d['zocc'][:, perm]; W = d['dbg_W']; ts = d['ts']
n, N = W.shape; m = int(zocc.max())
wm = np.zeros_like(W); wm[1:] = (W[1:] - (1 - alpha) * W[:-1]) / alpha; wm[0] = W[0]
dw = np.zeros_like(wm); dw[1:] = wm[1:] - wm[:-1]
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
tot = zocc.sum(1); dtot = np.zeros(n); dtot[1:] = tot[1:] - tot[:-1]
print(sys.argv[1].split('/')[-1], 'zones', N, 'occ/zone', m)
for name, h0, h1 in (('06-08', 6, 8), ('08-10', 8, 10), ('10-13', 10, 13), ('13-15', 13, 15), ('15-17', 15, 17), ('17-19', 17, 19)):
    idx = [i for i in range(2, n - LAG - 8) if h0 <= hours[i] < h1 and tot[i] > 0]
    noev = [i for i in idx if dtot[i] == 0 and np.all(zocc[i] == zocc[i - 1])]
    x = np.array([dw[i + LAG] - np.median(dw[i + LAG]) for i in noev]) if noev else np.zeros((1, N))
    sd = x.std(0)
    # single-event steps accuracy (one person changes in the building)
    ok1 = okw = cnt = 0
    for i in idx:
        dz = zocc[i] - zocc[i - 1]
        ev = np.flatnonzero(dz != 0)
        if len(ev) != 1 or abs(dz[ev[0]]) != 1: continue
        z = ev[0]; s = np.sign(dz[z])
        cand = (zocc[i - 1] < m) if s > 0 else (zocc[i - 1] > 0)
        a = (dw[i + LAG] - np.median(dw[i + LAG])) * s
        a1 = np.where(cand, a, -1e9)
        ok1 += int(np.argmax(a1) == z)
        # windowed level difference, J steps after vs before (uses future steps)
        J = 6
        lvl = wm[i + LAG:i + LAG + J].mean(0) - wm[i + LAG - J:i + LAG].mean(0)
        lvl = (lvl - np.median(lvl)) * s
        okw += int(np.argmax(np.where(cand, lvl, -1e9)) == z)
        cnt += 1
    print(f'  {name}: innovation noise std median {np.median(sd):6.1f} W  p90 {np.percentile(sd,90):6.1f} | single-person events {cnt:4d}  top-1 single-step {ok1/max(cnt,1):.2f}  windowed(J=6) {okw/max(cnt,1):.2f}')
