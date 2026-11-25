"""Offline check: can per-zone departures be detected from the disturbance estimate?"""
import sys, numpy as np
d = np.load(sys.argv[1])
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zocc = d['zocc'][:, perm]            # true occupancy during [t, t+5) at row i (reward of step i)
W = d['dbg_W']                       # policy's W estimate (includes scheduled gains) at the call after step i
ts = d['ts']
n, N = W.shape
print('rows', n, 'zones', N, 'max per-zone occ', zocc.max())
# innovation of W between consecutive calls
dW = np.diff(W, axis=0)              # dW[i] = W[i+1]-W[i]
docc = np.diff(zocc, axis=0)         # docc[i] = occ change between row i and i+1
tot = docc.sum(1)
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
for lagname, lags in (('lag1', [1]), ('lag1+2', [1, 2]), ('lag1..3', [1, 2, 3]), ('lag0..3', [0, 1, 2, 3])):
    for period, sel in (('afternoon', lambda h: h >= 13.0), ('morning', lambda h: h < 13.0)):
        ok = cnt = 0
        ranks = []
        snr = []
        for i in range(2, n - 6):
            if not sel(hours[i + 1]):
                continue
            ev = np.flatnonzero(docc[i] != 0)
            if len(ev) != 1 or abs(docc[i, ev[0]]) != 1:
                continue              # only single-person events
            z = ev[0]; sgn = docc[i, z]
            # W reflects the event some steps later (sensor lag + observer): sum innovations over lags
            x = sum(dW[i + L] for L in lags)
            x = x - np.median(x)      # remove common-mode drift
            cand = zocc[i] > 0 if sgn < 0 else zocc[i] < zocc.max()
            score = -x if sgn < 0 else x
            score = np.where(cand, score, -1e9)
            r = int((score > score[z]).sum())
            ranks.append(r); cnt += 1; ok += (r == 0)
            noise = np.std(x[np.arange(N) != z])
            snr.append(abs(x[z]) / max(noise, 1e-9))
        if cnt:
            print(f'{lagname:8s} {period:9s} single events {cnt:4d}  top-1 {ok/cnt:.2f}  top-3 {np.mean(np.array(ranks)<3):.2f}  median |x_true|/noise {np.median(snr):.2f}')
