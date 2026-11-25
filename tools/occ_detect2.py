"""Offline: reconstruct raw disturbance measurements and test matched-lag departure/arrival detection."""
import sys, numpy as np
d = np.load(sys.argv[1]); alpha = 0.6
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zocc = d['zocc'][:, perm]; W = d['dbg_W']; ts = d['ts']
n, N = W.shape
wm = np.zeros_like(W); wm[1:] = (W[1:] - (1 - alpha) * W[:-1]) / alpha; wm[0] = W[0]
dw = np.zeros_like(wm); dw[1:] = wm[1:] - wm[:-1]
docc = np.zeros_like(zocc); docc[1:] = zocc[1:] - zocc[:-1]       # docc[i]: change entering row i
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
occ_tot = zocc.sum(1)
print('rows', n, 'zones', N, 'max zone occ', zocc.max(), 'max total', occ_tot.max())
# event-triggered response by lag
for per, sel in (('dep', lambda i: hours[i] >= 13.0), ('arr', lambda i: hours[i] < 13.0)):
    for L in range(0, 6):
        v = []
        for i in range(3, n - 8):
            if not sel(i): continue
            for z in np.flatnonzero(docc[i] != 0):
                x = dw[i + L] - np.median(dw[i + L])
                v.append(x[z] / docc[i, z])
        print(per, 'lag', L, 'mean response per person %.1f W' % np.mean(v), 'n', len(v))
