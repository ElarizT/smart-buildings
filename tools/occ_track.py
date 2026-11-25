"""Offline tracker: per-zone occupancy from count changes + matched-lag disturbance innovations."""
import sys, numpy as np
d = np.load(sys.argv[1]); alpha = 0.6; LAG = 3; HEAT = float(sys.argv[2]) if len(sys.argv) > 2 else 70.0
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zocc = d['zocc'][:, perm]; W = d['dbg_W']; ts = d['ts']
n, N = W.shape; m = int(zocc.max())
wm = np.zeros_like(W); wm[1:] = (W[1:] - (1 - alpha) * W[:-1]) / alpha; wm[0] = W[0]
dw = np.zeros_like(wm); dw[1:] = wm[1:] - wm[:-1]
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
tot = zocc.sum(1)
k = np.zeros(N)                      # belief
ev_ok = ev_n = 0
false_empty = true_empty = miss_empty = 0     # zone-steps after 13:00
false_empty_wt = 0.0
noise = []
for i in range(1, n - LAG):
    if tot[i] == 0 and tot[i - 1] == 0:
        k[:] = 0
    dN = int(round(tot[i] - tot[i - 1]))
    x = dw[i + LAG] - np.median(dw[i + LAG])
    if dN == 0:
        noise.append(x)
    else:
        a = x / HEAT * np.sign(dN)                   # ~ number of unit changes per zone
        cand_cap = (m - k) if dN > 0 else k           # how many units each zone can still take
        gains = []
        for z in range(N):
            for u in range(int(cand_cap[z])):
                gains.append((2 * a[z] - (2 * u + 1), z))
        gains.sort(reverse=True)
        chosen = [z for _, z in gains[:abs(dN)]]
        true = []
        dz = zocc[i] - zocc[i - 1]
        for z in np.flatnonzero(dz != 0):
            true += [z] * int(abs(dz[z]))
        ev_n += abs(dN); ev_ok += len(set(chosen) & set(true)) if len(true) == len(set(true)) else sum(min(chosen.count(z), true.count(z)) for z in set(true))
        for z in chosen:
            k[z] += np.sign(dN)
    if hours[i] >= 13.0 and tot[i] > 0:
        be = k <= 0; te = zocc[i] <= 0
        true_empty += int((be & te).sum()); false_empty += int((be & ~te).sum()); miss_empty += int((~be & te).sum())
        false_empty_wt += (be & ~te).sum() / tot[i]
noise = np.array(noise)
print(f'{sys.argv[1].split("/")[-1]}: zones {N} occ/zone {m}  event units {ev_n}  correctly assigned {ev_ok/ev_n:.3f}')
print(f'  innovation noise std (no-event steps): median {np.median(noise.std(0)):.1f} W, 90th pct {np.percentile(noise.std(0),90):.1f} W')
print(f'  afternoon zone-steps: empty&believed-empty {true_empty}, believed-empty but OCCUPIED {false_empty} ({false_empty/max(true_empty+false_empty,1):.3%}), empty but believed occupied {miss_empty}')
print(f'  sum over steps of (wrongly released occupants / present occupants) = {false_empty_wt:.2f}')
