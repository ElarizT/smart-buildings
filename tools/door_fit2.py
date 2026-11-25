"""Fit inter-zone coupling (doors and shared walls) from disturbance innovations, correct alignment."""
import sys, numpy as np
sys.path.insert(0, '/root/starter-kit')
from pathlib import Path
from scipy import ndimage
from absl import logging; logging.set_verbosity(logging.ERROR)
from smart_buildings import evaluate as ev
d = np.load(sys.argv[1]); scen = Path(sys.argv[2])
env = ev.load_hybrid_environment(scen / 'config.gin'); md = ev.create_building_meta_data(env, scen)
fp = np.asarray(md['floor_plan']); zm = np.asarray(md['zone_map'])
lab, n = ndimage.label(zm == 0)
def neigh(mask):
    out = {}
    for r, c in np.argwhere(mask):
        adj = set()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < lab.shape[0] and 0 <= cc < lab.shape[1] and lab[rr, cc] > 0: adj.add(int(lab[rr, cc]))
        adj = sorted(adj)
        for i in range(len(adj)):
            for j in range(i + 1, len(adj)):
                out[(adj[i], adj[j])] = out.get((adj[i], adj[j]), 0) + 1
    return out
doors = neigh((fp == 0) & (zm != 0)); walls = neigh((fp == 1) & (zm == 1))
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zobs = d['zobs'][:, perm]; dwm = d['dbg_dwm']; zocc = d['zocc'][:, perm]; ts = d['ts']; rel = d['dbg_released'] > 0
N = len(zid); idx = {int(z.split('_')[1]): i for i, z in enumerate(zid)}
D = np.zeros((N, N)); Wl = np.zeros((N, N))
for (a, b), c in doors.items(): D[idx[a], idx[b]] += c; D[idx[b], idx[a]] += c
for (a, b), c in walls.items(): Wl[idx[a], idx[b]] += c; Wl[idx[b], idx[a]] += c
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
X = []; Y = []
for i in range(5, len(ts)):
    if not (13.4 <= hours[i] < 18.8): continue
    dy = zobs[i - 1] - zobs[i - 2]
    if not all(np.all(zocc[i - k] == zocc[i - k - 1]) for k in range(0, 4)): continue
    x = dwm[i] - np.median(dwm[i])
    keep = ~rel[i] & ~rel[i - 1] & ~rel[i - 2] & ~rel[i - 3]
    X.append(np.stack([D @ dy, D.sum(1) * dy, Wl @ dy, Wl.sum(1) * dy], 1)[keep]); Y.append(x[keep])
X = np.concatenate(X); Y = np.concatenate(Y)
m = (np.abs(X[:, 0]) > 0.3) | (np.abs(X[:, 2]) > 10)
sol, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
print('samples', m.sum(), 'coef: door-neigh %.1f door-own %.1f wall-neigh %.2f wall-own %.2f' % tuple(sol), ' rms before %.1f after %.1f' % (np.sqrt((Y[m] ** 2).mean()), np.sqrt(((Y[m] - X[m] @ sol) ** 2).mean())))
sol2, *_ = np.linalg.lstsq(X[m][:, [0, 2]], Y[m], rcond=None)
print('  neighbour-only fit: door %.1f W/K per door cell, wall %.2f W/K per wall cell; rms after %.1f' % (sol2[0], sol2[1], np.sqrt(((Y[m] - X[m][:, [0, 2]] @ sol2) ** 2).mean())))
