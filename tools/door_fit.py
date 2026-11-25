"""Find door cells (air in floor_plan, wall in zone_map) and fit the inter-zone coupling per door cell."""
import sys, numpy as np
sys.path.insert(0, '/home/user/smart-buildings/policy'); sys.path.insert(0, '/root/starter-kit')
from pathlib import Path
from scipy import ndimage
from absl import logging; logging.set_verbosity(logging.ERROR)
from smart_buildings import evaluate as ev
d = np.load(sys.argv[1]); scen = Path(sys.argv[2])
env = ev.load_hybrid_environment(scen / 'config.gin'); md = ev.create_building_meta_data(env, scen)
fp = np.asarray(md['floor_plan']); zm = np.asarray(md['zone_map'])
lab, n = ndimage.label(zm == 0)
door = (fp == 0) & (zm != 0)
print('door cells', int(door.sum()), 'zone_map values at door', np.unique(zm[door]), 'rooms', n)
dl, nd = ndimage.label(door)
pairs = {}
for k in range(1, nd + 1):
    cells = np.argwhere(dl == k)
    adj = set()
    for r, c in cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < lab.shape[0] and 0 <= cc < lab.shape[1] and lab[rr, cc] > 0:
                adj.add(int(lab[rr, cc]))
    adj = sorted(adj)
    if len(adj) == 2:
        pairs[tuple(adj)] = pairs.get(tuple(adj), 0) + len(cells)
    elif len(adj) != 1:
        print('door group', k, 'size', len(cells), 'adjacent rooms', adj)
print('door groups', nd, 'zone pairs linked', len(pairs), 'cells per pair', sorted(set(pairs.values())))
vid = list(d['vav_ids']); zid = list(d['zone_ids']); perm = [vid.index(z) for z in zid]
zobs = d['zobs'][:, perm]; dwm = d['dbg_dwm']; zocc = d['zocc'][:, perm]; ts = d['ts']
N = len(zid); idx = {int(z.split('_')[1]): i for i, z in enumerate(zid)}
G = np.zeros((N, N))
for (a, b), cnt in pairs.items():
    if a in idx and b in idx:
        G[idx[a], idx[b]] += cnt; G[idx[b], idx[a]] += cnt
hours = np.array([int(t[11:13]) + int(t[14:16]) / 60 for t in ts])
# rows: dbg at row i computed at the call that saw y = zobs[i] (sensor after step). dwm[i] ~ G*(y[i]-y[i-1]) relative
X = []; Y = []
for i in range(3, len(ts)):
    if not (13.4 <= hours[i] < 18.8): continue
    dy = zobs[i] - zobs[i - 1]
    f = G @ dy - G.sum(1) * 0          # sum over neighbours of doorcells * dT_neighbour
    own = G.sum(1) * dy                # own temperature change times its door count
    x = dwm[i] - np.median(dwm[i])
    quiet = np.all(zocc[i] == zocc[i - 1]) and np.all(zocc[i - 1] == zocc[i - 2]) and np.all(zocc[i - 2] == zocc[i - 3]) and np.all(zocc[i-3]==zocc[i-4])
    if not quiet: continue
    X.append(np.stack([f, own], 1)); Y.append(x)
X = np.concatenate(X); Y = np.concatenate(Y)
m = np.abs(X[:, 0]) > 0.05
sol, *_ = np.linalg.lstsq(X[m], Y[m], rcond=None)
pred = X[m] @ sol
print('fit W per door cell per K: neighbour %.1f, own %.1f ; n=%d ; rms before %.1f after %.1f' % (sol[0], sol[1], m.sum(), np.sqrt((Y[m]**2).mean()), np.sqrt(((Y[m]-pred)**2).mean())))
