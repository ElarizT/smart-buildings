"""Print a compact view of an sbrun trace."""
import sys
import numpy as np
f = sys.argv[1]
a, b = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (0, 10**9)
every = int(sys.argv[4]) if len(sys.argv) > 4 else 6
d = np.load(f)
n = len(d['r'])
print('steps', n, 'sum r', d['r'].sum(), 'comfort', -0.6*d['comfort'].sum(), 'cost', 0.2*d['cost'].sum(), 'carbon', 0.2*d['carbon'].sum())
occm = d['occ'] > 0
lo, hi = d['hs'], d['cs']
zt = d['zt']
below = np.maximum(lo[:, None] - zt, 0); above = np.maximum(zt - hi[:, None], 0)
print('occupied steps', occm.sum(), 'mean below (occ)', below[occm].mean(), 'mean above (occ)', above[occm].mean(),
      'max below', below[occm].max(), 'max above', above[occm].max())
if 'dbg_pred_err' in d:
    pe = d['dbg_pred_err']
    act = d['dbg_active'] > 0
    print('pred err: all rms', np.sqrt((pe**2).mean()), 'active rms', np.sqrt((pe[act]**2).mean()), 'active bias', pe[act].mean())
print('ts           occ   oat  ztmin ztmean ztmax  nb  na  comfort  fan  cool   gas  wt  | on  P        r      tset | perr_rms')
for i in range(max(a, 0), min(b, n), every):
    z = zt[i] - 273.15
    s = '%s %4.0f %5.1f %5.1f %5.1f %5.1f %3d %3d %7.3f %5.1f %5.1f %6.1f %5.1f' % (
        d['ts'][i][5:16], d['occ'][i], d['oat'][i] - 273.15, z.min(), z.mean(), z.max(),
        (zt[i] < lo[i]).sum(), (zt[i] > hi[i]).sum(), d['comfort'][i], d['fan'][i] / 1e3, d['cool'][i] / 1e3, d['gas'][i] / 1e3, d['wt'][i])
    if 'dbg_ahu_on' in d:
        s += ' | %s %s %s %s' % (''.join(str(int(x)) for x in d['dbg_ahu_on'][i]), np.round(d['dbg_ahu_p'][i]).astype(int), np.round(d['dbg_ahu_r'][i], 1), np.round(d['dbg_ahu_set'][i]).astype(int))
    if 'dbg_pred_err' in d:
        s += ' | %.3f' % np.sqrt((d['dbg_pred_err'][i]**2).mean())
    print(s)
