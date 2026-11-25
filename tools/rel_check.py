import sys, numpy as np
d=np.load(sys.argv[1])
vid=list(d['vav_ids']); zid=list(d['zone_ids']); perm=[vid.index(z) for z in zid]
zocc=d['zocc'][:,perm]; rel=d['dbg_released']>0; ts=d['ts']; zt=d['zt'][:,perm]
hours=np.array([int(t[11:13])+int(t[14:16])/60 for t in ts])
# dbg at row i = decision made for step producing row i+1; released flags apply to that action
n=len(ts)
occm=(d['occ']>0)
aft=(hours>=13)&(hours<19)&occm
te=zocc<=0
print('afternoon zone-steps: released & empty', int((rel&te)[aft].sum()), ' released & OCCUPIED', int((rel&~te)[aft].sum()), ' empty total', int(te[aft].sum()), ' capture %.2f'%((rel&te)[aft].sum()/max(te[aft].sum(),1)))
wrong=np.argwhere((rel&~te)&aft[:,None])
for i,z in wrong[:10]: print('  wrong', ts[i], zid[z], 'zone occ', zocc[i,z], 'tot', d['occ'][i])
lo=d['hs']; hi=d['cs']
dev=np.maximum(lo[:,None]-zt,0)+np.maximum(zt-hi[:,None],0)
print('comfort -0.6*sum', -0.6*d['comfort'].sum(), 'max dev of occupied zones', (dev*(zocc>0)).max(), 'mean dev released zones', dev[rel&aft[:,None]].mean() if (rel&aft[:,None]).any() else 0)
