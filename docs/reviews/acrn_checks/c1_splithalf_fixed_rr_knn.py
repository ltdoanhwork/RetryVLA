import sys, numpy as np, pandas as pd
sys.path.insert(0, '/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts')
from analyze_acrn import load, merged_causes, folds, Model, execute, mcnemar, obs_x
from retryvla import REPO_ROOT
recs = load(REPO_ROOT / 'data/acr/n1'); causes = merged_causes(recs)
opts = recs[0]['options']; F = len(opts)
U = [i for i, r in enumerate(recs) if r['set'] == 'U']
print('n', len(recs), 'U', len(U))
O = np.stack([r['O'] for r in recs])  # n,F,8
print('missing (-1) cells', (O < 0).sum())
succ = (O == 1)
tot = succ.reshape(len(recs), -1).sum(1)
print('U total successes per instance (of 80) distribution:', np.bincount(tot[U]))
print('U per-option per-trial rate:', dict(zip(opts, np.round(succ[U].mean((0, 2)), 3))))
# within-option ICC proxy: P(success variant j | success variant i of same option) vs marginal
for name, idx in (('U', U), ('all', list(range(len(recs))))):
    s = succ[idx]
    pairs_same = []; pairs_diff = []
    for x in s:
        for o in range(F):
            v = x[o]; k = v.sum()
            if len(v) > 1: pairs_same.append((k * (k - 1)) / (len(v) * (len(v) - 1)))
    print(name, 'P(two variants of same option both succeed)', np.mean(pairs_same), 'vs p^2 marginal', s.mean() ** 2)
# split-half: choose by variants 0..3, evaluate on 4..7 (P>=1 success in 4 trials of chosen option)
fo = folds(recs)
res = {k: [] for k in ['split_inst', 'fixed_marg', 'random_opt', 'hind_4_7', 'split_cause', 'roundrobin4']}
for i in U:
    tr = [j for j in range(len(recs)) if fo[j] != fo[i]]
    marg = succ[tr][:, :, :4].mean((0, 2))
    a = succ[i][:, :4].sum(1) + 1e-3 * marg  # instance's own first half, tie -> marginal
    ob = int(np.argmax(a)); om = int(np.argmax(marg))
    trc = [j for j in tr if causes[j] == causes[i]]
    mc = succ[trc][:, :, :4].mean((0, 2)) if trc else marg
    oc = int(np.argmax(mc + 1e-3 * marg))
    res['split_inst'].append(succ[i][ob, 4:].any())
    res['fixed_marg'].append(succ[i][om, 4:].any())
    res['split_cause'].append(succ[i][oc, 4:].any())
    res['random_opt'].append(succ[i][:, 4:].any(1).mean())
    res['hind_4_7'].append(succ[i][:, 4:].any(1).max())
    # round robin: 4 trials across top-4 marginal options, 1 each (variants 4..7)
    top4 = np.argsort(-marg)[:4]
    res['roundrobin4'].append(any(succ[i][o, 4 + t] for t, o in enumerate(top4)))
print('SPLIT-HALF on U (4 held-out trials):', {k: round(float(np.mean(v)), 3) for k, v in res.items()})
a = np.array(res['split_inst']); b = np.array(res['fixed_marg']); print(' split_inst vs fixed_marg McNemar', mcnemar(b, a))
a2 = np.array(res['split_cause']); print(' split_cause vs fixed_marg McNemar', mcnemar(b, a2))
# fixed-option arms and round robin at S=2800 using the executor (worst-case costs)
def worst(o): return 350 if opts[o].startswith('wm_') else (300 if opts[o] == 'vla' else 440)
W = np.array([worst(o) for o in range(F)], float)
for budget in (2800, 1400):
    out = {}
    for o in range(F):
        out[opts[o]] = np.mean([execute(recs[i], [o] * 8, budget, W)['solved'] for i in U])
    rr = []
    rrm = []
    for i in U:
        tr = [j for j in range(len(recs)) if fo[j] != fo[i]]
        marg = succ[tr].mean((0, 2)); order = list(np.argsort(-marg))
        seq = [o for k in range(8) for o in order]
        rr.append(execute(recs[i], seq, budget, W)['solved'])
        # round robin over top-3 options only
        seq3 = [o for k in range(8) for o in order[:3]]
        rrm.append(execute(recs[i], seq3, budget, W)['solved'])
    print(f'budget {budget} fixed single-option on U:', {k: round(v, 3) for k, v in out.items()})
    print(f'budget {budget} round-robin all options by marginal order: {np.mean(rr):.3f}; top-3 RR: {np.mean(rrm):.3f}')
    rows = pd.read_json(REPO_ROOT / f'runs/_analysis/acrn/rows_b{budget}.jsonl', lines=True)
    ru = rows[rows.set == 'U'].set_index('instance_id')
    ids = [recs[i]['instance_id'] for i in U]
    npb = ru.loc[ids, 'no_probe'].values.astype(bool)
    print('  RR vs no_probe McNemar (n01=RR wins)', mcnemar(npb, np.array(rr, bool)))
    # S set check for RR
    S = [i for i, r in enumerate(recs) if r['set'] == 'S']
    rrS = []
    for i in S:
        tr = [j for j in range(len(recs)) if fo[j] != fo[i]]
        marg = succ[tr].mean((0, 2)); order = list(np.argsort(-marg))
        rrS.append(execute(recs[i], [o for k in range(8) for o in order], budget, W)['solved'])
    print(f'  RR on S: {np.mean(rrS):.3f}; wm_only on S', round(np.mean([execute(recs[i], [1]*8, budget, W)['solved'] for i in S]),3))
# kNN instance selector (cross-fitted): neighbours in obs feature space, per-option rate -> greedy with conditional decay
X = np.stack([obs_x(r) for r in recs]); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
for budget in (2800, 1400):
  for K in (10, 20):
    sol = []
    for i in U:
        tr = np.array([j for j in range(len(recs)) if fo[j] != fo[i]])
        d = ((X[tr] - X[i]) ** 2).sum(1); nb = tr[np.argsort(d)[:K]]
        p = (succ[nb].mean((0, 2)) * K + 2 * succ[tr].mean((0, 2))) / (K + 2)
        # sequence: by p/cost, decay 0.6 after each failure of same option
        n = np.zeros(F); seq = []
        for _ in range(40):
            o = int(np.argmax(p * 0.6 ** n / W)); seq.append(o); n[o] += 1
        sol.append(execute(recs[i], seq, budget, W)['solved'])
    print(f'budget {budget} kNN selector K={K}: {np.mean(sol):.3f}')
