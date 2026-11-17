import sys, numpy as np, pandas as pd, collections
sys.path.insert(0, '/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts')
from analyze_acrn import load, merged_causes, folds, execute, mcnemar, obs_x
from retryvla import REPO_ROOT
recs = load(REPO_ROOT / 'data/acr/n1'); causes = merged_causes(recs)
opts = recs[0]['options']; F = len(opts); fo = folds(recs)
succ = np.stack([r['O'] == 1 for r in recs])
W = np.array([350 if o.startswith('wm_') else (300 if o == 'vla' else 440) for o in opts], float)
U = [i for i, r in enumerate(recs) if r['set'] == 'U']; S = [i for i, r in enumerate(recs) if r['set'] == 'S']
for budget in (2800, 1400):
    rows = pd.read_json(REPO_ROOT / f'runs/_analysis/acrn/rows_b{budget}.jsonl', lines=True).set_index('instance_id')
    # composition of no_probe trial sequences
    comp = collections.Counter(); ntr = []
    for i in U:
        t = rows.loc[recs[i]['instance_id'], 'no_probe__trials']; ntr.append(len(t))
        comp.update(opts[o] for o, _ in t)
    print(budget, 'no_probe trials/inst', np.mean(ntr), 'composition', comp.most_common())
    comp = collections.Counter()
    for i in U:
        comp.update(opts[o] for o, _ in rows.loc[recs[i]['instance_id'], 'oracle_cause__trials'])
    print(budget, 'oracle_cause composition', comp.most_common())
    # cross-fitted best fixed option (chosen on training folds by budget-level solved rate, U+S train) and best-fixed restricted to U-train
    for pool in ('all', 'Utrain'):
        sol = {'U': [], 'S': []}; picks = collections.Counter()
        for i in range(len(recs)):
            tr = [j for j in range(len(recs)) if fo[j] != fo[i] and (pool == 'all' or recs[j]['set'] == 'U')]
            sc = [np.mean([execute(recs[j], [o] * 8, budget, W)['solved'] for j in tr]) for o in range(F)]
            o = int(np.argmax(sc)); picks[opts[o]] += 1
            sol[recs[i]['set']].append(execute(recs[i], [o] * 8, budget, W)['solved'])
        npu = rows.loc[[recs[i]['instance_id'] for i in U], 'no_probe'].values.astype(bool)
        print(f'  CF best-fixed ({pool}): U {np.mean(sol["U"]):.3f} S {np.mean(sol["S"]):.3f} picks {dict(picks)} McNemar vs no_probe {mcnemar(npu, np.array(sol["U"], bool))}')
    # hover+grasp_slow alternating
    for combo in (('hover', 'grasp_slow'), ('grasp_slow', 'hover', 'grasp_firm'), ('grasp_slow','hover','wm_single')):
        seq = [opts.index(c) for k in range(8) for c in combo]
        su = [execute(recs[i], seq, budget, W)['solved'] for i in U]; ss = [execute(recs[i], seq, budget, W)['solved'] for i in S]
        print(f'  alt {combo}: U {np.mean(su):.3f} S {np.mean(ss):.3f}')
    # per-cause oracle within hindsight: if we knew cause and picked best fixed option per cause in hindsight (in-sample) 
    cs = sorted(set(causes)); tot = 0
    for c in cs:
        idx = [i for i in U if causes[i] == c]
        if not idx: continue
        best = max(range(F), key=lambda o: np.mean([execute(recs[i], [o]*8, budget, W)['solved'] for i in idx]))
        v = sum(execute(recs[i], [best]*8, budget, W)['solved'] for i in idx); tot += v
        print(f'    in-sample best fixed per cause {c} n{len(idx)}: {opts[best]} {v/len(idx):.2f}')
    print(f'  IN-SAMPLE cause->best fixed option (optimistic oracle-cause ceiling): U {tot/len(U):.3f}')
    best = max(range(F), key=lambda o: np.mean([execute(recs[i], [o]*8, budget, W)['solved'] for i in U]))
    print(f'  IN-SAMPLE best single fixed option, no cause: {opts[best]}')
# trials per option at 2800
print('trials affordable per option at 2800:', {o: int(2800 // W[k]) for k, o in enumerate(opts)})
# per-instance: how many options have >=1 success in variants (U)
k = succ[U].any(2).sum(1); print('U #options with any success:', np.bincount(k))
# probe 'success' rate
print('probe success any:', np.mean([any(p['success'] for p in r['probes'].values()) for r in recs]))
# cause map acc of b0 on U = 0.51 ; majority class rate
print('U cause share contact_free', np.mean([causes[i]=='contact_free' for i in U]))
