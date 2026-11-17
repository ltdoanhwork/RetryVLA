import sys, numpy as np, pandas as pd
from scipy.stats import binomtest
sys.path.insert(0, '/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts')
from analyze_acrn import load, merged_causes
from retryvla import REPO_ROOT
recs = load(REPO_ROOT / 'data/acr/n1'); causes = np.array(merged_causes(recs)); opts = recs[0]['options']
best = []
for r in recs:
    v = np.array([np.mean(r['O'][o] == 1) for o in range(len(opts))]); best.append(opts[int(np.argmax(v))] if v.max() > 0 else 'none')
best = np.array(best)
def mi(a, b):
    ct = pd.crosstab(a, b).values; p = ct / ct.sum(); pc, pb = p.sum(1, keepdims=True), p.sum(0, keepdims=True); nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pc @ pb)[nz])).sum())
rng = np.random.default_rng(0)
obs = mi(causes, best); null = [mi(rng.permutation(causes), best) for _ in range(2000)]
print('MI obs', round(obs, 3), 'perm null mean', round(np.mean(null), 3), '95%', round(np.quantile(null, .95), 3), 'p', np.mean(np.array(null) >= obs))
m = best != 'none'
obs2 = mi(causes[m], best[m]); null2 = [mi(rng.permutation(causes[m]), best[m]) for _ in range(2000)]
print('MI excl none (n', m.sum(), ')', round(obs2, 3), 'null mean', round(np.mean(null2), 3), 'p', np.mean(np.array(null2) >= obs2))
# power of exact McNemar, n=73, alpha .10 two-sided, for effect d with discordant fraction q
def power(n, d, q, sims=4000):
    p01 = (q + d) / 2; p10 = (q - d) / 2; hit = 0
    for _ in range(sims):
        x = rng.multinomial(n, [p01, p10, 1 - p01 - p10]); a, b = x[0], x[1]
        if a + b and a > b and binomtest(a, a + b, .5).pvalue < .10: hit += 1
    return hit / sims
for q in (0.10, 0.15, 0.25):
    for d in (0.05, 0.08, 0.10, 0.15):
        if d <= q: print(f'n73 q{q} d{d}: power {power(73, d, q):.2f}')
for n in (150, 250):
    print(f'n{n} q0.15 d0.08: power {power(n, .08, .15):.2f}; d0.05 {power(n,.05,.15):.2f}')
