import sys, numpy as np
sys.path.insert(0, '/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/scripts')
from analyze_acrn import load
from retryvla import REPO_ROOT
recs = load(REPO_ROOT / 'data/acr/n1'); opts = recs[0]['options']
for name, sel in (('U', 'U'), ('S', 'S')):
    R = [r for r in recs if r['set'] == sel]
    s = np.stack([r['O'] == 1 for r in R]); st = np.stack([r['O_steps'] for r in R])
    print(name, {o: f"{1000*s[:,k].sum()/st[:,k].sum():.2f}" for k, o in enumerate(opts)})
