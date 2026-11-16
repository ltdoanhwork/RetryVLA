# T2Mem PoC 1: instance-level test-time memory in recovery search (offline replay, verif_m5b train)

256 instances x 64 verified candidates, R = 4 rounds x K = 4 picks (16 verifications). Round 1 identical for all methods. Hits r2-4 = successes among the 12 picks made after >= 1 memory write.

| method | hits r1 | hits r2-4 | prec / verif r2-4 | solved@16 | prec per round | d hits r2-4 vs static [95% CI] | sign-flip p |
|---|---|---|---|---|---|---|---|
| static | 0.781 | 1.859 | 0.155 | 0.551 | 0.195 / 0.176 / 0.149 / 0.140 | +0.000 [+0.000, +0.000] | 1 |
| oracle_true | 0.781 | 2.387 | 0.199 | 0.578 | 0.195 / 0.243 / 0.192 / 0.161 | +0.527 [+0.344, +0.719] | 0 |
| knn_label | 0.781 | 1.934 | 0.161 | 0.547 | 0.195 / 0.154 / 0.165 / 0.164 | +0.074 [-0.027, +0.184] | 0.196 |
| knn_latent | 0.781 | 1.898 | 0.158 | 0.551 | 0.195 / 0.150 / 0.160 / 0.164 | +0.039 [-0.066, +0.145] | 0.519 |
| ttt_plain | 0.781 | 1.824 | 0.152 | 0.539 | 0.195 / 0.150 / 0.152 / 0.153 | -0.035 [-0.223, +0.148] | 0.742 |
| t2m_ss | 0.781 | 2.096 | 0.175 | 0.574 | 0.195 / 0.212 / 0.168 / 0.144 | +0.237 [+0.124, +0.358] | 5e-05 |
| t2m_ss_static | 0.781 | 1.859 | 0.155 | 0.551 | 0.195 / 0.175 / 0.146 / 0.144 | +0.000 [-0.012, +0.012] | 1 |
| t2m_label | 0.781 | 2.234 | 0.186 | 0.574 | 0.195 / 0.234 / 0.175 / 0.149 | +0.375 [+0.220, +0.538] | 0 |
| t2m_label_static | 0.781 | 2.327 | 0.194 | 0.578 | 0.195 / 0.237 / 0.192 / 0.153 | +0.467 [+0.296, +0.650] | 0 |

Pre-registered decision (primary = t2m_label; per-verification differences on rounds 2-4):

1. vs static: +0.0312 / verif, p 0 (need >= +0.03 and p < 0.10) -> PASS
2. vs best explicit baseline (knn_label): +0.0251 / verif, p 0.0001 (need > 0) -> PASS
3. vs its static control: -0.0077 / verif, p 0.0469 (need > 0 and p < 0.10) -> FAIL

**Decision: NO online PoC (negative / redesign)**

Secondary (not part of the decision): t2m_ss vs static +0.0197 / verif, p 5e-05; t2m_ss vs t2m_ss_static +0.0197, p 5e-05; knn_label vs static +0.0062, p 0.196; oracle_true vs static +0.0439, p 0.

Chosen hyper-parameters per fold: {"knn_label": [{"lam": 8, "bw": 0.1, "tau": 0.1}, {"lam": 8, "bw": 0.1, "tau": 0.1}, {"lam": 8, "bw": 0.1, "tau": 0.1}, {"lam": 2, "bw": 0.03, "tau": 0.5}], "knn_latent": [{"bw": 0.03, "tau": 2.0}, {"bw": 0.1, "tau": 0.1}, {"bw": 0.1, "tau": 0.03}, {"bw": 0.03, "tau": 2.0}], "ttt_plain": [{"eta": 0.01, "steps": 20}, {"eta": 0.01, "steps": 20}, {"eta": 0.003, "steps": 50}, {"eta": 0.003, "steps": 50}]}
