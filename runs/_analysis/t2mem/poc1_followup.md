# T2Mem PoC 1 follow-up F1: memory on top of a frozen recalibrator (2-stage alternation)

| method | hits r1 | hits r2-4 | prec / verif r2-4 | solved@16 | hits r1-4 | prec per round |
|---|---|---|---|---|---|---|
| static | 0.781 | 1.859 | 0.155 | 0.551 | 2.641 | 0.195 / 0.176 / 0.149 / 0.140 |
| oracle_true | 0.781 | 2.387 | 0.199 | 0.578 | 3.168 | 0.195 / 0.243 / 0.192 / 0.161 |
| recal_r1 | 0.977 | 2.163 | 0.180 | 0.574 | 3.139 | 0.244 / 0.204 / 0.180 / 0.157 |
| mem_on_recal | 0.977 | 2.159 | 0.180 | 0.574 | 3.135 | 0.244 / 0.208 / 0.182 / 0.150 |
| mem_on_recal_ss | 0.977 | 2.126 | 0.177 | 0.574 | 3.103 | 0.244 / 0.203 / 0.179 / 0.149 |

| comparison | d hits (rounds) | 95% CI | sign-flip p |
|---|---|---|---|
| mem_on_recal - recal_r1 | -0.004 (r2-4) | [-0.070, +0.061] | 0.938 |
| mem_on_recal_ss - recal_r1 | -0.036 (r2-4) | [-0.100, +0.021] | 0.276 |
| recal_r1 - static | +0.499 (r1-4) | [+0.315, +0.694] | 0 |
| oracle_true - static | +0.527 (r1-4) | [+0.344, +0.719] | 0 |

**F1 rule: memory adds value on top of the recalibrator -> NO**
