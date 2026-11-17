# RetryVLA: kế hoạch giai đoạn tiếp theo (Phase 1 data engine, Phase 2 distillation), 2026-11-17

> Trạng thái: **PLAN, chưa chạy thí nghiệm nào.** Viết 2026-11-17 theo yêu cầu của user (paraphrase): "Phase 1 (làm giàu
> dữ liệu) phải đủ tổng quát: cho BẤT KỲ world model nào, train nó / search để tìm recovery tốt hơn; nếu dữ liệu không phải
> LIBERO-Recover mà là một dataset kiểu LIBERO khác thì xây (failure + recovery) thế nào; Phase 2 (distillation) áp dụng
> recipe train của foundation model." Bổ sung của user 2026-11-17 10:04 UTC: **student của Phase 2 không chỉ là VLA**:
> (1) VLA (SmolVLA), (2) chính WORLD MODEL, (3) một DIFFUSION POLICY (claim policy-agnostic, so trực tiếp với SAILOR trên
> sân của họ).
> Quy tắc: mọi idea trong file này đi qua skill `retryvla-idea-loop` (intake có decision rule trong `experiments/PLAN.md`
> TRƯỚC khi có kết quả, oracle-first, paired, equal budget, train/pilot only, 3 seeds, nominal <= 2 pt, L1-L4, review đối
> kháng trước khi đóng / scale). File này là bản đồ; mỗi milestone khi bắt đầu sẽ có intake riêng trong PLAN.md.
> ACR-N follow-ups F1 / F1b / F2 đang chạy bởi agent khác trên 8b889199 GPUs 1-3: **không đụng**; plan này chỉ dùng kết
> quả của chúng làm gate (M-P1.4).

## 0. TL;DR (các quyết định chính)

1. **Bottleneck đã đo, plan đánh thẳng vào nó.** Với pick do search chọn, C^pi trên end state tưởng tượng 0.73 vs C^pi
   trên end state thật 0.27 vs sim SR 0.17; 80-85 % của độ lạc quan nằm ở WM (image tokens), phần còn lại ở critic; gap
   tăng theo ngân sách search (FP 0.72 -> 0.85 từ N = 8 -> 512). Đổi WM (V-JEPA 2-AC), xwm agreement, geometry (RGR),
   probing (ACR-N) đều không nâng solved rate có ý nghĩa. Vì vậy Phase 1 không thêm "selection rule" mới trên cùng
   imagination, mà **sửa imagination bằng chính dữ liệu đã verify** (verified negatives) và **giảm số lần phải tin
   imagination** (multi-fidelity verification, option library, learned proposal).
2. **Phase 1 = WM-agnostic recovery data engine** với một interface tối thiểu (`encode`, `rollout`, optional
   `uncertainty`, `finetune`) và 3 adapter (WM v1 DINOv2, V-JEPA 2-AC, Cosmos-Predict2.5-2B robot/action-cond) để chứng
   minh "bất kỳ WM nào". WM-agnosticism được chứng minh bằng **cùng một recipe sửa optimism áp dụng cho >= 2 WM** cho
   cùng chiều hiệu ứng, không phải bằng việc chạy được 3 WM.
3. **Ba student cho cùng một ngân hàng dữ liệu verified (Phase 2)**: (S1) SmolVLA, (S2) chính WM + critic ("WM
   self-improvement from verified negatives", nối Phase 1 và Phase 2 thành một vòng), (S3) Diffusion Policy (DP) trên
   robomimic / robosuite = sân của SAILOR, để claim "policy-agnostic recovery data engine" và so trực tiếp.
4. **Phần B (dữ liệu ngoài LIBERO-Recover)**: đính chính quan trọng: LIBERO-Recover (arXiv 2609.05178, đã đọc HTML v3)
   **không inject perturbation**: failure là failure tự nhiên của 6 policy (pi0, pi0.5, OpenVLA, GR00T, Wan-Policy,
   Cosmos-Policy) chạy từ init gốc; Qwen3.5-27B định vị đoạn failure (V_pre / V_fail / V_post), lấy sim state tại biên,
   rồi VLM gán level L1-L4; Stage 4 fine-tune bằng teleop recovery rồi chạy lại để thu thêm failure. Perturbation (±2 cm xy,
   gripper đóng / mở) chỉ có trong protocol eval. Do đó pipeline của mình: (1) **harvest failure tự nhiên** (trung thành
   với LIBERO-Recover) là nguồn chính; (2) **perturbation injection** là generator bổ sung (để phủ L4 / hiếm), luôn gắn
   nhãn "injected" và báo cáo riêng.
5. **Phase 2 recipe**: giữ SFT + 50 % nominal replay làm control; shortlist (a) RECAP-style advantage conditioning
   (pi*0.6) dùng C^pi làm value và verified negatives, (b) preference optimisation kiểu Diffusion-DPO chuyển sang flow
   matching trên các cặp (success, fail) **cùng failure state** (dữ liệu paired hoàn hảo mà ta đã có), (c) expert
   iteration nhiều vòng với critic refit theo pi_k, (d) noise-space RL (DSRL) từ failure snapshot làm tham chiếu RL rẻ,
   (e) WiSE-FT weight interpolation làm nominal guard. KI (pi0.5) gần như **đã có sẵn** ở dạng tầm thường: SFT của ta chỉ
   train action expert (97.5M params trainable, VLM + vision frozen); chỉ cần KI đầy đủ nếu mở VLM.
6. **Baseline bắt buộc** (từ overlap audit): Back-on-Track-style test-time planning arm, SAILOR-style (discriminator
   reward + ensemble mean-std + distill, không verify), no-WM sim-only search cùng sim budget (= 2610.05151-style
   acceptance), Dream2Fix / WM-DAgger-style unverified WM data, teleop "Combined" của LIBERO-Recover, scripted regrasp
   (ACR-N).
7. **Held-out**: Object test (frozen) + một suite mới dựng bằng Phần B: **LIBERO-Goal** (train trên failure do ta tự
   dựng, test trên 91 scene Goal CHÍNH THỨC của LIBERO-Recover đã có local -> kiểm chứng generator bằng scene không do
   ta làm), stretch: LIBERO-10.

## 1. Bằng chứng đầu vào (từ `experiments/RESULTS.md`, `docs/reviews/ACRN_REVIEW.md`)

| Kết quả | Số | Hệ quả cho plan |
|---|---|---|
| 6-arm ablation, 3 seeds | pi1 (verified, action-grounded) 0.418 ± 0.021, nominal 0.785 (ref pi1-nominal 0.796); action-free 0.12-0.13; hybrid 0.370; Backtrack-distill 0.166; demo-only 0.136 | chỉ dữ liệu có action và đã verify có tác dụng; Phase 2 phải giữ verify |
| RQ3 optimism | imagined C 0.727 / true-state C 0.265 / SR 0.171 (wm_diverse); FP(a) 0.79 vs FP(b) 0.61; gap WM tăng -0.01 -> +0.33 khi N 8 -> 512 | bottleneck = WM imagination dưới search pressure |
| V-JEPA 2-AC drop-in | solved 0.413 vs 0.349 (p 0.45); FP ~0.83 cả hai | WM mạnh hơn không tự hết bị exploit |
| xwm min-agreement | hits / sim 0.171 vs 0.146, FP 0.70 vs 0.82, solved p 0.29 (n 60) | lỗi tương quan giữa WM; pessimism chỉ gỡ một phần |
| RGR | geometry: precision / verif +36 %, solved không tăng (0.386 vs 0.425) | không phải geometry |
| ACR-N | oracle cause không giúp; fixed `grasp_slow` U 0.274 / S 1.000 vs no_probe 0.219; hindsight 0.52 chủ yếu là noise | option library scripted là đòn bẩy trên failure khó; selection bị giới hạn bởi noise |
| T2Mem lead | linear recalibrator trên CLS của imagined end state: hits r1-4 3.14 vs 2.64 (+19 %), ~ = ranking theo TRUE end state 3.17; prequential: inconclusive | imagination **đọc lại được** bằng nhãn verified: lead mạnh nhất cho S2 |
| M11 | WM planner từ noise 0.398 < library 0.458 | proposal từ library / học được, không plan từ noise |
| M6 round 2 | +0 so với chỉ train thêm (critic cũ) | vòng lặp chỉ có nghĩa nếu critic / WM refit theo pi_k |
| Object (held-out) | comp 0.354 vs random 0.362 (n.s.) | lợi thế WM là acquisition efficiency, chưa thành policy gain trên Object |

Đơn vị chi phí (NOTES): SFT 6k steps ~1.3 GPU-h A100; RSR Spatial test ~1 h, Object ~0.5 h, nominal ~0.4 h; search
~4 min / instance / 6 arms với 16 sim workers. Máy: local shell 8b889199, GPUs 1-3 (A100 80 GB), 32 CPU, 2 TB RAM;
MuJoCo là CPU-bound -> CPU, không phải GPU, giới hạn search / RL.

## 2. Phase 1 (A): WM-agnostic recovery data engine

### 2.1 Interface tối thiểu mà mọi WM phải có (`retryvla/wm/api.py`, mới; code cũ giữ nguyên, chỉ bọc lại)

```python
class WorldModel(Protocol):
    meta: WMMeta  # env_steps_per_model_step, horizon_max, action_space ("libero_osc_delta7"), latent_kind, cost_ms_per_rollout
    def encode(self, obs: dict) -> State: ...                         # real obs -> state latent (gồm proprio)
    def rollout(self, s0: State, actions: Tensor[N, H, 7]) -> Traj: ... # imagined states s_1..s_H (ít nhất s_H)
    def uncertainty(self, s0, actions) -> Tensor[N] | None: ...       # optional: ensemble / sampling / dropout
    def finetune(self, transitions: TransitionSet, cfg) -> "WorldModel": ...  # optional: cho S2 (student WM)
class Critic(Protocol):                                               # C^pi trên latent của CHÍNH WM đó
    def __call__(self, s: State, task: str) -> Tensor[N]: ...         # P(pi finishes | s); trained on REAL continuations
class Verifier(Protocol):                                             # sim (bit-exact restore) hoặc "high-fidelity WM"
    def verify(self, instance, bridge, policy, seeds) -> Outcome: ...  # outcome + true end obs / latent (luôn lưu)
```

- `Scorer` hiện có (`retryvla/search/scorers.py`) trở thành `ImaginationScorer(wm, critic, calibrator=None,
  penalty=None)`; CEM, proposals, verification, `RecoveryWriter` không đổi. Kiểm tra equivalence: với WM v1 + C^pi v2,
  scorer mới cho điểm bit-identical với `WMCriticScorer` trên 1 shard (test như `tests/test_af_default.py`).
- **Verifier luôn lưu end state thật** (obs + latent theo từng WM đã đăng ký) cho mọi candidate đã verify, thành công lẫn
  thất bại. Đây là nguyên liệu của S2 (student WM) và của mọi lever bên dưới; hiện `verif_*` đã có cho WM v1.
- **"WM card"** bắt buộc cho mỗi adapter (offline, 0 sim, trên 16,384 bridge của `verif_m5b` + 1,920 pick RQ3): open-loop
  ratio vs copy-frame (h1 / h4 / h10), critic AUC real / imagined, và bảng optimism (random vs search-selected:
  C(a), C(b), SR, FP, ECE, p@4). Không có card thì không dùng WM đó trong search.

| Adapter | Trạng thái | Việc cần làm | Chi phí |
|---|---|---|---|
| WM v1 (DINOv2 latent, `retryvla/wm/dynamics.py:LatentWM`) | có | bọc API | 0.5 ngày |
| V-JEPA 2-AC (arXiv 2506.09985; `retryvla/wm/vjepa2_ac.py:VJRanker`) | có, fine-tuned 3000 steps | bọc API; `finetune` = code `train_vjepa_ac.py` | 0.5 ngày |
| **Cosmos-Predict2.5-2B robot/action-cond** (arXiv 2511.00062; GitHub `nvidia-cosmos/cosmos-predict2.5`: robot/action-cond release 2025-10-14, có cả Predict2.5 robot/policy post-train trên LIBERO) | mới | action mapping LIBERO OSC delta 7-D -> action format của model; LoRA post-train trên `collect_v1` + `verif_*` video (agent-view; re-render từ snapshot trên CPU như V-JEPA); critic: encode frame sinh ra bằng DINOv2 -> dùng lại C^pi v2 (critic dùng chung qua encoder) | 3-5 ngày eng + ~20-40 GPU-h post-train; rollout video diffusion đắt (giây / candidate) -> chỉ dùng làm **tầng 2** rescoring top-K (K <= 32 / instance), không dùng trong CEM inner loop |

Claim boundary: "WM-agnostic" chỉ được nói sau khi một lever (2.2) cải thiện cùng chiều trên >= 2 WM (WM v1 + V-JEPA tối
thiểu; Cosmos là stretch). Cosmos trong sim là pixel WM duy nhất; kết quả của nó là bằng chứng gần nhất cho nhánh real
(Ctrl-World-like verifier, quyết định user 2026-11-02).

### 2.2 Làm WM + search tìm recovery tốt hơn: các lever đánh vào bottleneck

Instance set chuẩn: 256 M5b train failures (đầy đủ, n lớn cho power) với 127 RGR instances (shards 0-3) làm subset paired
với lịch sử; U = 73 single-unsolved (chỉ so paired trong cùng run; regression-to-mean -> control bắt buộc = `single` chạy
lại với seed mới, như F2). Ngân sách mặc định: **2800 env steps / instance (= 8 x 350)**, verification charged đầy đủ
(cả execution của bridge). Metric chính:
- **solved rate** (>= 1 verified) tại ngân sách cố định, **verified hits / 1k sim steps**, **unique-solved / 1k sim**
  (độ phủ failure là thứ quyết định distill, bài học M9 selection);
- **FP@pred>=0.5** của các pick, gap C(a) - C(b), ECE, p@4, tất cả theo N in {8, 32, 128, 512};
- solved trên U, per level (L1 / L2 / L3).
Power (xấp xỉ, McNemar alpha 0.10 hai phía, power 0.8, tỉ lệ discordant ~0.2): MDE ~10 pt ở n = 127, ~7 pt ở n = 256.
Primary test dùng n = 256; intake của từng lever tính lại MDE.

| # | Lever | Cơ chế (vì sao đánh vào bottleneck) | Offline trước (0 sim) | Target đo được | Decision rule (pre-registered khi intake) |
|---|---|---|---|---|---|
| L1 | **Verified-negative DAgger cho WM** (= student S2, mục 4.2) | WM được train trên rollout "bình thường"; search chọn đúng các action mà lỗi của WM lệch về phía C^pi thưởng. Thêm mọi candidate đã verify (đặc biệt các FP) vào data của WM với transition thật -> WM học dynamics ở chính vùng OOD mà search khai thác. Gần: CureWM 2610.09134 (execution-verified counterfactual FT, nhưng rollout thụ động), VLAW 2602.12063 (trộn success + failure gỡ over-optimism), WMPO 2511.09515 (policy-behaviour alignment) | train trên verif của round r, test trên pick của search KHÁC (`verif_rq3_sweep`, `verif_xwm_sweep`, RGR) | FP(a) @N=512 giảm >= 0.10 tuyệt đối; gap (a-b) giảm >= 30 %; p@4 +0.03 | online nếu offline đạt cả 3 (cluster bootstrap theo failure episode, CI > 0); giữ nếu online hits / 1k sim +15 % (sign-flip p < 0.10) và n01 >= n10 trên solved |
| L2 | **Critic học trên imagination + recalibrator** (T2Mem lead) | Thay vì sửa WM, học readout g(C^pi-embedding của ẑ_H) -> P(sim success) từ cặp (imagined, outcome thật); đã thấy +19 % hits, ~ ranking theo true state | cross-fit theo failure episode; held-out trên verif dir khác (pre-registered trong T2Mem, chưa làm) | hits r1-4 +0.3 / 16 verif trên held-out dir, prequential gain > 0 trên >= 2 / 3 thứ tự stream | online paired n = 256 như L1 |
| L3 | **Pessimistic / selection-aware scoring** | "optimizer's curse": max trên N candidate nhiễu luôn lạc quan (Smith & Winkler 2006, Management Science, không có arXiv). Score = LCB trên ensemble KHÁC NHAU thật (WM v1 seeds / bootstrap data, V-JEPA, Cosmos tầng 2) thay cho 4 head giống hệt (U == 0); hiệu chỉnh shrinkage theo N (học trên verif: E[SR | score, N]) | dùng lại xwm + RQ3 picks | FP giảm mà solved không giảm (xwm lam 1 giảm FP nhưng mất solved: bài học) | chọn hệ số trên train shards 0-1, đóng băng, test shards 2-7 |
| L4 | **Search trên option library + WM** (ACR-N) | Trên failure khó, `grasp_slow` cố định thắng WM search ~3x / step. Đưa options scripted (grasp_slow / hover / grasp_firm / place_goal / pull_free, pose từ geometry head = observation-only) vào proposal pool; WM + critic xếp hạng biến thể; sim verify | dùng outcome table ACR-N (97 instances x 80 trials) | solved trên U >= fixed `grasp_slow` + 5 pt, S >= 0.95 | **gate phụ thuộc F2** (agent khác): chỉ làm nếu head-pose `grasp_slow` giữ >= 80 % gain; nếu không, L4 dùng GT pose chỉ như upper bound |
| L5 | **Learned proposal (amortised search)** | Search hiện tại re-rank pool; M11 cho thấy plan-from-noise thua library. Học proposal q(a | s_f) (flow head nhỏ hoặc LoRA SmolVLA "recovery head") trên verified bridges round <= k (train only), dùng để sample pool + warm-start CEM | coverage: best-of-pool oracle success (verif ngẫu nhiên 64 / instance từ pool mới vs pool cũ) | coverage +20 % tương đối; hits / 1k sim +15 % | nếu coverage không tăng -> dừng (selection không cứu được pool không có nghiệm) |
| L6 | **Multi-fidelity + budgeted verification** | RQ3: ranking theo TRUE end state nâng p@4 0.193 -> 0.237. Tầng giữa rẻ: chỉ chạy bridge trong sim (H = 50 steps) rồi C^pi(true end) -> chỉ continuation 300 steps cho top. Kết hợp phân bổ adaptive giữa instance (successive halving, stop-on-success). Bài học T2Mem: tối ưu hits / verif làm mất solved -> objective = **unique-solved / 1k sim** | mô phỏng offline trên verif_m5b (đã có true end latent cho cả 64 bridge / instance) | unique-solved / 1k sim +20 % ở ngân sách 2800 | online paired; nếu solved giảm > 2 pt thì bác bỏ dù hits tăng |
| L0 | **Controls** | `single` (wm_unc_l0 pipeline), random search, **no-WM sim-only search cùng sim budget** (random / oracle CEM + continuation acceptance), fresh-seed `single` | - | - | mọi lever báo cáo vs cả ba |

Thứ tự: L2 và L6 gần như 0 sim (có sẵn dữ liệu) -> làm đầu; L1 cần GPU train WM (~2-4 GPU-h / round với WM v1, ~10 GPU-h
V-JEPA); L4 chờ F2; L5 sau khi có >= 2 round dữ liệu; L3 rẻ nhưng cần ensemble WM thật (train 3-5 seed WM v1 ~6 GPU-h).
Oracle arm cho Phase 1 = ranking bằng TRUE end state (sim bridge-only) và oracle-sim CEM: nếu oracle không vượt `single`
ở cùng ngân sách thì lever nào cũng không thể vượt -> chẩn đoán trước (idea-loop step 3).

### 2.3 Vòng lặp pi_k -> C^{pi_k} -> search -> verified data -> pi_{k+1} (+ WM_{k+1})

```
round k:  failures F_k của pi_k trên TRAIN scenes (seed protocol mới)
          -> C^{pi_k}: refit trên continuation THẬT của pi_k (từ verify + rollouts) [bắt buộc; M6 dùng critic cũ: 0 gain]
          -> WM_k: finetune trên mọi transition đã verify của round <= k-1 (L1 = student S2)
          -> search (best Phase-1 config, ngân sách cố định B / round) -> verify với continuation của pi_k
          -> bank_k (successes + negatives, đều lưu true end state)
          -> students: pi_{k+1} (S1 VLA / S3 DP) từ bank_{<=k}; WM_{k+1}, C^{pi_{k+1}} refit
```
- Control mỗi round: pi_k + cùng số SFT steps trên bank cũ (tách "dữ liệu mới" khỏi "train thêm", bài học M6).
- **Stopping rules** (đo trên PILOT, không trên test): dừng nếu (a) round gain vs control < 2 pt RSR (scene bootstrap
  CI chứa 0), hoặc (b) verified yield trên F_k < 50 % của round 1 ở cùng ngân sách, hoặc (c) nominal giảm > 2 pt so với
  pi1-nominal, hoặc (d) đã 3 round. **Nominal guard** mỗi round: nếu (c) xảy ra, thử WiSE-FT (mục 4.1) trước khi bỏ round.
- Báo cáo per round: |F_k|, solved / yield, FP của WM_k trên pick của round k+1 (S2 metric), RSR / nominal của pi_{k+1}.

## 3. Phần B: dựng dữ liệu (failure + recovery) từ dataset không phải LIBERO-Recover

### 3.1 LIBERO-Recover thực sự làm gì (đã kiểm trên arXiv HTML 2609.05178 v3, 2026-11-17)
- Scenario = (I, s_0, tau_fail, s_f, g, r). **"Rather than manually perturbing task states to synthesize failures,
  LIBERO-Recover grounds every scenario in a failure that actually occurred"**; "No objects, environment states, or
  initial configurations are manually perturbed, and no failures are injected during execution."
- Stage 1 Task Execution: 4 suite / 130 subtask, init gốc; 6 policy làm failure generator; render video.
- Stage 2 Failure Localization: Qwen3.5-27B chia video thành V_pre / V_fail / V_post; lấy robot sim state + object pose
  tại biên -> scene (`step_XXXX.bddl` + `step_XXXX_sim_state.npy`, thường 2 timestamp cách 20 steps / episode).
- Stage 3 Characterization: VLM gán level (L1 action retry, L2 action adaptation, L3 object-state recovery, L4
  environmental recovery) + category (grasp_failure, placement_failure, object_displacement,
  collision_or_interaction_failure, environmental_obstruction, other) theo "minimum corrective behavior"; tác giả spot-check.
- Stage 4: fine-tune bằng 3,184 teleop recovery (413 scenario, bị loại khỏi eval) rồi chạy lại để thu thêm failure.
- Phân bố level tự nhiên lệch (ít L4), giữ nguyên không rebalance.
- Eval: 5 gripper-closed + 5 open, ±2 cm xy noise trên free-joint objects (có bug off-by-one indexing, xem
  LIBERO_RECOVER_BASELINES §4), 520 steps.
- **Mâu thuẫn cần kiểm**: RESULTS (Per-level) ghi "upstream L4 are mostly constructed", paper nói không inject. Mở lại
  annotation L4 upstream trước khi dùng L4 làm claim.
- Đã có local: scene chính thức Goal 91 / LIBERO-10 207 / LIBERO-90 337 (`data/assets/scenes/scenes/`), và state mỗi
  20 steps (`data/assets/scenes_full/assets_times`, ~109k bddl); level chính thức chỉ có cho Spatial / Object.

### 3.2 Pipeline `retryvla/suitegen/` (mới), dùng được cho mọi LIBERO-type suite và mọi sim có restore state
1. **Harvest failure tự nhiên (nguồn chính, trung thành với LIBERO-Recover).** Chạy nhiều failure generator: SmolVLA
   base pi0, pi1-nominal, demo-only, pi1, các checkpoint trung gian (2k / 4k steps), tuỳ chọn StarVLA 4in1 (HF, cần env
   `starvla`); x 50 init x >= 3 seed. Lưu flat MuJoCo state mỗi 10 steps (như `scenes_full`). Nhiều generator = đa dạng
   failure mode (lesson của paper).
2. **Phát hiện + định vị failure tự động (không cần VLM trong sim).** Episode thất bại (BDDL `done` không bật trước T).
   Thời điểm failure t_f = sự kiện "không thể đảo ngược" đầu tiên từ predicates đặc quyền: grasp lost (mất finger contact
   khi object đang nâng / z giảm > 3 cm), object displaced (> 3 cm so với nominal-expected, hoặc rơi khỏi support / ra khỏi
   bàn), tip (up-axis z < 0.8), non-target object bị đẩy > 3 cm, fixture state đảo (drawer đóng lại, stove tắt), sub-goal
   predicate bị hoàn tác; fallback: stall (eef speed < thr và không đổi contact trong 40 steps) hoặc C^pi tụt > 0.3 trong 20
   steps. Snapshot tại t_f + {20, 40} steps (post-failure, như step_0180 / 0200). Tuỳ chọn Qwen3-VL cho localisation để so
   với recipe gốc.
3. **Perturbation injection (generator bổ sung, nhãn `injected`, báo cáo riêng).** Tại pha chọn ngẫu nhiên (pre-grasp,
   grasp, transport, pre-place): object pose noise (xy 2-8 cm, yaw ±30°), grasp drop (mở gripper khi transport), push /
   knock (impulse ngang lên object), tip-over, intrusion của distractor vào đường đi / lên goal (L4), fixture reset
   (đóng drawer / tắt stove; L4), gripper-state flip, eef offset; sau đó để policy chạy tiếp 20-60 steps để state là
   "hệ quả của policy" (gần tự nhiên hơn). Gần: HELM 2604.18791, RoboRecover 2609.28952 (replay action prefix), LIBERO-Plus
   2510.13626 / LIBERO-PRO 2510.03827 (perturbation robustness), FailSafe 2510.01642, AHA 2410.00371.
4. **Level taxonomy L1-L4 (rule-based từ state đặc quyền + VLM tuỳ chọn).** L1: task-relevant objects trong 2 cm và
   upright so với cấu hình kỳ vọng trước failure; L2: thay đổi giới hạn (target dời 2-5 cm, vẫn upright, vẫn graspable);
   L3: state của target đổi cần restore (tipped, rơi khỏi support, sai container, dời > 5 cm); L4: non-target object /
   fixture đổi và chắn task. **Calibrate** rule với 1,408 nhãn chính thức (Spatial 269 + Object 1,139): Cohen's kappa >= 0.6
   mới dùng; báo cả confusion matrix.
5. **Restore-from-snapshot.** Scene lưu đúng format LIBERO-Recover (bddl + flat state) để evaluator chính thức và
   `retryvla/envs/libero_env.py::load_scene` đều chạy; replay check bit-exact (physics) + render; OSC goal re-anchor; 20
   settle steps; protocol 5 + 5 gripper; sửa / ghi rõ bug noise index.
6. **Splits chống leakage.** Đơn vị leakage = (task, init index, generator run) = failure episode; mọi timestamp của
   cùng episode cùng split. LIBERO-Goal: 10 task dùng **chung một layout** -> split theo init index (và task), không theo
   layout. Với held-out Goal: scene chính thức (91) là TEST; scene tự dựng chỉ từ init index KHÔNG xuất hiện trong episode
   của scene chính thức (map episode_id -> init index; nếu không map được thì check trùng sim state frame 0).
7. **Quality checks** (báo cáo, và ghi nhãn chứ không xoá khỏi tập "all"):
   - feasible: oracle sim CEM (ngân sách lớn, 1024 sims) hoặc option library giải được >= 1 lần;
   - non-trivial: pi0 RSR trên scene < 0.8 qua 10 trial protocol; placebo re-run (2610.06921: 43 % failure thành công khi
     chạy lại);
   - diversity: dedupe theo khoảng cách object-pose / latent (cap 2 scene / episode), histogram task x level x category;
   - physics sanity: không penetration > 1 cm, không object lơ lửng;
   - **external validity**: trên Goal, so phân bố level / category và pi0 / pi1-nominal RSR giữa scene tự dựng và scene
     chính thức (mục tiêu |ΔRSR| <= 5 pt, thứ hạng policy giữ nguyên).
8. **Recovery data** cho suite mới = chạy Phase 1 engine trên train scenes của suite đó (WM + critic refit trên data của
   suite: collect bridges + continuation labels như M2 / M10).

Chi phí ước tính cho LIBERO-Goal: harvest 4 generator x 500 episode x ~400 steps ~0.8M steps (~4-6 h trên 32 CPU + 1 GPU
cho policy); level rules + calibration 2 ngày eng; feasibility check ~1k scene x 1024 sims (giới hạn bằng early stop) ~1 ngày
CPU; WM / critic refit ~6 GPU-h. Tổng ~1.5 tuần eng + compute.

### 3.3 Ánh xạ sang dữ liệu thật sau này (quyết định user 2026-11-02)
Không có snapshot restore ngoài đời: "restore" = điều kiện hoá một video WM kiểu Ctrl-World (2510.10125) trên frame /
proprio tại t_f. Failure lấy từ log deployment (DROID ~19k failure, RoboMIND failures có nguyên nhân), localisation bằng
VLM + progress / C^pi; cùng interface Phase 1, nhưng `Verifier` = high-fidelity WM (nhãn "WM-verified", không bao giờ
"sim-verified"), cộng một tập nhỏ verify thật làm anchor. Thí nghiệm sim nối với nhánh này: đo FP / ECE của Cosmos tầng 2
khi làm verifier so với sim (mục 2.1) -> biết "WM-verified" đáng tin đến đâu, và S2 (WM học từ verified negatives) là đúng
cơ chế giảm optimism mà nhánh real cần (VLAW báo cáo hiện tượng tương tự).

## 4. Phase 2 (C): distillation với recipe của foundation model, cho ba student

Ngân hàng dữ liệu chung (output của Phase 1, mỗi round k): cho mỗi failure state s_f: các bridge đã verify kèm outcome
(success / fail, success step), continuation của pi (action + obs), end state thật (obs + latent), imagined end state +
điểm của mỗi WM, level, generator. **Cùng một bank phục vụ cả ba student**: đây là nội dung của claim "policy-agnostic
recovery data engine".

### 4.1 Survey recipe (arXiv ID đã mở trên arxiv.org/abs ngày 2026-11-17; "unverified" = không mở được)

Cột "Cần": L = nhãn success / fail, N = negatives, O = tương tác online (sim), P = cặp paired cùng state.
Chi phí trên 8b889199 GPUs 1-3 (3 x A100 80GB), SmolVLA 450M (trainable 97.5M action expert theo config hiện tại).

| Recipe | Nguồn (arXiv) | Cần | Lợi ích kỳ vọng cho recovery | Rủi ro nominal | Hợp với SmolVLA flow matching? | Chi phí | Quyết định |
|---|---|---|---|---|---|---|---|
| **SFT + nominal replay** (co-training, control hiện tại) | pi0 2410.24164; data-mix: Re-Mix 2408.14037, Data Mixing Laws 2403.16952, data scaling IL 2410.18647 | L | đã đo: 0.418 | thấp ở r 0.5 (r 0.7 fail eps) | có | 1.3 GPU-h / seed | **control mọi arm** |
| Data-mix law nhỏ | 2403.16952, 2408.14037 | L | chọn r và trộn level / suite có nguyên tắc (fit RSR(r), NSR(r) từ P3 sweep + 2 điểm mới) thay vì đoán | - | có | 4-6 SFT | làm trong M-P2.1 (rẻ) |
| Knowledge Insulation | pi0.5 KI 2505.23705; pi0.5 2504.16054 | L | giữ VLM backbone khỏi gradient của action expert; ở ta VLM + vision đã frozen (`train_expert_only`) = dạng KI tầm thường | thấp | có | 0 | **không cần**, trừ khi mở VLM (LoRA trên VLM) -> khi đó stop-gradient từ expert vào VLM + co-train FAST-token / VL data |
| **RECAP / advantage-conditioned** | pi*0.6 2511.14759; CFG-RL: Diffusion Guidance as policy improvement 2505.23458; RISE / TACO (overlap audit) | L, N | dùng CẢ negatives: train trên bridge thành công và thất bại với token "advantage tốt / xấu" (advantage = outcome - C^{pi}(s_f), hoặc C^pi(true end) - C^pi(s_f)); inference điều kiện "tốt" (+ CFG weight). Tận dụng 85-90 % dữ liệu đang bị bỏ | trung bình (token mới trên nominal: gán "tốt" cho demo) | có: thêm 1 token / embedding vào action expert; CFG trên flow velocity | ~1.5x SFT | **shortlist #1** |
| AWR / weighted flow matching | AWR 1910.00177; VLAW 2602.12063 (weighted FM); `retryvla/train/awr.py` skeleton | L, N(soft) | trọng số exp(A/beta) cho bridge; đơn giản hơn RECAP | thấp | có | = SFT | ablation của RECAP |
| **Preference optimisation trên cặp (success, fail) cùng s_f** | DPO 2305.18290; Diffusion-DPO 2311.12908; GRAPE 2411.19309 | P | dữ liệu paired hoàn hảo (cùng snapshot, nhiều bridge, có cả hai outcome). Với flow matching: thay log-lik bằng hiệu MSE velocity (như Diffusion-DPO) so với reference pi1 | trung bình-cao (DPO có thể đẩy ra xa demo); cần reference + beta | được (Diffusion-DPO áp dụng cho velocity/epsilon loss) nhưng chưa có bản chuẩn cho flow VLA -> là rủi ro kỹ thuật | ~2x SFT (ref forward) | **shortlist #2** (chỉ trên recovery segment, SFT + replay giữ nguyên) |
| **Rejection-sampling FT / expert iteration** | ExIt 1705.08439; STaR 2203.14465; RFT 2308.01825; ReST 2308.08998; SI-EFM 2509.15155 | L, O | chính là vòng lặp pi_k (2.3) với policy làm proposal; điểm khác: sample từ pi_k để lấy thêm on-policy recoveries | thấp-trung bình (SI-EFM: giảm khi train quá đỉnh) | có | mỗi round ~1 ngày sim + 1.3 GPU-h / seed | **cấu trúc của M-P2.3** |
| DAgger multi-round | DAgger 1011.0686; WM-DAgger 2604.11351 | O + expert | "expert" = search + verify; on-policy failures mỗi round | thấp | có | như trên | gộp vào ExIt |
| RL FT flow policy (online) | ReinFlow 2505.22094; Flow-GRPO 2505.05470; pi_RL 2510.25889 (RLinf 2509.15965); VLA-RL 2505.18719; SimpleVLA-RL 2509.09674; DPPO 2409.00588; FPO 2507.21053; RIPT-VLA 2505.17016 | O nhiều | trực tiếp tối ưu RSR từ failure snapshot (reset vào s_f = lợi thế của sim) | **cao**: B3.3 residual RL collapse nominal (0.022); SimpleVLA-RL "pushcut" exploit predicate | được (pi_RL làm cho pi0 / pi0.5 flow) nhưng MuJoCo CPU-bound: ~10^6-10^7 steps | lớn (ngày CPU) | **chỉ làm baseline**, bản rẻ: GRPO-style trên failure snapshots với KL tới pi1 + 50 % nominal BC |
| Noise-space RL (DSRL) | DSRL 2506.15799 | O | RL trên noise đầu vào của flow policy, không đổi weights: an toàn cho nominal, sample-efficient | rất thấp (weights frozen) | có (SmolVLA nhận noise đầu vào) | nhỏ (MLP noise policy) | **baseline RL rẻ** (và là ứng viên deploy: noise policy chỉ bật khi C^pi thấp) |
| PA-RL / RLDG / PLD | PA-RL 2412.06685; RLDG 2412.09858; PLD 2511.00091 | O | RL specialists rồi distill vào generalist = phiên bản RL của data engine | thấp sau distill | có | lớn | so sánh ở mức related work; PLD ~ B3.3 |
| LoRA vs full FT | LoRA 2106.09685 | - | LoRA trên action expert / VLM: ít quên hơn? | thấp | có | ~ = SFT | ablation 1 seed x 2 (LoRA r 32 trên expert vs full expert vs + VLM LoRA với KI) |
| KL / behaviour regularisation | L_preserve hiện có (P3: không cần); WiSE-FT 2109.01903 | - | WiSE-FT: theta = (1-a) theta_pi1nom + a theta_new, chọn a trên pilot: guard nominal không tốn train | - | có | 0 train, vài eval | **nominal guard mặc định** khi vượt eps |
| Curriculum theo level | (Kintsugi 2609.31048 frontier; không có nguồn chuẩn cho VLA) | L, level | L1 -> L2 -> L3 hoặc oversample L3 (L3 là chỗ pi1 yếu: 0.383 vs L2 0.448) | thấp | có | = SFT | ablation rẻ trong M-P2.1 |
| ConRFT | 2502.05450 | L + online | consistency policy + RL; real-world | - | khác kiến trúc | - | chỉ related work |

Đề xuất thứ tự cho S1 (VLA): (1) SFT control + mix-law + curriculum (rẻ), (2) RECAP vs AWR, (3) Diffusion/flow-DPO trên
cặp paired, (4) ExIt 2-3 round với critic + WM refit, (5) DSRL / GRPO-lite làm baseline RL; WiSE-FT áp dụng khi cần.

### 4.2 Ba student

**S1. VLA (SmolVLA; hiện tại).**
- Dữ liệu: verified bridges + continuation (action-grounded; đã biết action-free vô ích), negatives cho RECAP / DPO, nominal
  demos 50 %. Recipe: mục 4.1.
- Eval: Spatial test RSR (dev) + per level, **Object test** (held-out, frozen recipe, 3 seeds), **Goal official 91 scenes**
  (held-out mới qua Phần B), nominal của suite tương ứng (eps 2 pt vs pi1-nominal, 3 seeds), inference cost = 0 overhead.
- Chi phí: ~1.3 GPU-h / SFT seed + 2 h eval -> ~10 GPU-h / arm (3 seeds, 2 suites).

**S2. World model (+ critic) làm student: "WM self-improvement from verified negatives".**
- Dữ liệu / nhãn: mọi transition đã verify (bridge actions, chuỗi obs thật, end state thật), outcome của continuation;
  quan trọng nhất là các **false positives** (imagined C cao, sim fail) và near-misses. Không cần nhãn mới: verifier đã
  lưu sẵn. Critic nhận (imagined latent, outcome thật) để học hiệu chỉnh.
- Recipe:
  (a) WM-DAgger-for-the-WM: fine-tune dynamics trên D_collect ∪ D_verif(<=k), oversample FP 2-4x, loss multi-step trên
  latent thật (không teacher forcing ở h > 1, vì search khai thác lỗi compounding);
  (b) failure-contrastive term (CLWM 2609.05834, CureWM 2610.09134): imagined end của bridge thất bại phải gần end thật thất
  bại và xa cụm "success end states" của cùng task;
  (c) critic-on-imagination: C^pi train trên (ẑ_H của WM_k, outcome thật) + (z_H thật, outcome) -> critic học không tin
  chỗ WM bịa; hoặc recalibrator g (T2Mem lead) là bản rẻ;
  (d) ensemble thật (bootstrap data / seed) cho L3.
  Tuỳ WM: WM v1 full FT (~2-4 GPU-h); V-JEPA 2-AC chỉ AC predictor (~10 GPU-h); Cosmos LoRA (~20 GPU-h).
- Eval (search-level, không phải RSR): trên pick của **search round k+1** (tức là sau khi search được phép khai thác WM_k
  mới, đây là test thật, không phải trên dữ liệu cũ): FP@0.5, gap C(a) - C(b), ECE, AUC(a) trên random vs selected, curve
  theo N in {8, 32, 128, 512}; **verified yield / 1k sim steps**, solved rate, unique-solved / 1k sim tại 2800 steps; open-loop
  card (không được xấu đi > 10 % trên val nominal). Held-out: train WM_k trên shards 0-3, đo trên shards 4-7 và trên Goal.
  Thành công nếu: FP@N=512 giảm >= 0.10 VÀ yield / 1k sim +15 % (p < 0.10) trên >= 2 WM.
- Vòng nối: WM_k tốt hơn -> search round k+1 yield cao hơn -> bank lớn / sạch hơn -> S1 / S3 tốt hơn. Báo cáo cả "tác động
  hạ nguồn": RSR của pi_{k+1} khi bank do WM_k vs WM_0 sinh, cùng ngân sách sim.
- Gần nhất: CureWM (thụ động), VLAW (co-improvement policy + WM, real), WoVR / WMPO (WM align với policy rollouts),
  Rollback (LoRA-refine WM bằng transition thật). Khác: ta đo và sửa **optimism dưới search pressure**, train trên
  verified negatives do chính search chọn, kiểm trên round sau.

**S3. Diffusion Policy (policy-agnostic; sân của SAILOR).**
- Hai setting:
  (i) **LIBERO-Recover**: DP (Diffusion Policy 2303.04137, U-Net hoặc transformer, ResNet image encoder, chunk 16 / exec 8)
  train trên LIBERO-Spatial demos -> DP-nominal là pi0 mới; failure của nó = harvest mới (Phần B, vì scene LIBERO-Recover
  là failure của VLA khác: vẫn eval được trên chúng). C^{DP} train trên continuation của DP; search + verify với
  continuation của DP; distill vào DP. Eval: RSR Spatial / Object test + nominal Spatial.
  (ii) **robomimic / robosuite (SAILOR 2506.05294)**: Lift, Can, Square (robomimic 2108.03298, demos có sẵn) + 1-2 robosuite
  (Door, Stack; robosuite 2009.12293). Base DP train trên N demo nhỏ (như SAILOR); failure = rollout thất bại của DP
  (snapshot qua `env.sim.get_state()`), Phần B pipeline với predicates của robosuite. So sánh: SAILOR (code GitHub
  công bố) ở **cùng số env interaction** vs RetryVLA engine (WM + C^{DP} + verify + distill) vs DP + DAgger-style
  sim-only search. Metric: nominal success (SAILOR báo cáo nominal) + recovery-from-failure-state success (của ta).
- Dữ liệu cần: verified bridges + continuation; negatives nếu dùng DPO / RECAP biến thể cho diffusion (Diffusion-DPO áp
  dụng trực tiếp; DPPO 2409.00588 là baseline RL chuẩn của DP).
- Chi phí: DP train ~2-4 GPU-h / task / seed (nhỏ), robomimic sim rẻ hơn LIBERO; port engine (env wrapper, snapshot,
  predicates, WM v1 retrain trên robomimic images) ~1-1.5 tuần; SAILOR reproduction 1 tuần (rủi ro: hyperparams).
- Novelty: (1) "policy-agnostic": cùng engine cải thiện VLA flow-matching và DP diffusion; (2) head-to-head với SAILOR:
  SAILOR search tại test time + ExIt distill, không verify; giả thuyết của ta: verify + C^pi > discriminator reward khi
  search sâu (họ báo "robust to reward hacking" ở 8x; RQ3 của ta nói ngược lại: đây là một thí nghiệm phân xử).

## 5. Ma trận thí nghiệm + baselines (D)

Chuẩn chung: paired (cùng instance / snapshot / seed protocol), equal budget (sim steps cho search; trajectory count + SFT
steps cho distill, `train.match_trajectory_count_of`), train / pilot only để tune, 3 seeds trước mọi claim, nominal <= 2 pt
drop (mean 3 seeds), per-level L1-L4, McNemar + scene x seed bootstrap.

### 5.1 Search-level (Phase 1; student S2 cũng đo ở đây)
| Arm | WM | Scorer / proposals | Verify | Vai trò |
|---|---|---|---|---|
| A0 `single` (+ fresh seed) | WM v1 | C^pi, library + VLA samples, CEM | sim | control |
| A1 no-WM sim-only | - | random / library, continuation acceptance | sim | "2610.05151-style", cùng sim budget |
| A2 oracle | sim | true end state / oracle CEM | sim | upper bound |
| A3 L2 recalibrated critic | WM v1, VJ | C^pi + g | sim | lever |
| A4 L1 / S2 WM_k (verified-negative FT) | WM v1, VJ, (Cosmos tầng 2) | C^pi_k | sim | lever + student S2 |
| A5 L3 pessimistic LCB | ensemble thật | LCB + N-shrinkage | sim | lever |
| A6 L4 option library + WM | WM v1 | scripted options (head pose) + VLA | sim | gate F2 |
| A7 L5 learned proposal | WM v1 | q(a|s_f) + CEM warm-start | sim | lever |
| A8 L6 multi-fidelity | WM v1 -> sim bridge-only -> full | | sim | lever |
| A9 best combo | 2 WM | | sim | xác nhận WM-agnostic |
Metric: solved, hits / 1k sim, unique-solved / 1k sim, FP / gap / ECE theo N, solved trên U, per level; wall-clock.

### 5.2 Policy-level (Phase 2; S1 VLA, S3 DP)
| # | Arm | Dữ liệu | Recipe | Test-time module | Gần với |
|---|---|---|---|---|---|
| P0 | demo-only | nominal | SFT | - | control |
| P1 | pi1 (hiện tại) | verified bank r1 | SFT + replay | - | ref |
| P2 | engine-best bank | bank từ A9 | SFT + replay | - | Phase 1 -> policy |
| P3 | RECAP | bank + negatives | advantage-cond. | - | pi*0.6 |
| P4 | flow-DPO | cặp paired | DPO-style + SFT | - | Diffusion-DPO / GRAPE |
| P5 | ExIt 2-3 rounds | bank_k, WM_k, C^{pi_k} | best of P2-P4 | - | ExIt / DAgger |
| P6 | DSRL / GRPO-lite từ failure snapshots | online sim | RL + KL + nominal BC | noise policy (DSRL) | RL baselines |
| B1 | **Back-on-Track-style test-time** | - | - | monitor (C^pi < tau) -> CEM trong WM -> hand back, trên pi0 và pi1 (+ "both": P5 + planner) | 2610.19579 |
| B2 | **SAILOR-style** | WM + discriminator reward (expert vs learner state) + critic ensemble mean - std, search, distill KHÔNG verify (+ biến thể có verify để tách hai yếu tố) | ExIt | (tuỳ chọn MPC) | 2506.05294 |
| B3 | **unverified WM data** (Dream2Fix / WM-DAgger style) | top WM picks theo WM verdict, cùng count | SFT | - | 2603.13528, 2604.11351 |
| B4 | no-WM sim-only search, cùng sim budget | A1 bank | SFT | - | 2610.05151 |
| B5 | teleop "Combined" | `ataier/LIBERO_Recovery_Expert` (leakage check) | SFT | - | 2609.05178 |
| B6 | scripted regrasp distill | ACR-N options (nếu F2 qua gate) | SFT | - | ACR-N |
| B7 | Kintsugi frontier start states | MC recoverability của pi0 | SFT | - | 2609.31048 |
| B8 | Backtrack (đã có) | - | - | rewind | 2610.32941 |
Cho S3 (DP): P0, P1, P2, P5 + B2 (SAILOR chính chủ trên robomimic), DPPO baseline, B4.

Metric: RSR overall + L1-L4 (Spatial dev, Object held-out, Goal-official held-out), nominal (eps 2 pt), sim cost thu thập
(steps, có tính WM / critic training khấu hao), inference cost (ms / step, số WM call), 3 seeds.

### 5.3 WM-level (student S2)
| # | Arm | Train data | Eval |
|---|---|---|---|
| W0 | WM_0 (collect_v1) | - | card + FP-curve + online yield |
| W1 | + verified successes only | D_verif^+ | như trên |
| W2 | + verified successes + negatives | D_verif | như trên |
| W3 | W2 + FP oversampling + failure-contrastive | D_verif | như trên |
| W4 | W2 + critic-on-imagination | D_verif | như trên |
| W5 | random-action data cùng số transitions (control "chỉ là thêm data") | D_random | như trên |
W2 vs W5 tách "data đúng chỗ (search-selected) vs thêm data"; W2 vs W1 tách vai trò negatives. Lặp cho WM v1 và V-JEPA.

## 6. Milestones, effort, go / no-go (E)

Thứ tự ưu tiên rẻ -> đắt; mỗi milestone có intake trong PLAN.md trước khi chạy. GPU-h là A100; CPU-h trên 32 cores local.

| M | Nội dung | Effort | Gate (go / no-go) |
|---|---|---|---|
| **N0** | API `WorldModel` / `Critic` / `Verifier`, bọc WM v1 + VJ, equivalence test, WM cards; verifier lưu true end state cho mọi WM | 3 ngày eng, ~4 GPU-h | equivalence bit-identical; không đụng run của agent khác |
| **N1** | Offline levers (0 sim): L2 recalibrator held-out dir; L6 multi-fidelity mô phỏng; S2 W0-W5 offline (WM v1) trên verif_m5b -> test trên rq3 / xwm / RGR picks | 4-5 ngày, ~15 GPU-h | >= 1 lever đạt target offline (mục 2.2) -> N2; nếu không: chẩn đoán + review (idea-loop 3-4) |
| **N2** | Online paired search n = 256 (A0-A3, A4, A8; A6 nếu F2 qua) | ~3 ngày CPU (16 workers), ~10 GPU-h | lever giữ nếu hits / 1k sim +15 % (p < 0.10) và solved không giảm; **G-P1**: có combo solved +7 pt (n 256, p < 0.10) hoặc unique-solved / 1k sim +20 % -> bank mới cho Phase 2 |
| **N3** | Phần B trên LIBERO-Goal: harvest, detection, level rules + kappa, quality, external validity vs 91 official; WM / critic refit cho Goal | 1.5 tuần | kappa >= 0.6, |ΔRSR pi0| <= 5 pt vs official; nếu fail: báo cáo pipeline như "injected / ours" và không dùng làm held-out claim |
| **N4** | S1 recipe shoot-out trên Spatial với bank cố định (pi1 bank + bank N2): P1-P4 1 seed screen, top 2 x 3 seeds; mix-law / curriculum / LoRA / WiSE-FT | 1.5 tuần, ~60 GPU-h | recipe giữ nếu RSR +3 pt vs P1 (scene x seed bootstrap CI > 0) và nominal pass |
| **N5** | Vòng lặp 2-3 round (2.3) với S2 (WM_k) + S1; control "train thêm" mỗi round | 2 tuần, ~40 GPU-h + CPU | stopping rules 2.3; **G-loop**: round 2 > control và FP của WM_k trên pick round k+1 giảm |
| **N6** | Baselines B1 (Back on Track), B2 (SAILOR-style), B3, B4, B5 trên Spatial + Object | 2 tuần (B1, B2 eng ~1 tuần), ~60 GPU-h | không có gate; mọi claim "engine > X" cần thắng X ở 3 seeds |
| **N7** | S3 Diffusion Policy: (i) DP trên LIBERO (pi0-DP, harvest, engine, distill); (ii) robomimic / robosuite vs SAILOR | 3 tuần, ~80 GPU-h | **G-DP**: engine cải thiện recovery của DP >= 5 pt với nominal pass ở (i) hoặc (ii) -> claim policy-agnostic; head-to-head SAILOR báo cáo dù thắng hay thua |
| **N8** | Frozen held-out: Object test + Goal-official, 3 seeds, recipe đóng băng trước khi nhìn | 3 ngày, ~30 GPU-h | kết quả cuối, chỉ chạy một lần |
| N9 (cond.) | Cosmos-Predict2.5 adapter + tầng 2 verifier / WM student | 1-1.5 tuần, ~60 GPU-h | chỉ nếu N1-N2 cho thấy lever có hiệu lực trên 2 WM latent và còn GPU |

Tổng ~10-12 tuần nếu tuần tự; N3 và N7(ii) có thể chạy song song với N1-N2 (CPU / eng khác nhau). GPU 1-3 hiện đang bận
bởi F1 / F2: N0 / N1 bắt đầu trên CPU + GPU trống sau khi F2 xong.

## 7. Rủi ro
- **Không lever nào phá được optimism** (lỗi WM tương quan, như xwm): khi đó claim chuyển sang "multi-fidelity
  verification + learned proposal" (giảm phụ thuộc vào imagination) và kết quả phân tích; S2 vẫn là phát hiện (âm hay dương).
- **Search gain không thành policy gain** (đã xảy ra: B4 critic, Object comp ~ random): luôn đo P2 vs P1 ở cùng count;
  nếu không, tối ưu unique-solved (coverage) thay vì hits.
- **RECAP / DPO làm hỏng nominal**: WiSE-FT + nominal replay; DPO chỉ trên recovery segment.
- **S2 overfit vào instance đã thấy**: luôn đo trên round sau / shard khác (không bao giờ trên pick dùng để train).
- **Phần B không khớp distribution chính thức** (VLM vs rule levels; generator khác): báo cáo external-validity check,
  claim trên Goal-official thay vì scene tự dựng.
- **DP / robomimic port tốn thời gian**; SAILOR reproduction có thể dưới số paper: báo cáo cả số paper và số tái lập.
- **Compute**: MuJoCo CPU-bound; RL arms có thể vượt ngân sách -> giữ RL chỉ ở dạng DSRL / GRPO-lite.
- **Leakage**: teleop data (B5) không có mapping scenario; Goal split theo init index; split file mới được git-track.
- **Overlap tiếp tục tăng** (nhiều paper 10/2026): re-sweep related work trước khi viết (skill idea-review).

## 8. Novelty sau stage này (so với Back on Track 2610.19579 / SAILOR 2506.05294)
| Điểm | Back on Track | SAILOR | RetryVLA sau stage |
|---|---|---|---|
| Khi nào dùng WM | test time, không verify | test-time MPC + ExIt, WM refit online, không verify per-sample | train time, verify bit-exact, amortise vào student, 0 module lúc test |
| Optimism dưới search | nêu là hạn chế | báo "robust" ở 8x | đo (decomposition WM / critic theo N) **và sửa** bằng verified negatives (S2), kiểm trên round sau, trên >= 2 WM |
| Student | không (policy frozen) | Diffusion Policy | VLA flow (SmolVLA) + Diffusion Policy + chính WM -> "policy-agnostic recovery data engine" |
| Dữ liệu | perturbed init LIBERO-Goal | robomimic nominal | LIBERO-Recover + pipeline dựng suite recovery từ mọi LIBERO-type suite, kiểm chứng trên scene chính thức |
| Recipe distill | - | BC distill (ExIt) | so sánh có kiểm soát SFT / RECAP / flow-DPO / ExIt / RL dưới ràng buộc nominal |
Rủi ro novelty: CureWM / VLAW đã sửa WM bằng failure thật (thụ động / real); khác biệt phải nằm ở "negatives do search
chọn + đo theo search pressure + hạ nguồn tới student". 2610.05151 đã có "continuation acceptance": không claim.

## 9. Câu hỏi mở cho user
1. Held-out suite mới: chọn **LIBERO-Goal** (có 91 scene chính thức để kiểm chứng) làm chính, LIBERO-10 hay LIBERO-Plus /
   PRO làm stretch?
2. Perturbation-injected failures: chỉ là augmentation / phân tích, hay được phép vào claim chính?
3. Có muốn dùng StarVLA 4in1 checkpoints làm failure generator (thêm env `starvla`, ~2-4 GPU-h / model)?
4. Ưu tiên S3 Diffusion Policy ở setting nào trước: (i) DP trên LIBERO-Recover hay (ii) robomimic đối đầu SAILOR?
5. Ngân sách cho Cosmos-Predict2.5 (N9, ~60 GPU-h) có chấp nhận được không, hay giữ 2 WM latent?
6. GPU 0 của 8b889199 có được dùng không (hiện 7.5 GB đã chiếm)?
7. Thứ tự Phase 1 vs Phase 2: chạy N4 (recipe shoot-out trên bank hiện có) song song với N1-N2 ngay, hay đợi bank mới?

## 10. Nguồn đã kiểm (arXiv abs, 2026-11-17, title khớp)
LIBERO-Recover 2609.05178 (+ HTML v3 Sec. 2.4 / App.); pi0 2410.24164; pi0.5 2504.16054; KI 2505.23705; pi*0.6 2511.14759;
CFG-RL 2505.23458; AWR 1910.00177; ExIt 1705.08439; STaR 2203.14465; RFT 2308.01825; ReST 2308.08998; DPO 2305.18290;
Diffusion-DPO 2311.12908; GRAPE 2411.19309; ReinFlow 2505.22094; Flow-GRPO 2505.05470; pi_RL 2510.25889; RLinf 2509.15965;
VLA-RL 2505.18719; SimpleVLA-RL 2509.09674; RIPT-VLA 2505.17016; DPPO 2409.00588; FPO 2507.21053; DSRL 2506.15799;
PA-RL 2412.06685; RLDG 2412.09858; PLD 2511.00091; RL-100 2510.14830; VLA-RFT 2510.00406; DAgger 1011.0686; LoRA 2106.09685;
WiSE-FT 2109.01903; Data Mixing Laws 2403.16952; Re-Mix 2408.14037; data scaling IL 2410.18647; SI-EFM 2509.15155;
Cosmos-Predict2.5 2511.00062 (+ GitHub README: robot/action-cond, robot/policy LIBERO); V-JEPA 2 2506.09985; Ctrl-World
2510.10125; VLAW 2602.12063; Back on Track 2610.19579; SAILOR 2506.05294 (+ HTML: robomimic Lift / Can / Square, 6
robosuite, 3 ManiSkill); Diffusion Policy 2303.04137; robomimic 2108.03298; robosuite 2009.12293; Dream2Fix 2603.13528;
WM-DAgger 2604.11351; CureWM 2610.09134; Kintsugi-VLA 2609.31048; 2610.05151; liveness OPE 2605.11479; Backtrack 2610.32941;
RoboRecover 2609.28952; 2610.37196; 2610.06921; WoVR 2602.13977; WMPO 2511.09515; Hi-WM 2604.21741; LIBERO-Plus 2510.13626;
LIBERO-PRO 2510.03827; MimicGen 2310.17596; SmolVLA 2506.01844; ConRFT 2502.05450; RecoveryChaining 2410.13979;
V-GPS 2410.13816; RoboMonkey 2506.17811; FOREWARN 2502.01828.
Lấy từ audit trước, không mở lại hôm nay: CLWM 2609.05834, HELM 2604.18791, FailSafe 2510.01642, AHA 2410.00371 (xem
`docs/related_work/WM_RECOVERY_OVERLAP.md`, `docs/reviews/ACRN_REVIEW.md`). Không có arXiv: Smith & Winkler 2006
"The Optimizer's Curse" (Management Science). Chưa xác minh được: một bản DPO chuẩn cho flow-matching VLA (ta tự chuyển
Diffusion-DPO sang velocity loss; coi là rủi ro kỹ thuật).
