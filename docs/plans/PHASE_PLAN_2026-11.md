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
## A. Phase 1: WM-agnostic recovery data engine

### A.1 Interface mà mọi WM phải expose (`retryvla/engine/interfaces.py`, mới; default path không đổi)

```python
class WorldModel(Protocol):
    name: str
    env_steps_per_model_step: int          # WM v1: 5 (chunk); V-JEPA 2-AC: 5 (20 Hz -> 4 Hz); Cosmos AC: theo config
    action_space: str                      # "libero_osc_delta7" (adapter tự map, ví dụ DROID physical map)
    def encode(self, obs: dict) -> State: ...                     # real obs -> latent state (z, q)
    def rollout(self, s0: State, actions: Tensor[N, H, 7]) -> Traj: ...   # action-conditioned; tối thiểu trả về s_H
    def uncertainty(self, s0, actions) -> Tensor[N] | None: ...   # optional: ensemble / sampling / None
    def finetune(self, transitions: TransitionSet, budget: int) -> "WorldModel": ...   # optional, cho DAgger-WM
    cost_per_rollout_ms: float                                    # để tính compute-matched budget

class Critic(Protocol):            # C^pi trên latent space của *chính WM đó*
    def __call__(self, s: State, task: str) -> Tensor[N]: ...     # P(pi finishes | s); labels = real continuations
    def embed(self, s: State) -> Tensor[N, d]: ...                # cho recalibrator / probe

class Verifier(Protocol):          # ground truth của Phase 1
    def verify(self, inst, bridge, seeds) -> Outcome: ...         # SimVerifier (MuJoCo restore + pi continuation)
                                                                  # | WMVerifier (real data later, xem B.6)
```

- `ImaginationScorer(wm, critic, calibrator=None, penalty=None)` thay cho `WMCriticScorer` / `VJRanker.score`. Nó giữ
  nguyên interface `Scorer.score(instance, cands, iteration) -> ScoreOut`, nên `run_search`, CEM, `rgr_search` và
  `acr_search` không cần đổi.
- Critic là per-WM: train trên latent của encoder của WM đó, label từ real continuations như cũ. Pixel / video WM có thêm
  một đường thứ hai: decode frame -> DINOv2 encoder của WM v1 -> **dùng chung C^pi v2**. Nhờ vậy tách được lỗi của
  dynamics khỏi lỗi của critic.
- **Default-path equivalence test** (bắt buộc trước N1): với WM v1 + C^pi v2 qua interface mới, scores trên 256 x 64
  candidates của `verif_m5b` phải identical với `WMCriticScorer` (max |diff| <= 1e-6, bf16 có tolerance ghi rõ). Làm tương
  tự cho `VJRanker`.

**Adapters để chứng minh WM-agnosticism**

| Adapter | Trạng thái | Việc cần làm | Chi phí |
|---|---|---|---|
| WM v1 (DINOv2 6x6 pooled, `retryvla/wm/dynamics.py:LatentWM`) | có | wrap | 0.5 ngày |
| V-JEPA 2-AC ([2506.09985](https://arxiv.org/abs/2506.09985) verified; `retryvla/wm/vjepa2_ac.py`) | có (fine-tuned 3k steps) | wrap + `finetune()` | 0.5 ngày |
| Cosmos-Predict2.5-2B/robot/action-cond ([2511.00062](https://arxiv.org/abs/2511.00062) verified; model card + post-training guide trên GitHub `nvidia-cosmos/cosmos-predict2.5`, release 2025-10-14, đã kiểm tra README) | mới | action map LIBERO OSC -> AC format, LoRA post-train trên collect_v1 + verif bridges (agent view, 256 px), decode -> DINOv2 -> C^pi v2. Có thể dùng thêm **Cosmos-Predict2.5/robot/policy (Libero)** như một policy / WM prior (chưa kiểm tra license / chi phí) | 4-6 ngày eng + LoRA khoảng 10-20 GPU-h (ước lượng, chưa đo) |

Cosmos là video diffusion (2B), chậm hơn latent WM nhiều bậc. Vì vậy nó **không** dùng để score cả pool 256 x CEM. Vai
trò dự kiến là **tier-2 re-scorer** trên top-k (k = 16-32) của tier-1 latent WM, hoặc làm WMVerifier cho mapping
real-data (B.6). WM-agnosticism được claim khi engine chạy end-to-end với >= 2 WM khác họ (latent DINOv2 và latent JEPA,
và video nếu N8 pass) và các lever của A.2 cho gain cùng dấu trên cả hai (claims policy hiện hành: ">= 2 WMs").

**WM card** (báo cáo chuẩn hoá cho mọi adapter, tính trên cùng candidates, 0 sim mới nhờ true end latents đã lưu):
- open-loop MSE / copy-frame tại h1 / h4 / h10;
- critic AUC trên real vs imagined;
- **optimism curve**: mean C(a), C(b), SR, FP, ECE, p@4 tại N = 8 / 32 / 128 / 512 (theo `scripts/diag_wm_true.py`,
  `analyze_wm_true.py`);
- WM share của gap;
- cost per rollout.

Đây vừa là artifact cho paper (phần measurement là phần còn defensible theo `WM_RECOVERY_OVERLAP.md` mục 3), vừa là unit
test cho mỗi adapter mới.

### A.2 Các lever để WM + search tìm recovery tốt hơn (tấn công trực tiếp optimism trên search-selected states)

Instance sets cố định:
- **R127**: RGR / M5b train shards 0-3, primary cho online;
- **R256**: toàn bộ 256 M5b train failures, dùng cho power;
- **U73**: single-unsolved, đánh giá hard failures. Lưu ý selection-on-failure: trên U phải dùng fresh `single`
  (rgr.seed_offset 1000, đang chạy trong ACR-N F2) làm control trung thực.

Budget chuẩn: **2800 env steps / instance** (= 8 x 350) cho verification, mọi execution đều tính phí. Imagination budget
được ghi lại và match giữa các arm (<= 768 WM bridge rollouts / instance, như RGR).

Metrics chính, dùng chung cho mọi lever:
- (m1) **solved rate** = tỉ lệ instance có >= 1 verified recovery robust (>= 2 / 3 continuation seeds);
- (m2) verified hits / 1k sim steps;
- (m3) FP của picks (pred >= 0.5 nhưng sim fail), gap a - b, ECE;
- (m4) solved trên U73;
- (m5) coverage = số unique failure episodes có data (bài học M9: failure-level coverage quyết định SFT).

Thống kê: exact McNemar (instance) + sign-flip (hits), cluster theo failure episode. Power xấp xỉ (alpha 0.10 two-sided,
power 0.8, tỉ lệ discordant khoảng 0.2): MDE khoảng +10 pt ở n = 127 và khoảng +7 pt ở n = 256. Dưới mức này kết luận là
"inconclusive at n", không phải "negative".

| # | Lever | Cơ chế + data | Prior work (verified) | Target đo được | Pre-registered decision rule (đề xuất; chốt ở intake) |
|---|---|---|---|---|---|
| **L1** | **DAgger cho WM trên verified negatives** (failure-aware, on-policy WM fine-tuning) | Mỗi round: mọi candidate đã verify (success **và** failure) cho ra một true transition (s_f, a, s_H^true). Đây chính là vùng action / state mà search khai thác. Fine-tune WM trên collect_v1 + các transition này (replay ratio 50 / 50 để không quên). **Data có sẵn, 0 sim**: verif_m5b (16,384 bridges, true end latents), true_end_rq3 (1,920 picks), verif_xwm, RGR picks | CureWM [2610.09134](https://arxiv.org/abs/2610.09134) (execution-verified counterfactual fine-tuning, passive), VLAW [2602.12063](https://arxiv.org/abs/2602.12063) (mixed success / failure rollouts loại bỏ over-optimism), WMPO [2511.09515](https://arxiv.org/abs/2511.09515), DAgger [1011.0686](https://arxiv.org/abs/1011.0686) | Offline: FP(a) tại N = 512 và gap a - b trên **held-out search-selected** candidates (train trên verif_m5b, test trên rq3 / xwm picks do search khác chọn; group folds theo failure episode). Online: m1 / m2 | Offline go nếu FP(a)@512 giảm >= 0.10 tuyệt đối **và** p@4 +0.03 (cluster-bootstrap 90 % CI > 0) mà AUC trên random candidates không giảm quá 0.02. Online go-to-distill nếu m2 +15 % (sign-flip p < 0.10) **và** m1 không thấp hơn (n01 >= n10) trên R127; "positive" chỉ khi m1 +7 pt, p < 0.10 trên R256 |
| **L2** | **Critic / recalibrator trên imagined states** (critic retraining trên search-selected states) | Train g(C^pi.embed(s_hat_H)) -> P(sim success) trên các cặp (imagined end latent, real outcome) của các verified candidates; cũng refit chính C^pi trên true end latents của picks (critic gap +0.05..+0.18). Cross-fit theo failure episode; dùng logistic / isotonic, không meta-learn (T2Mem: memory không giúp) | T2Mem lead (RESULTS); V-GPS [2410.13816](https://arxiv.org/abs/2410.13816) và RoboMonkey [2506.17811](https://arxiv.org/abs/2506.17811) (value bị khai thác khi N > 8-10) | hits r1-4 / 16 verifications, p@4, solved@16; held-out dir khác | Held-out go nếu gain >= +0.30 hits / 16 verif (cluster CI > 0) trên một verif dir mà search ở đó do **search khác** chọn; online như L1. Nếu L1 và L2 cùng pass thì test L1 + L2 (2x2) |
| **L3** | **Pessimistic / adversarial scoring + calibration theo search pressure** | (a) LCB trên **heterogeneous** ensemble (WM v1, V-JEPA, tier-2 Cosmos) = min (xwm) hoặc mean - k std; (b) ensemble thật bằng bootstrap data + seeds khác nhau (4 heads hiện tại identical, U = 0); (c) hiệu chỉnh "optimizer's curse" theo N: shrink score theo đường cong FP(N) của WM card | SAILOR [2506.05294](https://arxiv.org/abs/2506.05294) (mean - std critic ensemble); optimizer's curse (Smith & Winkler 2006, Management Science, **non-arXiv**) | FP, ECE, m1, m2 | Chỉ pre-register (a) min-arm ở n = 127 (follow-up đã nêu trong xwm). Go nếu m2 +12 % (p < 0.10) và m1 không giảm; lam 1 (calibration tốt nhưng solved giảm) không được chọn làm primary |
| **L4** | **Search trên scripted option library + WM** (ACR-N finding) | Proposal pool = SmolVLA samples + perturbations + primitives cũ + **options** (`grasp_slow`, `hover`, `grasp_firm`, `push_goal`, `place_goal`, ...) x variants, object pose lấy từ RGR geometry head (observation-only). WM + critic (+ L1 / L2) rank; diverse top-8 verify | RecoveryChaining [2410.13979](https://arxiv.org/abs/2410.13979) (learned option selection, sim-verified), DreamSteer 2607.02865 (primitives + samples), 2610.37196 (structured displacements tạo headroom) | m1 trên R127 **và** U73; control = constant `grasp_slow` x 8 ở cùng budget và fresh `single` | **Phụ thuộc ACR-N F2** (không chạy trước khi F2 có verdict). Go nếu WM + options > max(constant prior, fresh single) >= 5 pt trên U73 (p < 0.10) và >= single trên S. Nếu constant prior thắng: dùng "prior-first, WM-second" allocation (prior trials trước, WM bridges cho phần budget còn lại) |
| **L5** | **Learned proposal distribution** (amortized search, CEM warm-start) | Train một proposal head nhỏ (flow head 1-2M params hoặc LoRA trên action expert của pi0) trên các verified bridges của train split từ round <= k, conditioned trên obs tại failure; lấy K mẫu làm CEM init thay cho broad noise | ExIt [1705.08439](https://arxiv.org/abs/1705.08439), SAILOR (ExIt distill), M11 (planner from noise thua library) | **Coverage** = best-of-pool oracle success (sim-verify toàn bộ 64 candidates trên 60 instances: offline headroom như 2610.37196), m1, m2 | Go nếu best-of-pool coverage +5 pt so với library pool ở cùng pool size và m1 không thấp hơn. Lưu ý: pi_{k+1} cũng là một amortized proposal; L5 chỉ có giá trị nếu thắng "dùng pi_{k+1} samples làm proposals" |
| **L6** | **Budgeted verification allocation** (multi-fidelity + across-instance bandit) | (a) Multi-fidelity: verify bridge-only (H = 50 steps) -> C^pi trên **true** end state -> chỉ full continuation (300 steps) cho top. RQ3: ranking theo true end lift p@4 0.193 -> 0.237 với 1/7 chi phí mỗi screen. (b) Across instances: successive halving, ưu tiên instance có posterior solvable cao nhưng chưa có hit; dừng khi đã có 1 robust hit (coverage objective) | 2610.05151 (acceptance test bằng continuation), Kintsugi [2609.31048](https://arxiv.org/abs/2609.31048) (adaptive MC + Wilson) | **m1 per 1k sim (objective pre-registered = unique solved, không phải hits)**, m2 | T2Mem: reallocation tăng hits / verif +29-35 % nhưng solved -4 / -5 pt. Rule: chỉ go nếu m1 >= control **và** m2 +15 %; mọi arm báo cả hai |

Thứ tự ưu tiên: L1 / L2 offline trước (0 sim, rẻ nhất, tấn công trực tiếp nút thắt đã đo) -> L6a (rẻ, độc lập WM) ->
L4 (sau F2) -> L3 -> L5. **Oracle-first:** "ranking bằng C^pi trên TRUE end state" (đã có: p@4 0.237, hits 3.17) là upper
bound của L1 / L2. Lever nào không đóng được >= 50 % khoảng cách static -> oracle thì không được scale.

**Diagnosis bắt buộc nếu negative** (skill idea-loop mục 3):
- oracle vs method;
- budget split;
- coverage của pool;
- pred vs real của picks;
- continuation competence;
- power.

Nếu cả L1 và L2 thất bại offline: kết luận "imagined end states không sửa được bằng post-hoc training ở data scale này".
Khi đó engine chuyển sang **L6 + L4** (verification-centric, WM chỉ là cheap pre-filter), và paper framing đổi thành
"verification is the bottleneck".

### A.3 Iteration loop pi_k -> C^{pi_k} -> search -> verified data -> pi_{k+1}

```
round k (k = 1..K, K <= 3):
  1. failures_k  = rollouts của pi_k trên TRAIN scenes (protocol seeds mới, 5 close + 5 open), lấy failure snapshots
  2. C^{pi_k}    = refit trên continuations của pi_k (verified + failure states của round k, label = pi_k hoàn thành)
                   [bắt buộc: M6 dùng stale critic -> 0 gain]; báo AUC trên held-out failure episodes
  3. WM_k        = WM_{k-1}.finetune(verified transitions các round <= k)        [L1]
     cal_k       = refit recalibrator trên (imagined, outcome) các round <= k   [L2]
  4. search      = engine(WM_k, C^{pi_k}, cal_k, options, proposal_k) trên failures_k, cùng sim budget / instance
  5. verify      = bridge + pi_k continuation, robust >= 2/3 seeds
  6. pi_{k+1}    = distill (recipe chọn ở Phase 2) từ pi_k, data = aggregate các round (round-robin theo failure episode)
  7. guard       = nominal(pi_{k+1}) >= nominal(pi1-nominal) - 2 pt (mean 3 seeds); RSR trên PILOT (không phải test)
```

- **Control mỗi round**: pi_k + cùng số extra steps trên data cũ (M6 / M7 cho thấy extra steps tự nó cho +2-4 pt).
- **Stopping rule (pre-registered)**: dừng khi bất kỳ điều kiện nào xảy ra:
  - (i) pilot RSR(pi_{k+1}) - RSR(control_k) < 2 pt, hoặc CI chứa 0;
  - (ii) nominal guard fail;
  - (iii) verified yield trên failures của pi_k < 0.5 x yield round 1 ở cùng budget (failures cạn / quá khó);
  - (iv) K = 3.

  Chỉ round cuối cùng qua guard mới được eval trên test, một lần.
- Không claim "multi-round self-improvement" (claims policy) trừ khi round 2 thắng control của nó với 3 seeds.

---
## B. Dựng failure + recovery data từ data không phải LIBERO-Recover

### B.0 LIBERO-Recover thật sự đã làm gì (đọc từ arXiv HTML 2609.05178, 2026-11-17)

| Stage | Nội dung (trích / tóm tắt từ paper) |
|---|---|
| 1. Task execution | 4 LIBERO suites (130 subtasks), giữ instruction + init gốc. **6 policies làm "failure generators"** (pi0, pi0.5, OpenVLA, GR00T, Wan-Policy, Cosmos-Policy). *"No objects, environment states, or initial configurations are manually perturbed, and no failures are injected during execution."* |
| 2. Failure localization | Qwen3.5-27B xem video, chia V = V_pre ∪ V_fail ∪ V_post; *"retrieve the corresponding robot simulation states and object poses directly from the simulator"* tại các boundary. Assets lưu dạng `step_<t>.bddl` + `step_<t>_sim_state.npy` (flattened MuJoCo state), thường 2 timestamps cách nhau 20 steps trong cùng episode |
| 3. Failure characterization | VLM gán level L1-L4 + category (grasp_failure, placement_failure, object_displacement, collision_or_interaction_failure, environmental_obstruction, other) theo nguyên tắc "minimum corrective behavior"; các tác giả lấy mẫu và cross-verify thủ công |
| 4. Fine-tune & rollout | Fine-tune bằng teleop recovery (3,184 traj, 413 scenarios), deploy lại để **sinh thêm failures** (một vòng DAgger cho failure generator); các scenario đã dùng cho teleop bị loại khỏi eval |
| Eval protocol | 5 closed + 5 open gripper trials, ±2 cm xy noise trên free-joint objects (có bug off-by-one đã ghi trong LIBERO_RECOVER_BASELINES), 520 steps, success = BDDL `done` |

**Hiệu chỉnh so với yêu cầu ban đầu.** Recipe gốc **không** có object pose noise / grasp drop / push được inject vào lúc
tạo scene. Perturbation chỉ có ở eval protocol. Vì vậy B.1 (natural harvesting) là bản tái hiện trung thực, còn B.2
(injection) là phần mở rộng của mình. Mâu thuẫn cần kiểm tra: RESULTS ("Per-level") có ghi "upstream L4 are mostly
constructed", nhưng paper nói không inject gì. Mở 20 L4 scenes upstream (Goal / 10) xem `sim_state` có dấu hiệu dựng tay
không, trước khi dùng L4 làm tham chiếu (0.5 ngày, CPU).

**Assets có sẵn local** (không cần tải): `data/assets/scenes/scenes/selected_bddl_scenes_{spatial 168, object 77,
goal 91, 10 207, 90 337}` và `data/assets/scenes_full/assets_times` (state mỗi 20 steps; khoảng 109k bddl, gồm cả
Goal-type tasks). Official level annotations local chỉ có Spatial (269 files) + Object (1,139 files).

### B.1 Harvest natural failures (primary, trung thực với LIBERO-Recover)

- **Generators** (đa dạng failure mode, giống "6 policies"):
  - SmolVLA pi0 (`HuggingFaceVLA/smolvla_libero`, train trên 4 suites);
  - pi1-nominal, demo-only, pi1, M7 base;
  - 2-3 checkpoints trung gian (2k / 4k steps) của mỗi run;
  - 2-3 noise seeds của flow sampling;
  - tuỳ chọn: StarVLA `Qwen3-VL-PI / OFT-LIBERO-4in1` (cần conda env thứ hai, khoảng 1 ngày eng, đã khảo sát trong
    LIBERO_RECOVER_BASELINES B-2).

  Mỗi suite: 10 tasks x 50 init states x G generators x 2 seeds. Lưu flattened state mỗi 10-20 steps (cơ chế giống
  `scenes_full`).
- **Automatic failure detection** (không cần VLM vì sim cho privileged state; VLM chỉ để so taxonomy):
  - (a) outcome: `done` không fire trước T_max;
  - (b) **event predicates** trên privileged state, mỗi frame:
    - grasp lost (finger-object contact mất khi object z > 3 cm);
    - drop (object z rơi > 3 cm trong 10 steps);
    - knock / tip (up-axis < 0.8);
    - off-table;
    - target displaced > 3 cm khi chưa grasp;
    - non-target object / fixture state thay đổi (articulation joint, displacement > 3 cm);
    - BDDL sub-goal predicate regress (đã đạt rồi bị undo);
  - (c) **stall**: eef speed < 1 mm / step và không thay đổi contact trong 40 steps;
  - (d) **progress / competence drop**: C^pi_{pi0}(s_t) giảm > 0.3 so với max trước đó (chỉ dùng khi critic đã được
    train trên suite đó; nếu chưa có thì dùng (a) - (c)).
- **Localization**: t_fail = event đầu tiên trong (b) / (c), hoặc t của C^pi drop. Snapshot tại t_fail + {Δ1, Δ1 + 20}
  với Δ1 khoảng 20 steps (post-failure state, giống cặp step_0180 / 0200).
- **Level assignment**: rule-based từ privileged state, theo đúng định nghĩa của paper:
  - L1: s_f ≈ s_expected, tức target và support objects lệch < 2 cm so với pre-failure và upright;
  - L2: thay đổi giới hạn (2-5 cm, upright, vẫn graspable);
  - L3: target object state phải được restore (tipped, rơi khỏi support, lệch > 5 cm, nằm trong container sai);
  - L4: non-target object hoặc fixture trở thành vật cản (displaced vào workspace path, drawer / door state đổi).

  Thêm VLM label (Qwen3-VL, prompt của paper) trên một subset để so sánh. **Kiểm tra trước khi dùng**: agreement giữa
  rule-based và official labels trên Spatial + Object (1,408 annotations local), mục tiêu Cohen's kappa >= 0.6. Nếu thấp
  hơn thì dùng VLM labels làm primary và rule-based làm secondary.

### B.2 Perturbation injection (bổ sung, gắn nhãn `injected`, báo cáo riêng)

Dùng để tăng coverage các level hiếm (L3 / L4, vốn hiếm một cách tự nhiên trong paper; L4 không có trong splits của mình).
Injection xảy ra tại một **phase** được chọn của một nominal rollout: pre-grasp, grasp, transport, pre-place, xác định
bằng gripper / contact events.

| Injection | Thực hiện (MuJoCo) | Level kỳ vọng |
|---|---|---|
| object pose noise | xy ±2-8 cm, yaw ±30° trên target (sửa đúng qpos index, không lặp lại bug off-by-one) | L1 / L2 |
| grasp drop | ép mở gripper trong transport | L2 / L3 |
| push / knock | impulse ngang (0.5-2 x m g trong 20 steps) lên target hoặc non-target | L2 / L3 / L4 |
| tip over | xoay object để up-axis < 0.5 | L3 |
| obstruction | dời một non-target object vào đường tiếp cận / goal region | L4 |
| fixture change | đóng drawer đã mở / tắt stove / đóng cửa microwave | L4 (Goal / 10) |
| gripper / arm offset | đảo trạng thái gripper, lệch eef 3-5 cm (giống HELM 2604.18791 / RoboRecover 2609.28952) | L1 / L2 |

Sau injection, **base policy chạy tiếp 40-80 steps** rồi mới snapshot. Nhờ vậy state mang "hệ quả của policy" (gần
natural hơn), và vẫn qua cùng bộ detection + level. Mỗi injection type có quota, kèm quality filter như B.5. Main claims
dùng natural, còn injected là stress test / L4 coverage.

### B.3 Restore-from-snapshot

- Format giữ y hệt LIBERO-Recover (`.bddl` có `(:language)` + `_sim_state.npy`). Nhờ vậy `retryvla/envs/libero_env.py::load_scene`,
  official evaluator và `SimPool` đều dùng được ngay.
- Unit tests:
  - (i) bit-exact replay: restore rồi chạy lại 50 actions đã log, physics state identical;
  - (ii) render của frame đầu identical với frame đã log;
  - (iii) OSC goal re-anchoring + 20 settle steps như protocol (ghi rõ obs pre- / post-settle);
  - (iv) protocol noise: báo cả "official (có bug)" và "fixed" trên 1 suite để biết effect size.
- Snapshot pool của mỗi failure episode giữ toàn bộ timeline. Kintsugi-style "recoverability frontier" (B-3 baseline) chọn
  start state sớm hơn trên cùng timeline.

### B.4 Splits chống leakage

- Leakage unit = **source episode** = (suite, task, init-state index, generator, seed). Mọi snapshot của cùng episode nằm
  chung một split (như `spatial_v1`).
- **LIBERO-Goal**: 10 tasks dùng chung *một* scene layout, nên split theo (task, init-state index). Test = 91 official
  LIBERO-Recover Goal scenes. Train = harvested failures từ các init-state indices **không** xuất hiện trong episode ids
  của official test scenes; mapping episode_id -> init idx cần kiểm tra trước (0.5 ngày). Không dùng teleop expert data
  của test scenes.
- **LIBERO-10**: split theo scene (layout) để có held-out layout thật. LIBERO-90 chỉ dùng làm train / generator pool.
- `assert_allowed()` mở rộng cho split files mới (`data/splits/goal_v1.json`, `libero10_v1.json`), sha ghi trong mọi run
  summary.

### B.5 Quality checks (mỗi scene, ghi vào manifest)

1. **Feasibility**: oracle sim CEM (M1, <= 256 sims) hoặc scripted options + search giải được >= 1 lần, robust 2/3.
   Infeasible scenes vẫn được giữ trong manifest nhưng tách khỏi metric chính; báo cả hai.
2. **Not trivially solvable**: placebo re-run của pi0 (10 protocol trials) < 0.8. 2610.06921
   ([verified](https://arxiv.org/abs/2610.06921)) thấy 43 % failures tự thành công khi chạy lại.
3. **Physics sanity**: không penetration > 1 cm, objects trên bàn hoặc trong container hợp lệ, không có joint vượt limit.
4. **Diversity**: dedupe trong episode (giữ <= 2 snapshots cách nhau >= 20 steps), cluster theo (task, level, category,
  object displacement vector), cap theo cluster; báo histogram level / category so với Fig. 4 của paper.
5. **Distribution check vs official (Goal)**:
   - pi0 RSR trên scenes tự dựng vs official Goal scenes: kỳ vọng chênh <= 5 pt, báo cả hai;
   - level histogram;
   - rank correlation của 5 policies (pi0, pi1-nominal, demo-only, pi1, M7 base) giữa 2 tập: Spearman >= 0.8 thì xem
     là pipeline "valid".

### B.6 Mapping sang real data (user decision 2026-11-02: Ctrl-World-like WM làm verifier)

| Sim (hiện tại) | Real (sau này) |
|---|---|
| natural failures từ rollouts | failures trong deployment logs (của chính policy) + DROID failures (khoảng 19k) / RoboMIND (5k, có nguyên nhân); xem `reports/World model sinh dữ liệu robot.md` |
| event predicates trên privileged state | failure detector học được (progress / C^pi drop) + VLM localization (đúng Stage 2 của paper); FailBench 2609.03611 (balanced acc 0.77, chưa mở abs) cảnh báo độ chính xác |
| restore snapshot | **WM conditioning** trên frames thật tại t_fail (không restore được thật) |
| SimVerifier | **WMVerifier** = Ctrl-World-like video WM ([2510.10125](https://arxiv.org/abs/2510.10125) verified) fine-tune trên real data; verdict = "WM-verified", **không bao giờ gọi là sim-verified** |
| ground truth FP | anchor bằng một số ít real executions; sim study cho biết trước FP của một video WM khi bị search khai thác: đo bằng Cosmos tier-2 vs sim trong N8 |

Interface A.1 đã tách `Verifier`, nên engine không đổi. Chỉ verifier đổi. Hai tầng: latent WM rank -> video WM verify.

---
## C. Phase 2: distillation theo training recipe của foundation models

### C.0 Bối cảnh kỹ thuật của mình

- SmolVLA (450M; [2506.01844](https://arxiv.org/abs/2506.01844) verified), flow-matching action expert, chunk 50,
  `n_action_steps = 10`.
- Config mặc định của lerobot: `freeze_vision_encoder = True`, `train_expert_only = True`. Logs SFT ghi "trainable
  params: 97.5M", tức **chỉ action expert được train, VLM frozen**.
- Recipe hiện tại: init pi1-nominal, 6k steps, bs 32, lr 5e-5, 50 % nominal replay (r 0.5 là điểm tốt nhất của P3;
  r 0.7 fail eps).
- Đơn giá đo được (NOTES): SFT 6k khoảng 1.3 GPU-h trên A100; Spatial test RSR khoảng 1 h, Object khoảng 0.5 h, nominal
  khoảng 0.4 h. Một arm x 3 seeds x (SFT + RSR + nominal) **khoảng 8 GPU-h**.
- Hardware: local 8b889199 GPUs 1-3 (3 x A100 80 GB, 32 CPU cores). MuJoCo eval / search dùng CPU, nên eval song song bị
  giới hạn bởi số cores.

### C.1 Survey + đánh giá (mọi ID đã verify trên arXiv abs, 2026-11-17)

| Recipe | Paper | Cần gì | Lợi ích kỳ vọng cho recovery | Rủi ro nominal | Tương thích SmolVLA flow | Chi phí (3 seeds, Spatial) | Quyết định |
|---|---|---|---|---|---|---|---|
| **Co-training + nominal replay** (control hiện tại) | pi0 [2410.24164](https://arxiv.org/abs/2410.24164), pi0.5 [2504.16054](https://arxiv.org/abs/2504.16054), GR00T N1 [2503.14734](https://arxiv.org/abs/2503.14734) (co-train mixtures) | verified positives | đã có: 0.418 | thấp ở r 0.5 | native | 0 (đã có) | **control** |
| **Data mixing laws / mixture optimisation** | Data Mixing Laws [2403.16952](https://arxiv.org/abs/2403.16952), Re-Mix [2408.14037](https://arxiv.org/abs/2408.14037), Data Scaling Laws IL [2410.18647](https://arxiv.org/abs/2410.18647) | nhiều điểm mixture nhỏ | chọn r và tỉ lệ theo level / suite có cơ sở thay vì grid; dự báo được trade-off RSR / nominal | thấp | native | khoảng 5 mixtures x 1 seed x 2k steps proxy + fit khoảng 6 GPU-h | **làm** (rẻ, quan trọng khi thêm Goal / 10 data) |
| **Curriculum theo level** | (curriculum chung; không có paper VLA-recovery cụ thể đã verify) | level labels (B.1) | có thể giúp L3 | thấp | native | 1 arm khoảng 8 GPU-h | làm như ablation sau N4 |
| **Knowledge Insulation (pi0.5 KI)** | [2505.23705](https://arxiv.org/abs/2505.23705) | discrete FAST tokens cho VLM + stop-gradient từ action expert vào VLM | bảo vệ VLM khi **mở khóa** VLM | thấp (đó là mục đích) | cần FAST tokenizer + VLM loss head; SmolVLA không có sẵn | 3-5 ngày eng + khoảng 15 GPU-h | **chỉ làm nếu** N4 cho thấy unfreeze VLM cần thiết (expert-only plateau); hiện tại expert-only *đã* insulate VLM theo nghĩa mạnh nhất |
| **RECAP / advantage-conditioned policy (pi\*0.6)** | [2511.14759](https://arxiv.org/abs/2511.14759); CFG-style: [2505.23458](https://arxiv.org/abs/2505.23458) | value (C^pi đã có) + **positives và negatives** (đã có: 16k bridges, khoảng 83 % failures) + indicator token / embedding | dùng failures thay vì vứt bỏ; test time conditioning "good" + CFG weight; TACO / RISE cũng dùng advantage conditioning | trung bình (data negatives trên recovery states; nominal giữ indicator = good) | thêm 1 learned embedding vào action expert (expert-only vẫn ổn) | 2 ngày eng + khoảng 10 GPU-h | **làm** (ưu tiên 1: tận dụng đúng tài sản độc quyền là verified negatives) |
| **RFT / expert iteration / STaR / ReST** | ExIt [1705.08439](https://arxiv.org/abs/1705.08439), STaR [2203.14465](https://arxiv.org/abs/2203.14465), ReST [2308.08998](https://arxiv.org/abs/2308.08998), RFT [2308.01825](https://arxiv.org/abs/2308.01825); robot: Self-Improving EFMs [2509.15155](https://arxiv.org/abs/2509.15155), 2610.05151 | vòng A.3 + verifier | chính là pi_k -> pi_{k+1}; RFT cho thấy **số lượng reasoning paths khác nhau** quan trọng, tương ứng coverage theo failure ở mình | trung bình (drift qua rounds; SI-EFM: giảm khi train quá đỉnh) | native | mỗi round khoảng 8 GPU-h + search CPU | **làm** (= N5) |
| **Preference optimisation cho diffusion / flow** | Diffusion-DPO [2311.12908](https://arxiv.org/abs/2311.12908), DPO [2305.18290](https://arxiv.org/abs/2305.18290), GRAPE [2411.19309](https://arxiv.org/abs/2411.19309) (trajectory preference cho VLA) | **cặp (success, failure) từ cùng failure state**, có sẵn: cùng instance, verified vs failed bridges | dạy policy *tránh* recovery actions trông hợp lý nhưng fail (đúng loại lỗi WM khai thác) | trung bình-cao (DPO có thể đẩy khỏi data manifold); cần reference model + beta | flow-DPO: thay log-likelihood bằng ELBO-style flow-matching loss difference vs reference (như Diffusion-DPO); thêm 1 forward reference | 3 ngày eng + khoảng 10 GPU-h | **làm** (ưu tiên 2) |
| **AWR / weighted flow matching** | AWR [1910.00177](https://arxiv.org/abs/1910.00177); `retryvla/train/awr.py` (skeleton có sẵn); VLAW / WEAVER weighted FM | advantage per trajectory | rẻ hơn RECAP; giữ positives, re-weight theo robustness (3/3 vs 2/3 seeds) | thấp | native (per-sample weight) | khoảng 8 GPU-h | làm như ablation rẻ của RECAP |
| **RL fine-tuning flow policies** | ReinFlow [2505.22094](https://arxiv.org/abs/2505.22094), Flow-GRPO [2505.05470](https://arxiv.org/abs/2505.05470), pi_RL [2510.25889](https://arxiv.org/abs/2510.25889), FPO [2507.21053](https://arxiv.org/abs/2507.21053), DPPO [2409.00588](https://arxiv.org/abs/2409.00588), VLA-RL [2505.18719](https://arxiv.org/abs/2505.18719), SimpleVLA-RL [2509.09674](https://arxiv.org/abs/2509.09674), RIPT-VLA [2505.17016](https://arxiv.org/abs/2505.17016), RLinf [2509.15965](https://arxiv.org/abs/2509.15965) | online sim interaction từ **failure snapshots** (reset = restore), sparse BDDL reward | có thể vượt imitation ceiling; SimpleVLA-RL / pi_RL báo LIBERO gain lớn | **cao**: B3.3 residual PPO của mình: RSR 0.295, nominal 0.022; SimpleVLA-RL ghi "pushcut" predicate hacking | pi_RL / ReinFlow là flow-native nhưng cần stochastic flow (noise injection) | 1-2 tuần eng; hàng chục nghìn episodes, CPU-bound (khoảng 1 ngày sim / 100k episodes ở 16 workers, ước lượng) | **làm bản rẻ nhất trước**: DSRL (dưới) + RIPT-style (dynamic sampling, LOOCV advantage, KL / nominal replay). Full pi_RL chỉ khi DSRL cho thấy RL headroom |
| **DSRL: RL trong noise space của diffusion / flow** | [2506.15799](https://arxiv.org/abs/2506.15799) | train một policy nhỏ chọn **initial noise** của flow sampler; base policy frozen | không đụng weights nên nominal an toàn tuyệt đối khi noise policy chỉ bật ở recovery states; rất sample-efficient | rất thấp | tương thích trực tiếp (SmolVLA nhận noise) | 3 ngày eng + CPU sim | **làm** (RL arm chính; cũng là cầu nối tự nhiên với L5 learned proposal) |
| **PA-RL / RLDG / PLD** | PA-RL [2412.06685](https://arxiv.org/abs/2412.06685), RLDG [2412.09858](https://arxiv.org/abs/2412.09858), PLD [2511.00091](https://arxiv.org/abs/2511.00091) | RL specialist rồi distill vào generalist bằng SFT | đây là "search hoặc RL làm teacher, SFT làm student" giống pipeline của mình | thấp (SFT) | native | đã có tương đương (B3.3 + SFT) | chỉ cite; framing |
| **DAgger multi-round** | [1011.0686](https://arxiv.org/abs/1011.0686), WM-DAgger [2604.11351](https://arxiv.org/abs/2604.11351) | failures của pi_k làm states mới (on-policy) | đúng A.3; data aggregation chống covariate shift | thấp-trung bình | native | = N5 | **làm** (= A.3) |
| **LoRA vs full / expert-only FT** | LoRA [2106.09685](https://arxiv.org/abs/2106.09685) | - | LoRA giới hạn drift (nominal); có thể kém về capacity cho behaviour mới | LoRA thấp | PEFT trên expert / VLM | 2 arms khoảng 16 GPU-h | làm như ablation: expert-only (hiện tại) vs LoRA r 32 trên expert + VLM vs unfreeze VLM (với / không KI) |
| **KL / behaviour regularisation + weight interpolation** | WiSE-FT [2109.01903](https://arxiv.org/abs/2109.01903); L_preserve hiện có | reference policy | WiSE-FT: theta = (1 - a) theta_0 + a theta_ft, chọn a trên pilot; **0 extra training**, dịch chuyển Pareto RSR / nominal | giảm rủi ro | native | chỉ eval (khoảng 1.5 GPU-h / alpha) | **làm** (guard rẻ nhất; dùng cho mọi arm không qua eps) |
| ConRFT | [2502.05450](https://arxiv.org/abs/2502.05450) | real-world, consistency policy | - | - | đổi head | - | không làm (real-world, đổi head) |

Cách đọc chung:
- Tài sản hiếm của mình là **verified negatives có ground truth, paired theo failure state**. Ít recipe foundation model
  nào có được thứ này: pi\*0.6 có negatives từ deployment, GRAPE có preference pairs.
- Vì vậy hai recipe ưu tiên là **RECAP-style advantage conditioning** và **flow-DPO**. Cả hai đều offline, rẻ, và tương
  thích expert-only FT.
- RL online chỉ đi qua DSRL trước, vì đã có bằng chứng nội bộ rằng RL làm sập nominal (B3.3).

### C.2 Thiết kế Phase 2 shoot-out (N4)

- **Data bank cố định** cho mọi arm. Nguồn: verified positives của engine tốt nhất từ N2, hoặc nếu chưa có thì bank hiện
  tại của pi1 (208 M5c) cộng toàn bộ verified negatives của cùng instances. Cùng init pi1-nominal, cùng 6k steps, cùng
  50 % nominal replay, cùng eval seeds.
- **Arms**:
  - D0 SFT (pi1 recipe);
  - D1 AWR-weighted;
  - D2 RECAP (indicator; test-time CFG w ∈ {1, 1.5, 2} chọn trên pilot);
  - D3 flow-DPO (beta ∈ {0.1, 1} chọn trên pilot) + SFT anchor;
  - D4 DSRL noise policy trên pi1 (0 weight change, budget sim khớp với một vòng search);
  - D5 WiSE-FT interpolation của arm tốt nhất nếu nó fail eps.
- **Metrics**:
  - Spatial test RSR (dev), per level L2 / L3;
  - nominal (eps 2 pt vs pi1-nominal 0.796, mean 3 seeds);
  - McNemar vs D0;
  - scene x seed bootstrap;
  - cost (GPU-h, sim steps; D4 tính cả sim online).
- **Decision rule**:
  - recipe R thay D0 làm default chỉ nếu dRSR >= +3 pt với CI 95 % > 0 (3 seeds), nominal pass, và không tụt > 3 pt ở
    level nào;
  - nếu không arm nào qua: giữ SFT và báo "verified negatives không thêm giá trị ở data scale này". Đây vẫn là một
    finding có ích, đối chiếu với pi\*0.6 / GRAPE.
- Ablation phụ sau khi chọn R:
  - expert-only vs LoRA-all vs unfreeze VLM (+ KI);
  - curriculum theo level;
  - mixing law fit cho r khi gộp Spatial + Goal data.

---
## D. Experiment + baseline matrix

### D.1 Search level (Phase 1; train split; R127 primary, R256 power, U73 hard)

| Arm | Mô tả | Vai trò |
|---|---|---|
| S0 `single` (fresh seeds) | wm_unc_l0 pipeline hiện tại, WM v1 + C^pi v2 | control |
| S1 no-WM sim-only search | random / library proposals + continuation acceptance, **cùng sim budget 2800 steps** (= kiểu 2610.05151) | bắt buộc: "WM có cần không?" |
| S2 oracle ranking | C^pi trên TRUE end state (bridge-only sim screen, chi phí tính riêng) | upper bound của L1 / L2 / L6a |
| S3 constant prior | `grasp_slow` x 8 (ACR-N) | control cho L4 |
| L1, L2, L1 + L2 | A.2 | method |
| L3 (min x-WM), L4 (options + WM), L5 (learned proposal), L6 (multi-fidelity + allocation) | A.2 | method |
| S4 SAILOR-style scorer | discriminator reward "expert-like vs learner state" (train trên demos vs pi0 rollouts, latent WM v1) + critic ensemble mean - std, cùng CEM / budget | baseline bắt buộc (2506.05294) |
| S5 Back-on-Track-style scorer | discounted liveness value (2605.11479 [verified](https://arxiv.org/abs/2605.11479)) train trên cùng continuations, thay C^pi | baseline critic (WM_RECOVERY_OVERLAP mục 5.4) |
| mỗi arm x {WM v1, V-JEPA} (+ Cosmos tier-2 nếu N8) | | WM-agnosticism |

### D.2 Policy level (Phase 1 data x Phase 2 recipe; test RSR, 3 seeds)

| Arm | Data / phương pháp | Test-time module | Vai trò |
|---|---|---|---|
| pi0, pi1-nominal, demo-only | - | - | anchors (đã có) |
| pi1 (D0) | verified bridges, SFT | none | control |
| **Engine data + best recipe R** | N2 engine bank + N4 recipe | none | method chính |
| **Back-on-Track-style test-time planning** | pi0 và pi1, monitor (C^pi < tau) -> CEM trong WM -> hand back; không distill | WM + critic + CEM khi chạy | bắt buộc: "plan vs distill vs both"; báo latency / inference cost |
| SAILOR-style distill | S4 picks, ExIt distill **không verify** + phiên bản có verify | none | tách đóng góp verification |
| Dream2Fix / WM-DAgger-style unverified WM data | WM-verdict-labelled bridges (top C^pi, không sim check) ở cùng count | none | "unverified WM data" (B3.2 / arm 3 gần nhưng không phải action-labelled) |
| no-WM sim-only data | S1 bank, cùng sim budget | none | 2610.05151-style |
| Human teleop "Combined" | `ataier/LIBERO_Recovery_Expert` Spatial subset (khoảng 1,185 ep) + count-matched | none | B-1 trong LIBERO_RECOVER_BASELINES |
| Kintsugi frontier | start states theo recoverability frontier + library search | none | B-3 |
| Backtrack-distill, scripted-regrasp distill | đã có / ACR-N F2 | none | no-WM recovery baselines |

**Metrics**:
- RSR overall + per level (official labels; L4 chỉ trên suite có L4, như Goal / 10);
- nominal (<= 2 pt drop, mean 3 seeds);
- sim cost (env steps cho data, kể cả WM / critic data collection, phân bổ đều);
- train GPU-h;
- inference cost (ms / step, có module test-time hay không);
- McNemar + scene x seed bootstrap.

**Held-out generalization** (recipe đóng băng trước khi nhìn):
1. **Object test** (26 scenes; đã dùng một lần cho M10, nên dùng lại với recipe mới đóng băng; báo rõ là lần thứ hai);
2. **official LIBERO-Recover Goal scenes** (91; train data chỉ từ B.1 harvest trên Goal train init states, nên đây là
   test "dữ liệu tự dựng -> benchmark chính thức");
3. **LIBERO-10-R tự dựng** (B.1 + B.2, split theo layout; test chỉ eval một lần). Official LIBERO-Recover-10 scenes
   (207) dùng làm external check thứ hai.

---

## E. Milestones, effort, go / no-go gates

Ưu tiên compute: không chiếm 8b889199 GPUs 1-3 cho tới khi ACR-N F2 xong. Các việc 0-GPU (N0, N1 offline, phần eng của
N3) bắt đầu ngay.

| M | Nội dung | Effort (eng / compute) | Gate (pre-register ở PLAN.md trước khi chạy) |
|---|---|---|---|
| **N0** | Interface A.1 + adapters WM v1 / V-JEPA + WM card (offline, verif_m5b) + equivalence tests | 3 ngày / khoảng 2 GPU-h | equivalence max diff <= 1e-6; WM card tái tạo số RQ3 (gap 0.46, FP N curve) ± 0.01 |
| **N1** | Offline levers L1, L2 (+ L6a mô phỏng trên true end latents đã lưu): train trên verif_m5b, test trên rq3 / xwm / RGR picks | 4 ngày / khoảng 10 GPU-h, 0 sim | **G1**: ít nhất một trong L1 / L2 đạt rule A.2 offline **và** đóng >= 50 % gap static -> oracle. Fail cả hai -> diagnosis + review (skill), pivot sang L6 + L4 verification-centric |
| **N2** | Online paired search trên R127 (rồi R256 cho arm thắng): S0, S1, S2, S3, best của L1 / L2, L6, (L4 sau F2), L3; x 2 WMs | 2 ngày eng / khoảng 3-4 ngày sim (32 cores; khoảng 4 min / instance / 6 arms đo ở RGR) + khoảng 10 GPU-h | **G2**: engine arm có m1 >= S0 + 7 pt (R256, McNemar p < 0.10) **hoặc** m2 +15 % với m1 không thấp hơn; dấu cùng chiều trên cả 2 WMs (WM-agnostic claim). Và phải thắng S1 (no-WM) ở cùng sim, nếu không thì claim chuyển sang "verification-centric, WM optional" |
| **N3** | Suite builder B: Goal (harvest 5 generators x 2 seeds x 500 inits; detection; levels; QC), check vs 91 official Goal scenes; rồi LIBERO-10 | 6-8 ngày eng / khoảng 2 ngày sim CPU + khoảng 4 GPU-h policy inference | **G3**: rule-vs-official kappa >= 0.6 (Spatial / Object); pi0 RSR tự dựng vs official chênh <= 5 pt; Spearman (5 policies) >= 0.8; feasibility >= 70 % sau QC. Fail -> dùng VLM labels / sửa detection trước khi sinh train data |
| **N4** | Phase 2 shoot-out C.2 (D0-D5), 1 seed screen -> 3 seeds cho top-2 | 7 ngày eng (RECAP, flow-DPO, DSRL) / khoảng 45 GPU-h | **G4**: rule C.2. Chọn recipe R (hoặc giữ SFT) |
| **N5** | Iteration loop A.3, K <= 3, engine + R, critic refit mỗi round | 2 ngày eng / khoảng 2 ngày sim + khoảng 25 GPU-h / round | **G5**: stopping rule A.3; round k chỉ giữ nếu thắng control "+ extra steps" của nó |
| **N6** | Baselines bắt buộc D.2: Back-on-Track test-time arm (pi0 / pi1), SAILOR-style (± verify), unverified WM data, no-WM sim, teleop Combined, Kintsugi | 6 ngày eng / khoảng 60 GPU-h | báo cáo, không gate; nhưng nếu Back-on-Track-style pi1 + planner > distilled policy ở cùng inference budget thì đổi framing thành "both" |
| **N7** | Frozen held-out eval: Object test, official Goal (91), LIBERO-10-R test, + nominal 3 suites | 1 ngày / khoảng 25 GPU-h | claim generalization chỉ khi engine + R > pi1-recipe (cũng train trên data suite đó) trên >= 2 / 3 held-out sets, nominal pass |
| **N8** (có điều kiện: G2 pass) | Cosmos-Predict2.5 AC adapter làm tier-2 + đo FP của video-WM verifier vs sim (chuẩn bị cho real) | 5 ngày eng / khoảng 30 GPU-h | video WM card; tier-2 giảm FP >= 0.1 ở cùng sim budget thì giữ |

Tổng: khoảng 6-8 tuần eng (một người + agent), khoảng 200 GPU-h, khoảng 10 ngày CPU sim (song song được với GPU jobs).
Review đối kháng (`retryvla-idea-review`) chạy sau G1, G2, G4 và trước N7.

### E.1 Rủi ro và giảm thiểu

| Rủi ro | Xác suất | Giảm thiểu |
|---|---|---|
| L1 / L2 overfit vào search-selected distribution của một search, không transfer sang search mới | trung bình-cao | test held-out trên verif dir do search *khác* chọn; online test bắt buộc; prequential evaluation |
| Search khai thác WM đã fine-tune theo hướng mới (moving target) | cao | chính là lý do loop có nhiều round; đo FP mỗi round; L3 LCB làm guard |
| Constant prior / no-WM search thắng engine (WM không cần) | trung bình | S1, S3 là arms chính thức. Nếu thua, framing = "verification + library + learned selection", WM chỉ là pre-filter. Vẫn publish được measurement |
| Distill gain không theo search gain (đã thấy: B4 / B2) | cao | coverage (m5) là metric chính; Phase 2 dùng negatives; báo cả hai tầng |
| RL / DPO làm sập nominal | trung bình (RL cao) | DSRL (không đổi weights), WiSE-FT, nominal replay, eps gate |
| Pipeline B sinh distribution khác official | trung bình | G3 với official Goal scenes; báo cả natural vs injected |
| Leakage init-state giữa Goal train và official Goal test | trung bình | mapping episode -> init idx trước N3; split theo init idx |
| Compute bị chiếm (ACR-N, chủ shell) | cao | N0 / N1 / N3-eng là 0-GPU; shells probe-first theo `retryvla-multi-gpu-shells` |
| Power thấp (n 73-127) | cao | R256 cho primary; Goal / 10 tăng n; ghi "inconclusive at n" |
| Cosmos quá đắt hoặc kém trên LIBERO (domain gap như V-JEPA zero-shot 2.0x copy-frame) | trung bình | N8 có điều kiện; LoRA post-train; chỉ tier-2 |

### E.2 Novelty sau giai đoạn này vs Back on Track / SAILOR

| Khía cạnh | Back on Track (2610.19579) | SAILOR (2506.05294) | RetryVLA sau stage này |
|---|---|---|---|
| Khi nào WM được dùng | test time, không verify | test time MPC + ExIt distill | **offline data engine**, mọi label sim-verified; deployed policy không có module |
| Xử lý WM exploitation | nêu là limitation | WM / RM refit online; "robust to hacking at 8x" | **đo** optimism theo search pressure (WM card) **và sửa** bằng DAgger trên verified negatives của chính search + recalibration; báo FP / gap mỗi round, cho >= 2 WMs |
| Value | liveness value (policy-conditioned) | discriminator reward + critic ensemble | C^{pi_k} refit mỗi round; S5 so sánh trực tiếp với liveness value |
| Dùng failures | không | không trực tiếp | verified negatives cho WM, critic **và** policy (RECAP / flow-DPO) |
| Benchmark / data | perturbed LIBERO-Goal init | robomimic / robosuite / ManiSkill | LIBERO-Recover + **suite builder** tái tạo recipe LIBERO-Recover cho suite bất kỳ, kiểm tra trên official Goal scenes |
| Nominal | không đổi policy | không báo constraint | eps 2 pt pre-registered, 3 seeds |

Phần overlap cần nói thẳng:
- CureWM / VLAW đã fine-tune WM trên failures (passive);
- 2610.05151 đã dùng continuation acceptance;
- pi\*0.6 đã dùng advantage conditioning.

Phần mới của mình là **sự kết hợp và việc đo**: sửa optimism *dưới search pressure* bằng chính các candidate bị search
khai thác, rồi amortise vào policy bằng các recipe dùng được negatives, kèm WM-agnostic evidence và một suite builder.
Phải chạy targeted search (2025-11 -> 2026-11) về "world model fine-tuning on planner-selected / adversarial candidates"
trong intake của L1 trước khi claim.

---

## F. Câu hỏi mở cho user

1. **Held-out suite mới**: Goal (có official scenes để validate, khuyến nghị) và LIBERO-10 (dài, có L4), hay ưu tiên
   LIBERO-Plus / PRO (robustness perturbations, chưa có recovery scenes, cần env code riêng)?
2. **Injected data**: chấp nhận dùng `injected` scenes trong *training* data (main claims vẫn trên natural test), hay chỉ
   dùng như stress test?
3. **Failure generators**: có dựng conda env StarVLA để dùng thêm Qwen3-VL-PI / OFT làm generators không (đa dạng failure
   hơn, khoảng 1 ngày eng + khoảng 4 GPU-h / suite)?
4. **Cosmos-Predict2.5 (N8)**: có muốn đầu tư (khoảng 5 ngày + khoảng 30 GPU-h) ngay giai đoạn này để chuẩn bị cho
   real-data verifier, hay để sau khi có real data?
5. **Thứ tự Phase 1 vs Phase 2**: N4 có thể chạy song song N2 trên bank hiện tại (pi1 bank + negatives). Có chấp nhận như
   vậy không (nhanh hơn khoảng 1.5 tuần, nhưng recipe chọn trên data cũ)?
6. **RL arm**: DSRL là đủ, hay cần pi_RL / flow-GRPO đầy đủ (1-2 tuần eng, rủi ro nominal) để trả lời reviewer "sao không
   RL"?
7. **Compute**: ngoài 8b889199 GPUs 1-3, có được mượn shells khác (probe-first) cho N2 / N3 sim không?
