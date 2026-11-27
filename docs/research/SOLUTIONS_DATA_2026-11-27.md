# Giải pháp phía dữ liệu cho Phase-1 recovery data: learnable + effective (2026-11-27)

Research task (literature + phân tích CPU trên dữ liệu có sẵn; không train GPU, không sim mới). Viết muộn so với deadline
02:00 UTC (CPU job xong lúc ~03:00 UTC). Code: `scripts/data_research/fm_loss_m0.py` (M0 FM loss từng anchor, CPU fp32,
prefix KV cache; selfcheck khớp training forward tới 1e-5), `scripts/data_research/analyze_fm_m0.py`.
Output: `runs/_analysis/data_research/fm_m0/{anchors.jsonl, summary.md, summary.json}`.

## 0. TL;DR

- **Đo trực tiếp (767 anchor, M0 = `m0_s7/ckpt_04000`, init của mọi arm Goal):** per-anchor flow-matching loss của
  demo D0 = 0.132. Bridge của bank onset (G2) = 1.21 (**9.2x**), bridge bank cũ (F0/F1) = 2.57 (**19.5x**), LRS v1 = 2.38
  (18x), LRS v2 = 1.35 (10x), LRS v2-GT = 1.06 (8x). Continuation của M0 = 0.62-0.69 (**~5x**, M0 tự sinh mà vẫn 5x vì
  state off-distribution). Survivor của phys filter: 0.43 (strict, 3.2x) / 0.49 (noseam) so với reject 1.34 / 1.88
  (MWU p < 0.01): **phys filter đang ngầm làm FM-loss filter**.
- **Theo family (heuristic trên action bridge):** pi0-smooth 0.66 (5x), primitive 0.97 (7.4x), prim+CEM 1.86 (14x),
  pi0-rough 4.8 (36x), perturbation-like 14.6 (111x, n = 8). Primitive mượt nhưng **không** on-manifold; perturbation là
  nguồn gradient độc nhất.
- **Tương quan với harm (18 arm Goal, Spearman với Δ nominal):** mean recovery loss ρ = −0.92 (p 4e-8; −0.92 khi bỏ LRS),
  loss-weighted gradient share ρ = −0.40 (−0.59 bỏ LRS, p 0.02), **anchor share một mình ρ = −0.05**. Tức là
  "bao nhiêu % anchor" không giải thích harm; "anchor đó khó học cỡ nào với M0" giải thích. Ví dụ rõ nhất: F1 (chỉ bridge,
  4.3 % anchor, 18.6x loss) −20 pt; LRS v2 (4.1 %, 10.6x) −41; G2-phys (1.6 %, 3.6x) +3.
- **Per-task: không có tương quan** (arm-demeaned ρ = −0.04, n = 180 arm x task); task không có recovery data trong arm
  vẫn giảm TB −1.1 pt. Harm là **global** (đẩy cả action head), không phải cục bộ theo task: khớp với LRS v1 làm sập cả
  `turn_on_stove`.
- **Literature (mục 2):** mọi pipeline thắng đều làm dữ liệu "gần base policy": Agentic-Teacher lọc theo FM loss của base
  đóng băng (+6 pt trong ablation, 120 -> 110 traj) và thêm correlated noise; PLD cho base policy "probe" trước rồi mới
  residual takeover (α = 0.6 tối ưu), lý giải bằng RL's Razor (forgetting ∝ KL tới base); RLDG cho thấy action quality
  quan trọng hơn state distribution; critic guidance cho flow VLA (GAF, QPILOTS, AGF) cho +4 pt trên LIBERO-Goal SmolVLA
  ở test time.
- **Top-3 đề xuất Phase-1 (mục 4):** (P1) FM-loss / gradient-budget filter trên bank đã có (0 sim, ~17 GPU-h);
  (P2) re-track bridge đã verify bằng chính M0, giữ cái M0 tái tạo được (~19 GPU-h);
  (P3) thay perturbation/CEM bằng guided pi0 sampling tại failure state với value guidance + sim verify (~26 GPU-h).

## 1. Phân tích CPU trên dữ liệu của mình

### 1.1 Phương pháp
- Loss = pi0 FM loss (mean 50 target x 7 dim), K = 2 draw (noise, t) seed cố định dùng chung cho mọi nguồn (paired).
  Bank của mình: unmasked (như lúc train); LRS: masked theo valid target + L_min 5 (như lúc train).
- Lấy mẫu: D0 64 anchor ngẫu nhiên; onset bank (G2) 1 anchor bridge + 1 continuation / trajectory (132 wm + 115 random);
  bank cũ (F0/F1) 20 trajectory / shard; LRS v1 50, v2 40, v2-GT 13. Tốc độ ~15 s / anchor trên node CPU chia sẻ.
- Arm-level: loss của arm = trung bình có trọng số độ dài segment trên **pool đủ điều kiện** (keep list áp dụng; round-robin
  selection không replay) -> các arm cùng pool (G2_wm@5/@18/@36, P2-3 A) có cùng L_rec, đó là giới hạn. Gradient share
  = n_rec L_rec / (n_rec L_rec + n_D0 L_D0) (xấp xỉ bậc nhất: gradient norm tỉ lệ với loss).

### 1.2 Theo nguồn / segment

| source | segment | n | M0 FM loss [95 % CI] | x D0 | chunk d3 | grip flips / chunk |
|---|---|---|---|---|---|---|
| D0 demo | - | 64 | 0.132 [0.112, 0.155] | 1.0x | 0.069 | 0.62 |
| onset bank wm (G2) | bridge | 132 | 1.214 [1.007, 1.456] | 9.2x | 0.237 | 0.92 |
| onset bank wm (G2) | M0 continuation | 98 | 0.653 [0.570, 0.734] | 5.0x | 0.191 | 1.42 |
| onset bank random | bridge | 115 | 0.729 [0.635, 0.814] | 5.5x | 0.201 | 0.88 |
| onset bank random | M0 continuation | 90 | 0.688 [0.613, 0.765] | 5.2x | 0.224 | 1.33 |
| old bank wm (F0/F1) | bridge | 60 | 2.570 [1.221, 4.545] | 19.5x | 0.270 | 1.38 |
| old bank wm (F0/F1) | M0 continuation | 45 | 0.621 [0.527, 0.718] | 4.7x | 0.232 | 1.36 |
| old bank random | bridge | 60 | 1.869 [0.917, 3.108] | 14.2x | 0.253 | 0.75 |
| LRS v1 (bang-bang servo) | window | 50 | 2.384 [2.005, 2.799] | 18.1x | 0.474 | 0.24 |
| LRS v2 (demo servo) | window | 40 | 1.353 [1.202, 1.509] | 10.3x | 0.130 | 0.45 |
| LRS v2-GT (sim-verified) | window | 13 | 1.057 [0.709, 1.400] | 8.0x | 0.096 | 0.23 |

Đọc:
- WM-selected bridge (1.21) khó học hơn random bridge (0.73) trên cùng entry: search chọn đúng những candidate lạ
  (winner's curse ở WM_OPTIMISM_VISUAL cũng là "off-manifold curse" ở đây).
- LRS v2 mượt (d3 0.13, gần demo) mà vẫn 10x: **mượt không đủ**; cái làm khó là chunk không dự đoán được từ ảnh
  (stop / turn / gripper toggle). Smoothness và learnability là hai trục khác nhau.
- Continuation của M0 cũng 5x: state recovery nằm ngoài D0, nên ngay cả action của chính M0 cũng không "dễ" với M0
  (M0 stochastic; sample của nó ở state lạ có entropy cao). Đây là sàn cho mọi dữ liệu recovery, kể cả on-policy.

### 1.3 Theo family (bridge, heuristic giống `scripts/wm_optimism/cache_stats.action_family`)

| family | n | M0 FM loss | x D0 |
|---|---|---|---|
| pi0 sample, smooth | 247 | 0.664 [0.590, 0.752] | 5.0x |
| primitive (rot 0, gripper ±1) | 52 | 0.968 [0.833, 1.097] | 7.4x |
| primitive + CEM | 37 | 1.861 [1.546, 2.220] | 14.1x |
| pi0 sample, rough | 23 | 4.786 [3.102, 7.390] | 36.4x |
| perturbation-like | 8 | 14.63 [6.53, 25.72] | 111x |

- Thứ tự learnability: pi0-smooth > primitive > prim+CEM >> pi0-rough >> perturbation. Primitive bị WM khai thác ít nhất
  nhưng **không** gần manifold của M0 (rotation đúng 0, gripper ±1 tuyệt đối, vận tốc khác demo): cần demo-matched
  servo / re-track, không dùng nguyên.
- 8 bridge perturbation-like có loss TB 14.6: nếu chúng nằm trong mix 36 traj thì chúng một mình mang phần lớn gradient
  recovery. Heuristic family có sai số (calibrate trên Spatial).

### 1.4 Phys filter = FM-loss filter ngầm

| filter (onset wm bank) | segment | keep n / loss | reject n / loss |
|---|---|---|---|
| strict phys | bridge | 18 / 0.426 | 114 / 1.338 (MWU p < 0.01) |
| phys no-seam | bridge | 63 / 0.488 | 69 / 1.876 (p < 0.01) |
| phys no-seam | continuation | 46 / 0.628 | 52 / 0.675 (p 0.43) |

Lý giải nhất quán cho G4: arm "phys" thắng không hẳn vì vật lý hợp lý mà vì dữ liệu dễ học cho M0 (3.2-3.7x thay vì
9x); "G2-phys gain là share effect" (G4) và "là loss effect" không loại trừ nhau: cả hai đều hạ gradient share.

### 1.5 Arm-level: loss giải thích harm, share thì không

| arm | rec anchors | anchor share | M0 loss rec (x D0) | gradient share | Δ nominal |
|---|---|---|---|---|---|
| F0_wm (all 282) | 34,078 | 47.1 % | 1.32 (10.1x) | 90 % | −11.3 |
| F0_random (all 282) | 34,554 | 47.5 % | 1.83 (13.9x) | 93 % | −11.7 |
| F1_wm (bridge only) | 1,718 | 4.3 % | 2.44 (18.6x) | 46 % | −20.3 |
| F1_random (bridge only) | 1,650 | 4.1 % | 1.77 (13.5x) | 37 % | −21.3 |
| D5 (old bank 36) | 4,203 | 9.9 % | 1.32 (10.1x) | 53 % | −5.0 |
| G2_wm@36 | 4,864 | 11.3 % | 0.87 (6.6x) | 46 % | −6.7 |
| G2_wm@18 | 2,535 | 6.2 % | 0.87 (6.6x) | 30 % | −2.7 |
| G2_wm@5 | 973 | 2.5 % | 0.87 (6.6x) | 14 % | +2.3 |
| G2_random@36 | 4,455 | 10.4 % | 0.72 (5.5x) | 39 % | −0.7 |
| G2-phys (18) | 632 | 1.6 % | 0.48 (3.6x) | 5.6 % | +3.0 (5 seeds) |
| G2-physns (36) | 4,346 | 10.2 % | 0.56 (4.2x) | 32 % | +2.3 |
| strict-phys (46) | 1,917 | 4.8 % | 0.55 (4.2x) | 17 % | +3.0 (5 seeds) |
| P2-3 A (72) | 9,508 | 19.9 % | 0.87 (6.6x) | 62 % | +0.0 |
| G3-rep (72, s10-12) | 9,181 | 19.4 % | 0.87 (6.6x) | 61 % | −4.8 |
| G3-c20 (cont ≤ 20) | 4,366 | 10.2 % | 1.05 (8.0x) | 48 % | −9.0 |
| LRS v1 | 1,708 | 4.3 % | 2.15 (16.3x) | 42 % | −65.0 |
| LRS v2 | 1,645 | 4.1 % | 1.39 (10.6x) | 31 % | −40.7 |
| LRS v2-GT | 655 | 1.7 % | 0.88 (6.7x) | 10 % | −5.0 (1 seed) |

Spearman với Δ nominal (n = 18 arm): anchor share −0.05 (p 0.84); mean recovery loss −0.92 (p 4e-8); gradient share
−0.40 (p 0.10). Bỏ 3 arm LRS (n = 15): share −0.30 (p 0.28), loss −0.92 (p 1e-6), gradient share −0.59 (p 0.02).

Caveats: arm không độc lập (chia pool, chia seed); L_rec ở mức pool, không phải đúng 36 traj được chọn; LRS khác
loss (masked + q weight) và khác chất lượng nhiệm vụ (judge precision 0.07-0.11), nên loss không phải nguyên nhân duy
nhất. Đây là tương quan quan sát, cần thí nghiệm can thiệp (P1) để thành nhân quả.
- F1 là bằng chứng mạnh nhất: cùng bank với F0, ít anchor hơn 20x, nhưng chỉ giữ bridge (18.6x) -> tệ hơn F0 9 pt.
- G3-c20 (cắt continuation, tức tăng tỉ lệ bridge) tệ hơn P2-3 A 9 pt: cùng hướng.

### 1.6 Per-task: harm không cục bộ
180 điểm (arm x task): Spearman(excess loss mass của task, Δ task) = −0.11 (p 0.12), sau khi trừ mean của arm −0.04
(p 0.61); anchor share theo task −0.04. Task không có dữ liệu recovery trong arm vẫn Δ TB −1.1. Kết luận: dữ liệu khó
học làm lệch action head chung (global), nên **budget phải đặt trên toàn mix (gradient share), không theo task**.
Gain nhỏ của G2-phys vẫn tập trung ở push plate / wine rack (G4), tức lợi ích thì cục bộ, còn harm thì global.

## 2. Literature (2023-2026), mỗi nguồn kiểm trên 2 nguồn trừ khi ghi (1 src)

### 2a. Làm dữ liệu gần base policy (on-manifold / learnability)

| work | ID (nguồn 2) | điều liên quan |
|---|---|---|
| Agentic-Teacher | [2611.17394](https://arxiv.org/abs/2611.17394) (HTML + catalyzex) | code-policy repair từ failure state; correlated arm noise (rho 0.85, std 0.03 / 0.02, gripper giữ nguyên), bỏ idle frame (1 mm / 3 mrad / 0.1 mm), lọc trajectory có FM loss anomalously high dưới base pi0 đóng băng (K = 4 chunk x 32 draw). Ablation LIBERO step 4k: raw 0 %, + noise 81 %, + idle 89 %, + FM filter 95 % (120 -> 110 traj). Họ nói rõ: success không đảm bảo supervision tốt |
| PLD (ICLR 2026) | [2511.00091](https://arxiv.org/abs/2511.00091) (ICLR proceedings + UT RPL page) | residual RL takeover sau khi base policy probe ngẫu nhiên; dữ liệu clustered gần base policy nên forget ít hơn; alpha (tỉ lệ probe) tối ưu 0.6, tăng nữa thì giảm; RL expert thuần mượt, tối ưu nhưng hẹp và xa base -> kém hơn; LIBERO 99 %; 50-500 traj / task |
| RL's Razor (ICLR 2026) | [2509.04259](https://arxiv.org/abs/2509.04259) (ICLR proceedings) | forgetting dự đoán bởi KL(fine-tuned || base) trên task mới, độc lập thuật toán (R2 0.96 toy, 0.71 LLM); SFT trên label xa base quên nhiều. Lý thuyết cho việc M0 FM loss là proxy của harm nominal |
| RLDG | [2412.09858](https://arxiv.org/abs/2412.09858) (project page) | dữ liệu RL > human cho OpenVLA / Octo; relabel state của human bằng action của RL cải thiện > 50 %, nhưng full RL data vẫn tốt nhất: action quality là yếu tố chính |
| Should we learn from sampling-based planners? | [2412.09743](https://arxiv.org/abs/2412.09743) (RA-L DOI) | planner data entropy cao -> BC kém; cần consistency |
| DataMIL (ICLR 2026) | [2505.09603](https://arxiv.org/abs/2505.09603) (ICLR proceedings, GitHub) | chọn dữ liệu bằng datamodel trên proxy = policy loss trên held-out target; loss là tín hiệu nhiễu nhưng dùng được |
| Field guide (Attarian et al.) | [2610.19143](https://arxiv.org/abs/2610.19143) (LEAF lab page) | pi0.5 LIBERO-90: với success-only data SFT = critic-based steering; với mixed data steering > SFT (~70 vs ~40 %); on-policy data tốt hơn off-policy |
| RECALL (1 src) | [2606.23617](https://arxiv.org/abs/2606.23617) | recovery-only fine-tune sập (28 % vs 72 % với full replay): recovery data hẹp, cần replay |

### 2b. Sinh recovery bằng chính policy + guidance nhỏ

| work | ID (nguồn 2) | điều liên quan |
|---|---|---|
| V-GPS (CoRL 2024) | [2410.13816](https://arxiv.org/abs/2410.13816) (PMLR v270) | sample K action của generalist, re-rank bằng Cal-QL Q; K = 50 real; không đổi weights |
| GAF | [2607.02092](https://arxiv.org/abs/2607.02092) (GitHub) | SmolVLA đóng băng, critic 2.7 M, gradient của Q tại clean estimate a_hat = x_t - t v_t, clip 1, coeff 0.5, cả 10 flow step; real 60 -> 82.5 % |
| QPILOTS | [2606.14801](https://arxiv.org/abs/2606.14801) (Semantic Scholar, mlsanity) | Q-guidance tại Tweedie-denoised action, frozen VLA, 6 task sim |
| AGF | [2609.34944](https://arxiv.org/abs/2609.34944) (alphaXiv) | amortised critic guidance; LIBERO-Goal SmolVLA 76.6 -> 81.0 (QGF 80.6, Q-BoN 78.0); pooled +3.6 pt, vs QGF +0.7 ns |
| DSRL (CoRL 2025) | [2506.15799](https://arxiv.org/abs/2506.15799) (GitHub dsrl_pi0, RLinf docs) | RL trên latent noise của pi0 đóng băng; output luôn là output của base policy (đúng nghĩa on-manifold) |
| CFGRL | [2505.23458](https://arxiv.org/abs/2505.23458) (GitHub kvfrans/cfgrl) | guidance = policy improvement có điều khiển (w); w lớn = lệch base nhiều hơn |
| PA-RL | [2412.06685](https://arxiv.org/abs/2412.06685) (project page, GitHub) | sample từ policy, giữ top-m theo Q, vài bước gradient trên action, rồi distill bằng loss supervised: đúng khuôn amortised search |
| FORCE (1 src) | [2606.26006](https://arxiv.org/abs/2606.26006) | self-distill sample của chính VLA có Q trên baseline, weight exp(Q/tau) |
| RTC (NeurIPS 2025) | [2506.07339](https://arxiv.org/abs/2506.07339) (paper notes) | inpainting guidance cho flow policy: đóng băng prefix, soft mask -> nối chunk mượt. Dùng được để ép M0 bám waypoint của bridge |
| DynaGuide / ITPS | [2506.13922](https://arxiv.org/abs/2506.13922) (GitHub) / [2411.16627](https://arxiv.org/abs/2411.16627) (1 src) | guidance từ dynamics model / tương tác người; stochastic sampling cân bằng alignment vs distribution shift |

Cảnh báo cho mình: guidance bằng gradient của WM + C^pi chính là adversarial direction mà WM_OPTIMISM_VISUAL đo được
(lỗi nằm đúng hướng nhạy của critic). Gradient guidance mạnh sẽ khai thác nhanh hơn CEM. Phải giữ weight nhỏ, dùng
critic trên state thật (sim verify) và cross-WM.

### 2c. Library / option / amortised search, corrective labels

| work | ID (nguồn 2) | điều liên quan |
|---|---|---|
| RecoveryChaining | [2410.13979](https://arxiv.org/abs/2410.13979) (IROS 2025 DOI, GitHub) | HRL với nominal controller làm terminal option; reward = MC chạy suffix nominal (giống verify của mình); 51 -> 83 % shelf |
| SRSA (ICLR 2025) | [2503.04538](https://arxiv.org/abs/2503.04538) (ICLR proceedings) | dự đoán zero-shot transfer success để chọn skill từ library rồi fine-tune |
| SAILOR | [2506.05294](https://arxiv.org/abs/2506.05294) (repo, NeurIPS 2025) | WM + reward + critic search residual quanh base plan, expert iteration distill |
| D-TOUR (1 src) | [2510.07030](https://arxiv.org/abs/2510.07030) | diffusion amortise trajectory-optimisation problem cho recovery |
| IntervenGen | [2405.01472](https://arxiv.org/abs/2405.01472) (IROS 2024 DOI) | 10 intervention người -> 1,000 synthetic; mistake do policy thật sinh ra (closed-loop), recovery replay + chỉ giữ success |
| CCIL (ICLR 2024) | [2310.12972](https://arxiv.org/abs/2310.12972) (ICLR proceedings) | corrective label từ Lipschitz dynamics, lọc theo lỗi dynamics |
| Diffusion Meets DAgger | [2402.17768](https://arxiv.org/abs/2402.17768) (RSS 2024 proc.) | label hướng về frame t+k (k = 3) để tránh overshoot |
| RaC | [2509.07953](https://arxiv.org/abs/2509.07953) (project page) | mỗi intervention = recovery về state quen + correction; terminate sau intervention |
| SkillMimicGen | [2410.18907](https://arxiv.org/abs/2410.18907) (CoRL PMLR) | skill segment từ demo + motion-planned transit: khuôn demo-matched servo cho primitive |
| Recova | [2610.01178](https://arxiv.org/abs/2610.01178) (NVIDIA project page) | agent viết recovery skill (code + rollout) trong digital twin, VLM verify, DAgger |

### 2d. Bao nhiêu recovery data, mix bao nhiêu
- 2610.05151: 27 path / 1,639 window = 4.1 % anchor (1.5 % target), +10 (1 seed).
- Agentic-Teacher: 110-120 trajectory sau xử lý (ablation LIBERO); PLD: 50-500 traj / task; RLDG: 45 RL episode đủ
  100 % trên VGA; IntervenGen: 1,000 từ 10; Field guide: 500 success / task.
- Mix: mọi công trình giữ demo / replay (RECALL: không replay thì sập). Không ai báo một "tỉ lệ đúng" phổ quát;
  PLD tối ưu theo tỉ lệ probe (0.6). Số của mình: share anchor không dự đoán harm (rho −0.05), loss-weighted share thì có:
  **đặt budget theo gradient share, không theo anchor share.**
- Cut point: Agentic-Teacher chọn causal frame tại hoặc trước tương tác lỗi; 2610.05151 chọn earliest correctable state;
  CUT_POINT_STUDY của mình: T/4 phần lớn là pre-onset, M0 tự re-succeed 46 %. Không đổi kết luận đó.

## 3. Chẩn đoán tổng hợp
1. Harm nominal đến từ **gradient mass của anchor khó học**, không từ số anchor. Bridge do search chọn là nguồn chính
   (9-20x D0), perturbation / pi0-rough là đuôi cực đoan (36-111x).
2. Search khuếch đại vấn đề: WM chọn bridge lạ hơn random (1.21 vs 0.73), cùng cơ chế winner's curse.
3. Mượt hoá không giải quyết learnability (LRS v2 mượt mà 10x). Cần dữ liệu mà **M0 tự sinh được** (PLD, RL's Razor,
   DSRL), hoặc ít nhất là lọc theo M0 loss (Agentic-Teacher).
4. Continuation của M0 đã là 5x: có một sàn không tránh được; muốn share lớn hơn ~5 % mà không mất nominal thì phải
   hạ loss của bridge xuống gần mức continuation.

## 4. Ba đề xuất Phase-1, xếp theo expected gain / cost (pre-registerable)

Chung: LIBERO-Goal protocol 2610.05151, val 100 scene, init M0 s7, recipe 2k update, so với Original 5 seed
(82 / 77 / 79 / 76 / 81 = 79.0) paired, seeds 7-11, scene x seed bootstrap; RSR_E8 (168 held-out M0 failure) là
secondary; train / pilot entries only. Chi phí GPU ước lượng: 1 SFT 2k ~1 GPU-h + val ~0.25 + E8 ~0.4 = ~1.6 GPU-h /
arm-seed (A100). Visual check (skill retryvla-visual-check) sau mỗi bước sinh dữ liệu.

### P1. Learnability budget: FM-loss filter + gradient-share cap trên bank có sẵn (0 sim mới) — rẻ nhất, làm trước
- Cách làm: chấm M0 FM loss cho MỌI anchor của onset bank (wm + random; GPU inference ~0.5 GPU-h, hoặc CPU ~10 h).
  Score trajectory = mean loss của bridge (giống Agentic-Teacher, K = 4 chunk). Giữ trajectory có bridge loss
  <= 3x D0 (~0.40), thêm theo thứ tự loss tăng dần cho tới khi gradient share đạt 15 % (hoặc hết pool). Continuation giữ
  nguyên.
- Arms: `FMF` (như trên) vs `FMF-ctrl` = cùng số anchor và cùng gradient share nhưng chọn ngẫu nhiên trong pool đã qua phys
  noseam (tách hiệu ứng loss khỏi hiệu ứng share), 5 seed mỗi arm. Original tái dùng.
- Hypothesis: harm ∝ gradient share; loss-filtered data cho phép share anchor lớn hơn (dự kiến 8-12 % thay vì 1.6 %)
  mà không mất nominal, và RSR_E8 cao hơn G2-phys.
- Decision rules (viết trước kết quả):
  - "FMF beats Original": Δ >= +3 và CI lower > 0 (5 seed) -> default filter cho mọi bank.
  - "loss matters beyond share": FMF − FMF-ctrl >= +3 nominal hoặc >= +4 RSR_E8 (CI > 0) -> loss là nhân quả; ngược lại
    (|Δ| < 2) -> chỉ share quan trọng, P2 / P3 hạ ưu tiên.
  - "recovery without cost": nominal >= Original − 1 và RSR_E8 >= +8 -> đủ điều kiện scale lên Spatial (V2R).
  - Falsified nếu FMF <= G2-phys − 3.
- Chi phí: ~0.5 (scoring) + 10 x 1.6 = **~17 GPU-h**, 0 sim. MDE ~5-6 pt nominal ở 5 seed.

### P2. Re-track: chỉ giữ recovery mà M0 tái tạo được (policy-conditioned relabel)
- Cách làm: với mỗi bridge đã verify (onset bank 132 wm + 115 random), chạy lại từ snapshot của entry với chính M0:
  mỗi replan sample K = 8 chunk, chọn chunk gần đường EE của bridge nhất (DTW trên waypoint, cộng phạt gripper khác
  pha), tuỳ chọn RTC-style inpainting guidance về waypoint với weight nhỏ; sau bridge để M0 chạy tiếp; giữ episode đạt
  BDDL success. Label = action do M0 sinh ra (on-manifold theo định nghĩa), bridge gốc chỉ là "intent". Tương đương PLD
  (base policy sinh action, guidance chỉ chọn) và RLDG relabel, nhưng không cần RL.
- Gate G0 (trước SFT): yield >= 40 % bridge; M0 FM loss của bridge re-track <= 1.5x continuation (~1.0) và <= 0.5x bridge
  gốc; visual check 6 sample. Không đạt -> báo cáo, không SFT.
- Arms: `RT` (re-tracked, cùng số traj với FMF / G2-physns = 36) vs `G2-physns` (đã có 3 seed, bổ sung seed 10-11).
- Decision rules: "RT beats Original" như P1; "RT > physns" khi Δ nominal >= +3 hoặc RSR_E8 >= +4 (CI > 0) ở share anchor
  bằng nhau. Nếu yield < 40 % thì kết luận "M0 không với tới recovery đó", dùng P3.
- Chi phí: re-track ~250 entry x 2 attempt x (~70 replan x K = 8 batched) ~6-8 GPU-h + sim CPU (~2.8k env step / attempt);
  SFT 2 x 5 seed (1 arm mới + 2 seed bổ sung) ~11 GPU-h -> **~19 GPU-h**.

### P3. Thay perturbation / CEM bằng guided pi0 sampling tại failure state (value guidance + sim verify)
- Cách làm: proposal = M0 sample với guidance nhỏ trong flow (GAF / QPILOTS: gradient của Q tại a_hat = x_t − t v_t, clip,
  coeff ≤ 0.5), Q = action-chunk critic học từ candidate đã verify của mình (verif_m5b / collect_v1 / goal_search,
  label là sim GT), chấm trên state THẬT (không mơ bằng WM, tránh adversarial direction). Bỏ hoàn toàn family
  perturbation và prim+CEM; giữ pi0-smooth + primitive đã re-track (P2). Best-of-N (N = 16) -> sim verify top-4 -> bank.
- Lý do: perturbation là nguồn vừa bị WM khai thác nặng nhất (+0.69 lỗi WM) vừa khó học nhất (111x); pi0-smooth là
  family dễ học nhất (5x). Guidance cho phép với tới recovery mà sample thuần không với tới.
- Gate: trên 63 train instance paired với `wm_diverse` hiện tại: solved >= wm_diverse − 0.05 (McNemar) VÀ M0 FM loss
  bridge <= 0.8 (<= 6x D0) VÀ lỗi optimism của Q trên pick <= +0.15. Không đạt -> đóng (sau review độc lập).
- Arms: `GUIDE` bank 36 traj vs `G2-physns`, 5 seed.
- Decision rules như P2; thêm "guidance cần thiết": GUIDE − (guidance weight 0, cùng N) >= +0.08 solved.
- Chi phí: critic ~2 GPU-h, search 168 entry x N 16 x 2 round ~8 GPU-h, SFT ~16 GPU-h -> **~26 GPU-h**. Rủi ro cao nhất
  (critic bị khai thác bởi gradient), gain kỳ vọng cao nhất nếu thành (bank mới vừa on-manifold vừa giải nhiều entry).

### Thứ tự và rủi ro
P1 trả lời nhân quả "loss vs share" với 0 sim, và nếu thành thì P2 / P3 chỉ là cách tạo thêm dữ liệu loss thấp. P2 phụ
thuộc yield; P3 phụ thuộc critic. Mọi kết quả âm phải qua `retryvla-idea-review` trước khi đóng.

## 5. Giới hạn của phân tích này
- 767 anchor, 1-2 anchor / trajectory, K = 2 draw: CI rộng cho family nhỏ (perturb n 8).
- Arm loss ở mức pool; tương quan quan sát trên 18 arm không độc lập; LRS còn khác ở chất lượng nhiệm vụ.
- FM loss của M0 là proxy cho KL tới base (RL's Razor); chưa đo trên Spatial (SmolVLA) vì thiếu thời gian CPU.
- Viết sau deadline; thời gian CPU bị chia với job khác trên node.

