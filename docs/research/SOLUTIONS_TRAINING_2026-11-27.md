# Training-side solutions: recovery skill without nominal regression (LIBERO-Goal, 2610.05151 protocol)

Viết 2026-11-26/27 (research only, 0 GPU, 0 rollout mới). Input: `experiments/RESULTS.md` (Goal G2 / P2-3 / G3 / G4,
V2R), `docs/research/{REVIEW_2026-11-26_GOAL,DATA_FUNNEL_AND_FORGETTING,CUT_POINT_STUDY}.md`,
`docs/plans/RETRYVLA_V2_PLAN.md`. Ghi chú: `docs/plans/PAPER_FRAMING_2026-11.md` **không tồn tại** trong repo lúc viết
(chỉ có `PHASE_PLAN_2026-11.md`, `RETRYVLA_V2_PLAN.md`), nên framing lấy từ V2 plan + review 11-26.

Quy ước kiểm nguồn: mỗi bài có (1) arXiv export API (title / ngày / comment, query 2026-11-26 ~17:30 UTC) và (2) một
nguồn độc lập (proceedings, OpenReview, project page, GitHub chính thức). Venue chỉ ghi khi một trong hai nguồn nói rõ.
Bài trước 2023 chỉ dùng làm nền (WiSE-FT, soups, EWC, L2-SP, LP-FT, RvS, AWR, Recovery RL).

## 0. TL;DR

- **Chẩn đoán lại vấn đề (từ dữ liệu của mình).**
  1. Forgetting trên Goal là **global shift của tham số chung**: task không có recovery data cũng tụt (F0: −4.7 trên
     2 task no-data), và share càng lớn càng tụt (11-47 % → −5 đến −20). Ở 1-2 % share thì không tụt.
  2. Label quality quan trọng hơn volume: bridge-only (F1, 4 %) tệ hơn F0 (47 %, phần lớn là continuation của M0).
     Continuation anchors là "on-policy-ish" (gần phân phối của M0), bridge là off-policy (jerk 2-5x demo).
  3. Phương sai giữa seed SFT (Original 76-82, G2_wm@5 74-86) **bằng cỡ gain** (+3). Mọi phương pháp chỉ có ích nếu
     vừa dịch mean vừa không làm phương sai to thêm.
  4. Trong recipe này, **SigLIP + projector được train full** (≈ 468 M tham số trainable, `retryvla/policy/pi0_lora.py`
     D2), LLM qua LoRA r16, expert qua LoRA r32. Recovery frame (gripper ở vị trí lạ, object lệch) đi thẳng vào vision
     tower dùng chung cho mọi task: một kênh forgetting rất có thể.
- **Bằng chứng ngoài (2023-2026) nói gì.** Ba họ giải pháp có bằng chứng tốt nhất cho đúng mẫu "học skill mới, giữ
  skill cũ" trên VLA flow-matching:
  - **Weight-space merging / interpolation** (RETAIN, ICLR 2026, π0 trên LIBERO + DROID; WiSE-FT, model soups): rẻ
    nhất, không cần train, đã được chứng minh trên π0.
  - **Cách ly tham số / gradient** (KI của π0.5, NeurIPS 2025; LoRA-per-skill + router: CORAL, CLARE RA-L 2026,
    MergeVLA CVPR 2026, OrthoSkillVLA): recovery chỉ được sửa phần nhỏ, không được viết lại representation chung.
  - **Giữ update gần base** (RL's Razor ICLR 2026: forgetting ∝ KL tới base trên dữ liệu mới; RL on-policy forget ít
    hơn SFT vì tự chọn nghiệm KL nhỏ). Với mình: dữ liệu gần phân phối M0 (continuation, FM-loss thấp) ít hại hơn.
- **Top-3 thí nghiệm (xếp theo expected gain / cost; chi tiết mục 4):**
  1. **X1 WiSE / soup (0 training, ~8 GPU-h eval):** nội suy trọng số Original ↔ G2-physns / G2-phys tại α ∈ {0.25,
     0.5, 0.75}, cộng control nội suy Original ↔ Original-seed-khác và uniform soup. Mục tiêu: giữ nominal ≥ Original
     + 2 và giữ ≥ 1/2 gain RSR_E8; đồng thời giảm phương sai seed.
  2. **X2 oracle-gated switch (0 training, ~1.5 GPU-h):** chạy Original, khi có failure event (E-v1) hoặc tới step 150
     chưa xong thì chuyển sang recovery model. Là upper bound rẻ cho ý "recovery adapter + gate/router"; có control
     "chuyển sang một Original seed khác" và "re-roll Original".
  3. **X3 recovery-insulated SFT (~9 GPU-h):** per-sample gradient routing kiểu KI: loss của recovery anchor chỉ
     backprop vào action expert LoRA + projections (detach K/V của backbone cho các sample đó); D0 vẫn train mọi thứ.
     Test ở share 20 % (recipe P2-3 A, RSR +10 nhưng nominal không ổn định) xem có xoá được share penalty không.

## 1. Mẫu thất bại cần giải (số của mình)

| hiện tượng | số | hệ quả cho thiết kế |
|---|---|---|
| share nhỏ thì không hại | G2-phys 1.6 %: 82.0 (+3.0 [−3.8, +9.8], 5 seed); strict-phys 3 %: 82.0; G2_wm@5: 81.7 | gain nominal hiện tại = share effect, filter vật lý không thêm gì đo được |
| share lớn thì hại | G2 11 %: 72.7; P2-3 A 20 %: 79.3 → 73.7 (rep); F0 47 %: 68.0 | cần cơ chế cho phép thêm recovery data mà không trả giá |
| hại cả task không có data | F0: bowl_on_stove −3.0, wine_cabinet −1.7 | shift toàn cục, không phải interference cục bộ → cách ly tham số / merging hợp lý |
| bridge-only rất hại | F1 wm 4.3 %: 63.5; F1 random 49 | target off-policy, jerk cao → loss lớn → gradient lớn |
| continuation giúp | G3-c20 (cắt continuation): 70.3 vs 79.3 | dữ liệu gần phân phối M0 = "anchor"; khớp RL's Razor |
| recovery thật sự học được | RSR_E8 +5..+14 (G2-physns +9.9 [+3.6, +17.1]) | phía lợi ích là thật, nằm ở failure state |
| D_P làm FM positive | P2-3 B 75.3 (Goal, không giúp); V2R R3 +0.015 RSR (Recover, nhỏ) | pair positives như SFT thường không đủ |
| seed variance | Original SD 2.5 (76-82) | mọi so sánh cần paired seeds; soups có thể giảm variance |

Điểm kỹ thuật quan trọng cho X1: mọi run Goal đều là **M0 (merged) + một LoRA mới + full update của SigLIP/projections
trong 2k step, lr 2.5e-5**. Hai run khác nhau chỉ ở dữ liệu và seed, cùng init M0 → điều kiện chuẩn để fine-tune cùng
init nằm trong một basin (Neyshabur et al. NeurIPS 2020 2008.11687; Frankle et al. ICML 2020 1912.05671), tức là nội
suy tuyến tính có cơ hội không gặp loss barrier.

## 2. Bảng tổng hợp giải pháp (đánh giá theo fit với π0 flow matching + LoRA, cost trên 1-2 A100, effect lên mẫu lỗi)

| họ | ví dụ (đã kiểm) | evidence size | fit π0 + LoRA | cost trên 1-2 A100 | expected effect lên mẫu của mình |
|---|---|---|---|---|---|
| A. weight interpolation / soups | RETAIN 2512.08333, WiSE-FT 2109.01903, soups 2203.05482, task arithmetic 2212.04089, TIES 2306.01708 | RETAIN: π0 LIBERO 3 task + DROID 2 task real, ~40 % OOD gain real | **rất cao** (cùng init M0) | eval only (~20 min / ckpt) | nominal về gần Original hoặc hơn, RSR giữ một phần; giảm variance |
| B. adapter riêng + router | CORAL 2603.09298, CLARE 2601.09512 (RA-L 2026), MergeVLA 2511.18810 (CVPR 2026), OrthoSkillVLA 2608.19589 (PRCV 2026), MoILE 2506.04595 (ACL 2025), MoLE 2404.13628, LoRAMoE 2312.09979, TAIL 2310.05905 (ICLR 2024), SDP 2407.01531 (CoRL 2024) | nhiều bài continual / multi-task, LIBERO + real | cao, nhưng router của họ route **theo task**, mình cần route **theo trạng thái (off-nominal)** | train 1 adapter ≈ 1 SFT run; router nhỏ | nominal = Original nếu router không bắn nhầm; gain nominal bị chặn bởi số episode off-nominal (~20-38 / 100) |
| C. knowledge insulation / freeze | KI 2505.23705 (NeurIPS 2025), VLM2VLA 2509.22195 (ICLR 2026), LoRA forgets less 2405.09673 (TMLR), OFT 2502.19645 (RSS 2025) | KI: π0.5 recipe; VLM2VLA 800+ real trials | cao; KI gốc cần FAST-token loss cho backbone, mình dùng **per-sample** (chỉ recovery bị insulate) nên không cần | ≈ 1x SFT | chặn kênh global shift qua SigLIP / VLM-LoRA; cho phép share lớn hơn |
| D. loss mask / per-sample weight / data selection | Agentic-Teacher 2611.17394 (FM-loss filter), CUPID 2506.19121 (CoRL 2025), DataMIL 2505.09603 (ICLR 2026), DemInf 2502.08623 (RSS 2025), SCIZOR 2505.22626, Re-Mix 2408.14037 (CoRL 2024) | Agentic-Teacher ablation: + FM filter 89 → 95 %; Re-Mix +38 % vs uniform | cao (mình đã có `MaskedLoss` per-anchor weight) | scoring 0.3-1 GPU-h + 1 SFT / arm | giảm gradient từ target off-policy; nhưng filter vật lý đã không thêm gì → kỳ vọng vừa phải |
| E. quality / advantage conditioning | RECAP π*0.6 2511.14759, CFGRL 2505.23458 (NeurIPS 2026), RvS 2112.10751, RePO-VLA 2605.09410 (NeurIPS 2026) | RECAP: real, throughput 2x; CFGRL: offline RL + GCBC | trung bình (thêm token text vào prompt π0 dễ; CFG cần 2 forward / step) | 1 SFT / arm + label | "recovery-mode" token tách hai phân phối action trong cùng model; nominal prompt giữ nguyên → ít interference |
| F. anchoring tới base (KL / velocity) | RL's Razor 2509.04259 (ICLR 2026), L2-SP 1802.01483, EWC 1612.00796, L_A trong V2 plan | RL's Razor: forgetting–KL R² 0.96 (toy) / 0.71 (LLM) | cao; L_A = ‖v_θ − v_θ0‖² trên D0 states | 1.3-1.5x SFT (thêm forward base) | trực tiếp chặn drift trên nominal states; chưa từng chạy trên Goal |
| G. replay ratio / curriculum / two-stage | RETAIN co-FT, LP-FT 2202.10054, CLEAR-style replay; Spatial P3 sweep của mình | Spatial: r 0.5 OK, 0.7 fail margin | cao | 1 SFT / arm | re-anchoring stage cuối trên D0 có thể "xoá" recovery; cần đo RSR |
| H. preference (DPO / KTO) cho flow | Diffusion-DPO 2311.12908 (CVPR 2024), Diffusion-KTO 2404.04465 (NeurIPS 2024), FlowPRO/RPRO 2606.05468, CARF 2609.21982, FAR 2607.01111 (CoRL 2026), GRAPE 2411.19309, APO 2506.07127 (NeurIPS 2025) | FlowPRO: plain Flow-DPO bị reward hacking, cần proximal reg | trung bình: FM loss làm surrogate log-lik (cần ref forward) | 1.5-2x SFT | D_P ít (78 traj Spatial; vài chục cặp Goal) → tín hiệu nhỏ, nguy cơ đẩy a- làm lệch nominal |
| I. RL fine-tuning (contrast) | π_RL 2510.25889, ReinFlow 2505.22094 (NeurIPS 2025), FPO 2507.21053, Flow-GRPO 2505.05470, DPPO 2409.00588, SimpleVLA-RL 2509.09674, DSRL 2506.15799 (CoRL 2025) | π_RL: LIBERO few-shot π0 57.6 → 97.6 %, 8x H100 | cao về lý thuyết (RL's Razor: forget ít) | **cao** (8 H100 trong π_RL; mình 1-2 A100) | đúng hướng chống forgetting nhưng ngoài budget; DSRL (base đóng băng, chỉ học noise policy) là bản rẻ nhất |

## 3. Chi tiết từng họ

### 3.A Weight interpolation / model soups (WiSE-FT, RETAIN, task arithmetic)

- **Cơ chế.** θ(α) = (1 − α)·θ_A + α·θ_B với θ_A, θ_B fine-tune từ cùng init. Nếu hai điểm cùng basin, θ(α) thường giữ
  được skill của cả hai và "mượt" hơn từng endpoint (WiSE-FT: ID ≈ / tốt hơn, OOD +4-6 pp; soups: trung bình nhiều seed
  tốt hơn seed tốt nhất). Task arithmetic: τ = θ_ft − θ_pre là "task vector", có thể cộng / scale.
- **Evidence trên VLA.** RETAIN (Yadav et al., arXiv 2512.08333 v3; ICLR 2026 theo journal_ref arXiv + proceedings
  iclr.cc; code `yajatyadav/RETAIN_code`): π0 trên LIBERO (pretrain LIBERO-{object, spatial, goal, 90}, fine-tune 3
  task LIBERO-10) và π0-FAST-DROID (2 task real). Nội suy pretrained ↔ fine-tuned giữ generalist skill (20 task
  LIBERO / 44 task DROID) và tăng OOD của task mới, ~40 % trên real. α chọn trên một OOD "val" scene; DROID α ∈ {0.25,
  0.5, 0.75}, α = 0.5 thường tốt; LIBERO dùng α = 0.9 (task-FT) vì base LIBERO ít "generalist". Co-FT + merge tốt hơn
  task-FT + merge. MergeVLA (2511.18810, CVPR 2026) cảnh báo: merge các LoRA **train trên task khác nhau** cho ~0 %
  success; Bring-Back-Language (2611.18891, EACL 2027) dùng SVD của task vector và hệ số per-component.
- **Fit.** Rất cao, và tốt hơn bối cảnh RETAIN: hai endpoint của mình (Original, G2-physns) cùng init M0, cùng D0, chỉ
  khác ~2-10 % recovery anchors. Đây không phải merge hai task khác nhau (trường hợp MergeVLA thất bại).
  - LoRA: phải nội suy **ΔW = B·A**, không nội suy A, B riêng (hai run có A init khác). Có cách chính xác mà không tăng
    disk nhiều: (1 − α)·B₁A₁ + α·B₂A₂ = [√(1−α)·B₁ | √α·B₂] · [√(1−α)·A₁ ; √α·A₂], tức là một LoRA rank r₁ + r₂
    (32 cho LLM, 64 cho expert, lora_alpha = rank để scale 1). Cần `_inject` đọc rank từ `meta.json`.
  - SigLIP / projector / state-action proj: nội suy tuyến tính trực tiếp (cùng init M0).
  - Kích thước ~1-1.7 GB / ckpt merged (quota /pfss: xoá `trainable.pt` merged sau eval, giữ recipe để tái tạo).
- **Cost.** 0 training. Merge trên CPU vài phút; eval val ~5-6 min (10 workers), E8 ~15 min / ckpt trên 1 A100.
- **Expected effect.** Nominal: về khoảng giữa hoặc trên cả hai endpoint; soups seed có thể giảm SD 2.5. RSR_E8: một
  phần của +9.9 (giả thuyết ~+5 ở α = 0.5). Rủi ro: barrier tuyến tính (nominal giữa đường tụt) do vision tower full FT.

### 3.B Adapter riêng cho recovery + router (mixture-of-adapters)

- **Cơ chế.** Base đóng băng (ở đây: Original), một LoRA riêng cho recovery; router chọn adapter theo input.
  CORAL (2603.09298): 1 LoRA / task, route bằng instruction, cách ly tham số tuyệt đối, LIBERO + real Galaxea R1.
  CLARE (2601.09512, RA-L 2026, `tum-lsy.github.io/clare`): adapter mở rộng theo layer-wise feature similarity, router
  autoencoder không cần task label, LIBERO + 5 task thật, không exemplar. OrthoSkillVLA (2608.19589, PRCV 2026): MoE
  decoder per-skill, router training-free theo feature affinity; nhận xét quan trọng: **action head mã hoá velocity
  pattern cục bộ, rất nhạy với update**. MoILE (2506.04595, ACL 2025), MoLE (2404.13628), LoRAMoE (2312.09979: LoRA
  MoE để giảm world-knowledge forgetting), TAIL (2310.05905, ICLR 2024), Sparse Diffusion Policy (2407.01531, CoRL 2024).
- **Khác biệt then chốt.** Mọi bài trên route **theo task** (instruction hoặc cluster task). Recovery không phải task
  mới, mà là **mode trong cùng task**, xuất hiện giữa episode. Router phải dựa trên **lịch sử / trạng thái** (off-nominal
  detection), giống switching trong Recovery RL (2010.15920, RA-L 2021: recovery policy riêng + safety critic quyết định
  khi nào chuyển). Mình chưa thấy bài VLA 2023-2026 nào route recovery adapter theo failure-state classifier: vừa là
  khoảng trống, vừa là rủi ro (router sai trên nominal state = forgetting quay lại qua cửa sau).
- **Fit / cost.** Train adapter = 1 SFT run (~2.3 h). Router: linear probe trên prefix feature hoặc rule time-based.
  Upper bound rẻ (X2) không cần train gì: dùng hai checkpoint đã có, switch tại event.
- **Expected effect.** Nominal = Original trên episode không bị trigger. Gain bị chặn bởi episode off-nominal: với
  Original s7 / s8 / s9: 22 / 21 / 23 episode có event, trong đó 8 / 14 / 9 fail; 31 / 38 / 36 episode còn chạy ở step
  150, trong đó Original tự thành công 13 / 14 / 17. Switch chỉ có lợi nếu recovery model hoàn tất **tốt hơn** Original
  từ chính các state đó (Original tự cứu ~40-45 %). Ceiling thực tế: +0-4 điểm.

### 3.C Knowledge insulation (π0.5 KI) và freeze backbone

- **Cơ chế.** KI (Driess et al., 2505.23705; NeurIPS 2025 proceedings; blog `pi.website/research/knowledge_insulation`):
  stop-gradient trên K/V của backbone ở đường attention mà action expert dùng (P_ab·sg(V_b)), backbone vẫn được train
  bằng FAST-token loss. Mục đích: gradient của expert mới khởi tạo không làm hỏng VLM. openpi issue #649 (nguồn chính
  thức): KI dùng khi pretrain π0.5; khi fine-tune, "freeze toàn bộ VLM chỉ train expert likely works a lot worse", và
  openpi chưa thấy bằng chứng KI giúp lúc fine-tune. VLM2VLA (2509.22195, ICLR 2026): LoRA-only giữ VQA, forgetting chủ
  yếu do data mismatch. LoRA Learns Less and Forgets Less (2405.09673, TMLR): LoRA forget ít hơn full FT và hơn
  weight decay / dropout.
- **Áp dụng cho mình (không phải KI nguyên bản).** Không freeze backbone cho mọi dữ liệu (D0 vẫn cần chỉnh SigLIP / VLM
  như recipe Original), mà **insulate theo sample**: recovery anchors chỉ được sửa action expert LoRA + action/time/
  state projections; D0 anchors sửa mọi thứ như cũ. Lý do: phân tích forgetting cho thấy shift toàn cục, và recovery
  frame là phân phối ảnh lạ đi vào vision tower full-trainable. Cài đặt trong lerobot π0 (`compute_layer_complete`,
  `modeling_pi0.py` dòng 229-290): Q / K / V của prefix (backbone) và suffix (expert) được concat rồi attention chung.
  Per-sample KI = với sample có cờ `is_rec`, suffix queries attend tới `K_b.detach()`, `V_b.detach()` (prefix queries
  giữ nguyên); làm bằng 2 lời gọi attention (prefix rows, suffix rows) hoặc `torch.where` trên K/V. Prefix forward của
  sample recovery khi đó không nhận gradient nào (không có loss trên prefix trong π0), nên SigLIP / VLM-LoRA chỉ học
  từ D0.
- **Cost.** ≈ 1.0-1.1x SFT. Engineering ~3-4 h (sửa attention + test gradient bằng hook: grad của SigLIP từ batch
  recovery-only phải bằng 0).
- **Expected effect.** Chặn kênh vision/VLM; không chặn drift trong action expert (OrthoSkillVLA: expert là phần nhạy
  nhất). Nếu forgetting chủ yếu qua backbone → share 20 % hết hại; nếu qua expert → không đổi. Một thí nghiệm, hai kết
  luận đều có ích (định vị forgetting).

### 3.D Loss masking, per-sample weighting, data selection

- **Cơ chế / evidence.**
  - Agentic-Teacher (2611.17394): lọc continuation theo **FM loss của base đóng băng**; LIBERO-10 79.4 → 87.8; ablation
    raw code-policy 0 %, + correlated noise 81, + bỏ idle 89, + FM filter 95 (theo review `AGENTIC_SMOOTHNESS_REVIEW`).
  - CUPID (2506.19121, CoRL 2025 PMLR v305): influence của từng demo lên closed-loop return (REINFORCE-style, 25-100
    rollout); < 33 % dữ liệu đã curate đạt SOTA Robomimic.
  - DataMIL (2505.09603, ICLR 2026): datamodels với surrogate loss trên target demos, chọn dữ liệu từ OXE.
  - DemInf (2502.08623, RSS 2025): mutual information state-action; SCIZOR (2505.22626): lọc state-action pair kém /
    dư thừa.
  - Re-Mix (2408.14037, CoRL 2024 PMLR v270): trọng số domain bằng group DRO trên **excess loss** so với reference
    (down-weight domain dễ và domain không fit được); +38 % so với uniform.
- **Fit.** Cao: `MaskedLoss` đã có per-anchor weight w (LRS-G). Mình đã có filter vật lý (không thêm gì ngoài share).
  FM-loss của M0 là trục khác: đo **policy-compatibility**, khớp RL's Razor (target KL-gần base thì ít forget).
- **Expected effect.** Vừa phải. Ưu điểm lớn: phần scoring rẻ (1 forward pass M0 trên mọi recovery anchor, ~0.3
  GPU-h) và cho diagnostic trước khi train: nếu anchors của các arm có hại (F1 bridge) có FM-loss cao hơn rõ rệt so với
  continuation, thì weighting đáng train; nếu không thì bỏ.

### 3.E Advantage / quality conditioning, recovery-mode token

- **Cơ chế.** RECAP / π*0.6 (2511.14759; blog pi.website): value phân phối → advantage → chỉ báo nhị phân "Advantage:
  positive/negative" trong prompt, dropout để dùng CFG; inference đặt positive. CFGRL (2505.23458; NeurIPS 2026 theo
  trang hội nghị; code `kvfrans/cfgrl`): product policy π̂·f(A)^w, guidance weight w điều khiển mức policy improvement
  lúc test, không cần Q-function khi dùng optimality label nhị phân. RvS (2112.10751): conditioning outcome là đủ cho
  nhiều offline-RL task. RePO-VLA (2605.09410, NeurIPS 2026): value token trên π0.5, trong đó segment lỗi có value thấp.
- **Áp dụng.** Token "mode: recovery" cho recovery anchors, "mode: nominal" cho D0 (dropout 10-20 %). Inference: (a)
  luôn "nominal" → nominal gần Original, recovery skill nằm trong model nhưng chỉ bật khi (b) một predictor (từ history:
  event detector, step count, gripper state) bật token. Đây là bản "một model" của 3.B, router thay bằng conditioning.
- **Rủi ro.** Text token qua LLM-LoRA vẫn cho gradient recovery chạy vào backbone (không có insulation); với 2k step
  token có thể không đủ tách hai phân phối. Nên kết hợp với 3.C.

### 3.F KL / velocity anchoring về base trên nominal state

- **Cơ chế.** L_A = E_{o∈D0} ‖v_θ(x_t, t, o) − sg(v_θ0(x_t, t, o))‖² (V2 plan, mục 5). RL's Razor (2509.04259, ICLR
  2026): forgetting dự đoán được bởi KL(π_ft ‖ π_base) **trên dữ liệu task mới** (R² 0.96 toy / 0.71 LLM), RL on-policy
  ngầm chọn nghiệm KL nhỏ; SFT trên đúng phân phối KL-minimal thậm chí tốt hơn RL. L2-SP, EWC là bản weight-space.
- **Nhận xét quan trọng.** Theo RL's Razor, anchoring nên đặt **trên state của dữ liệu mới** (recovery states), không
  chỉ trên D0: tức là giới hạn mức lệch của policy khỏi M0 ở chính các state recovery, trừ phần cần sửa. Với flow
  matching, KL không có dạng đóng; velocity-MSE với base là proxy hợp lý.
- **Cost / effect.** 1.3-1.5x SFT (forward base thêm). Chưa chạy trên Goal. Có khả năng giảm drift, nhưng cũng giảm
  RSR (trade-off cần sweep λ_A, đắt hơn X1-X3).

### 3.G Replay ratio, curriculum, two-stage

- Spatial P3: ratio 0.5 giữ nominal, 0.7 fail margin; Goal pooling đều (không ratio control). Two-stage "recovery
  rồi re-anchor nominal" (LP-FT 2202.10054 về thứ tự train; RETAIN continual: merge sau mỗi stage) có nguy cơ xoá
  recovery, nên chỉ có nghĩa khi đo RSR_E8. Ở mức share hiện tại, ratio schedule và share đang trùng nhau; mình đã
  có đường cong share (G4 share sweep đang chạy). Không đề xuất arm riêng trước khi share sweep xong.

### 3.H Preference objectives cho flow / diffusion

- Diffusion-DPO (2311.12908, CVPR 2024), Diffusion-KTO (2404.04465, NeurIPS 2024: chỉ cần nhãn nhị phân per-sample),
  FlowPRO / RPRO (2606.05468; project page): DPO trên FM action head của π0 / π0.5, FM loss là surrogate −log π, **plain
  Flow-DPO bị reward hacking**, cần proximal regularizer neo |r_θ| về 0. CARF (2609.21982): attraction-repulsion với
  ReLU chặn. FAR (2607.01111, CoRL 2026): FCPA log-sigmoid lúc test. GRAPE (2411.19309), APO (2506.07127, NeurIPS 2025)
  cho VLA autoregressive.
- **Fit với mình.** L_P (bounded hinge) đã thiết kế; D_P như FM positive không giúp trên Goal. Preference chỉ thêm tín
  hiệu ở **state pre-failure** (ít: vài chục cặp), và repulsion trên a- là chính loại update có thể đẩy policy xa base
  (ngược RL's Razor). Ưu tiên thấp cho mục tiêu "không hại nominal".

### 3.I RL fine-tuning (contrast)

- π_RL (2510.25889; HF model card RLinf): LIBERO few-shot π0 57.6 → 97.6 %, π0.5 77.1 → 98.3 %, freeze VLM, chỉ train
  expert 300 M, **8x H100**. ReinFlow (2505.22094, NeurIPS 2025), FPO (2507.21053), Flow-GRPO (2505.05470), DPPO
  (2409.00588), SimpleVLA-RL (2509.09674). DSRL (2506.15799, CoRL 2025; code `nakamotoo/dsrl_pi0`): **π0 đóng băng**,
  SAC học noise ban đầu (~500 K tham số), LIBERO / Aloha / Franka.
- **Đánh giá.** RL on-policy là lời giải "đúng" theo RL's Razor (forget ít) và π_RL cho thấy Goal có headroom lớn,
  nhưng budget vượt 1-2 A100 nhiều lần (rollout MuJoCo CPU-bound + 3B model). DSRL là phương án rẻ duy nhất và giữ
  nominal theo cấu trúc (base không đổi), nhưng nó là test-time steering, không phải "distill recovery data" (lệch
  khỏi câu hỏi nghiên cứu). Dùng làm contrast trong paper, không làm arm.

## 4. Ba thí nghiệm đề xuất (pre-registerable; val + E8 only, test split không động tới)

Chung: protocol Goal 2610.05151 (100 val scene, policy seed 7, 300 step, replan 5); RSR_E8 = 168 entry held-out M0
failure (`data/goal_rsi/audit_m0_s8/entries_s8.jsonl`). Δ = paired theo seed SFT, bootstrap scene × seed 10k, cộng
McNemar fixed / broken. Checkpoint có sẵn (đã kiểm `trainable.pt` tồn tại): `orig_s7..s11`, `g2physns_wm_s7..s9`,
`g2phys_wm_s7..s11`, `p23_A_s7..s9`, `g3rep_A_s10..s12`, `m0_s7/ckpt_04000`. Mỗi thí nghiệm phải có intake trong
`experiments/PLAN.md` (skill `retryvla-idea-loop`) trước khi chạy; mục này là bản nháp rule.

| # | tên | training | GPU-h | engineering | expected gain | rủi ro chính |
|---|---|---|---|---|---|---|
| X1 | WiSE / soup Original ↔ recovery | 0 | ~8 (2 GPU ≈ 4 h wall) | ~2 h (merge script + rank-từ-meta trong loader) | nominal +1..+3 vs Original, RSR ~1/2 của endpoint | barrier tuyến tính; gain chỉ là averaging (có control) |
| X2 | oracle-gated switch Original → recovery | 0 | ~1.5 | ~2 h (entries từ val episodes) | nominal +0..+3, upper bound cho router | ceiling thấp (8-14 fail có event / seed) |
| X3 | recovery-insulated SFT (per-sample KI) ở share 20 % | 3 seed | ~9 | ~3-4 h (attention + gradient test) | share penalty mất: nominal ≈ Original, RSR +8..+10 | forgetting đi qua expert, không qua backbone |

### X1. WiSE-FT / model soup giữa Original và recovery models (ưu tiên 1)

- **Câu hỏi.** Nội suy trọng số Original ↔ recovery model (cùng init M0) có giữ nominal ≥ Original và giữ phần lớn RSR
  không? Và có phải chỉ là hiệu ứng "trung bình hai model bất kỳ" không?
- **Checkpoint merged (định nghĩa trong weight space đã merge ΔW = BA; LoRA concat chính xác, xem 3.A):**
  - X1-a (RSR mạnh): θ = (1 − α)·orig_s + α·g2physns_s, s ∈ {7, 8, 9}, α ∈ {0.25, 0.5, 0.75} → 9 ckpt.
  - X1-b (5 seed cho nominal): α = 0.5, orig_s ↔ g2phys_s, s ∈ {7..11} → 5 ckpt.
  - X1-c control "averaging only": α = 0.5, orig_s ↔ orig_{s+1} cho (7,8), (8,9), (9,10), (10,11), (11,7) → 5 ckpt.
  - X1-d exploratory (1 eval mỗi cái, không có seed replication): uniform soup orig s7-11, g2phys s7-11, g2physns
    s7-9 → 3 ckpt.
  - Bổ sung E8 cho orig s10 / s11 (chưa có) để paired RSR 5 seed: 2 eval.
- **Primary (chốt trước):** α = 0.5 của X1-a (3 seed) cho RSR, X1-b (5 seed) cho nominal. Lưới α chỉ báo cáo mô tả,
  **không chọn α trên val rồi claim trên val**.
- **Decision rules.**
  - R1 "free lunch": X1-b nominal Δ vs Original ≥ +2 với CI lower > −2 (non-inferiority margin 2) **và** X1-a α = 0.5
    RSR_E8 Δ vs Original ≥ +5 với CI > 0 **và** X1-b − X1-c ≥ +1 (không chỉ là averaging) → WiSE thành bước post-hoc
    mặc định; chạy test split đúng 1 lần (× 3 seed, cùng Original) với α = 0.5.
  - R2 "averaging, không phải recovery": X1-c ≥ Original + 2 và |X1-b − X1-c| < 1 → claim "seed averaging giúp nominal"
    (độc lập với recovery data); recovery data chỉ đóng góp RSR. Vẫn hữu ích: soup là cách rẻ giảm SD 2.5.
  - R3 "barrier": nominal ở α = 0.5 < min(hai endpoint) − 3 trên ≥ 2 / 3 seed của X1-a → linear path có barrier (nghi
    vision tower); thử lại một lần với chỉ nội suy LoRA, giữ SigLIP của Original (1 seed, exploratory); nếu vẫn barrier
    thì đóng X1.
  - R4 "RSR mất hết": RSR_E8 ở α = 0.5 < Original + 3 → interpolation pha loãng recovery skill; đi X2 / X3.
- **Chẩn đoán rẻ đi kèm.** FM loss held-out trên val demos D0 dọc theo α ∈ {0, 0.25, 0.5, 0.75, 1} (1 forward / ckpt,
  ~2 min) để kiểm loss barrier; đây cũng là "held-out nominal FM loss" mà DATA_FUNNEL nói là chưa log.
- **Budget.** 24 ckpt × (val 6 min + E8 15 min) ≈ 8.4 GPU-h, E8 cho X1-d bỏ được (−1 h). Quota: merged ckpt ~1-1.7 GB,
  xoá sau eval (giữ `summary.json`, `episodes.jsonl`).

### X2. Oracle-gated switch Original → recovery model (ưu tiên 2; upper bound cho "recovery adapter + router")

- **Câu hỏi.** Nếu một router hoàn hảo bật recovery model đúng lúc episode lệch nominal, nominal success có tăng
  không, hơn hẳn việc chỉ "đổi sang một policy khác"?
- **Thiết kế (không cần train, không cần model mới).** Từ `orig_s/ckpt_02000/eval_val_ev/{actions.npz, events.jsonl}`
  (s7-9; s10-11 có `eval_val/actions.npz`, cần chạy `goal_events` offline): với mỗi episode có trigger, tạo entry
  (scene init + prefix actions của Original tới t_trigger), replay bit-exact rồi cho policy P chạy tiếp tối đa
  300 − t_trigger step (`eval_goal_entries --cont`). Nominal SR_switch(P) = (#thành công trước trigger) + (#entry P
  thành công). Episode không có trigger giữ outcome của Original (cùng model, cùng seed, deterministic tới trigger).
  - Trigger T1 (oracle, primary): first E-v1 event (grasp_lost / displaced / tipped / stall), ~21-23 episode / seed.
  - Trigger T2 (deployable, secondary): còn chạy ở step 150, 31-38 episode / seed.
  - P ∈ {g2physns_s (recovery), g2phys_s, orig_s re-roll (cùng ckpt, policy seed hash mới), orig_{s+1} (Original khác
    seed)}.
- **Decision rules.**
  - "Router đáng xây": SR_switch(g2physns) ≥ SR(Original) + 3 (paired 3 seed) **và** ≥ max(SR_switch(orig re-roll),
    SR_switch(orig_{s+1})) + 2 ở T1 hoặc T2 → bước tiếp: (i) router deployable (linear probe trên prefix feature của
    Original, train trên train-split D0 frames vs bank entry states; FPR ≤ 2 % trên val demos), (ii) adapter recovery
    riêng (LoRA trên expert, train chỉ trên bank + 20 % D0) thay vì cả model G2-physns.
  - "Chỉ là re-roll": SR_switch(g2physns) ≈ SR_switch(orig_{s+1}) (|Δ| < 2) → lợi ích do đổi noise / policy, không do
    recovery skill; đóng hướng router cho nominal (recovery vẫn có giá trị trên E8).
  - "Hại": SR_switch(g2physns) < SR(Original) − 2 → recovery model phá các episode Original tự cứu (13-17 / seed ở T2);
    đóng.
- **Budget.** Entry ~22 + ~35 / seed × 4 policy × 3 seed ≈ 680 rollout ngắn ≈ 1.5 GPU-h (E8 168 entry ≈ 15 min).

### X3. Recovery-insulated SFT: per-sample gradient routing kiểu KI (ưu tiên 3)

- **Câu hỏi.** Penalty nominal của share lớn có đến từ việc recovery frame viết lại representation chung (SigLIP full
  FT + VLM-LoRA) không? Nếu chặn kênh này, share 20 % có giữ được nominal và RSR +10 không?
- **Arm.** Recipe P2-3 A (72 traj, ~20 % share, M0 + 2k, như cũ), thêm cờ `--insulate_recovery`: với anchor có `src > 0`
  (recovery), suffix queries attend tới `sg(K_prefix)`, `sg(V_prefix)` → SigLIP / projector / VLM-LoRA chỉ nhận gradient
  từ D0; recovery chỉ sửa expert LoRA + `state_proj`, `action_in/out_proj`, `action_time_mlp_*`. Seeds 7, 8, 9.
  - Smoke (bắt buộc, 0.3 GPU-h): batch recovery-only → grad norm của `vision_tower.*` và `lora_*` trong LLM = 0 chính
    xác; batch D0-only → loss / grad trùng code cũ (bit-for-bit trong fp32 trên 1 batch).
  - Baseline paired: P2-3 A s7-9 (83 / 78 / 77) + rep s10-12 (75 / 75 / 71); Original s7-9.
- **Decision rules.**
  - "Insulation xoá share penalty": nominal ≥ Original − 1 (paired s7-9, CI lower > −5) **và** RSR_E8 ≥ P2-3 A − 2
    (P2-3 A 0.248) → mở bước scale: F0-size share (47 %) insulated, 1 seed sàng lọc rồi 3 seed.
  - "Forgetting nằm ở expert": nominal ≤ P2-3 A mean (76.5 trên 6 seed) → insulation backbone không đủ; bước tiếp là
    L_A (velocity anchoring) trên recovery states (3.F) hoặc expert-only adapter riêng (3.B), không phải backbone.
  - "Recovery bị tê liệt": RSR_E8 < Original + 3 → recovery cần chỉnh perception (vision tower) để nhận ra failure
    state; đóng X3, chuyển sang adapter + router (X2 → B).
- **Budget.** 3 × (2.3 h train + 0.1 h val + 0.25 h E8) ≈ 8 GPU-h + smoke 0.3 + 1 seed bonus nếu cần ≈ 9 GPU-h.

### Backup (rẻ, chạy nếu còn GPU)

- **X4 FM-loss diagnostic → weighting (0.3 GPU-h diagnostic).** FM loss của M0 (cùng 8 (t, ε) cố định / anchor) trên
  mọi anchor của F1 bridge, G2 continuation, G2-physns, D0. Rule: nếu median FM-loss của F1 bridge ≥ 2x D0 **và** của
  continuation ≤ 1.3x D0 → train arm weighting w = clip(L_D0_median / L_anchor, 0.2, 1) tại share 20 % (3 seed, 8
  GPU-h); nếu không thì đóng (label "rough" không thể hiện qua FM loss).
- **X5 recovery-mode token (1 seed sàng lọc, 2.8 GPU-h).** Prompt "… [recovery]" cho recovery anchors, dropout 20 %;
  eval nominal luôn với prompt gốc, E8 với cả hai prompt. Chỉ chạy sau X3 (để biết có cần insulation kèm theo không).

### Thứ tự và tổng chi phí

X1 (8 GPU-h) và X2 (1.5 GPU-h) chạy song song trên 2 GPU, xong trong ~4-5 h wall, không đụng training. X3 chỉ chạy sau
khi đọc X1 / X2 (nếu X1 R1 đạt thì X3 vẫn đáng chạy vì nó trả lời câu "forgetting ở đâu"). Tổng ~19 GPU-h cho cả ba.

## 5. Vì sao không ưu tiên các hướng khác

- **DPO / KTO cho flow (3.H):** tín hiệu ở pre-failure state ít (vài chục cặp Goal), plain Flow-DPO bị reward hacking
  (FlowPRO), repulsion đẩy xa base, đúng loại update gây forgetting; D_P như FM positive đã không giúp trên Goal.
- **RL fine-tuning (3.I):** đúng hướng chống forgetting nhưng π_RL dùng 8x H100; ngoài budget 1-2 A100 trước deadline.
- **Freeze toàn bộ VLM cho mọi dữ liệu:** openpi #649 nói fine-tune như vậy kém hơn; recipe Original cần SigLIP FT trên
  D0. Bản per-sample (X3) tránh được vấn đề này.
- **Data selection kiểu CUPID / DataMIL:** cần nhiều rollout / nhiều fine-tune nhỏ (KIN-1 đã đo chi phí micro-fine-tune
  tương tự); filter vật lý đã không thêm gì ngoài share, nên kỳ vọng thấp với chi phí cao. X4 là bản rẻ.

## 6. Nguồn (2 nguồn mỗi bài; arXiv export API 2026-11-26 + nguồn thứ hai)

| arXiv | bài | venue / trạng thái | nguồn thứ hai |
|---|---|---|---|
| 2512.08333 | RETAIN: Robust Finetuning of VLA Robot Policies via Parameter Merging | ICLR 2026 | proceedings.iclr.cc; retain.yajatyadav.com; GitHub RETAIN_code |
| 2109.01903 | WiSE-FT: Robust fine-tuning of zero-shot models | CVPR 2022 (nền) | arXiv comment + CVPR |
| 2203.05482 | Model soups | ICML 2022 (nền) | arXiv comment |
| 2212.04089 | Editing Models with Task Arithmetic | ICLR 2023 | arXiv comment |
| 2306.01708 | TIES-Merging | NeurIPS 2023 | arXiv comment |
| 2511.18810 | MergeVLA | CVPR 2026 | mergevla.github.io |
| 2611.18891 | Bring Back Language to Actions (VLA ↔ VLM merging) | EACL 2027 (comment) | arXiv HTML |
| 2008.11687 / 1912.05671 | What is being transferred / Linear Mode Connectivity | NeurIPS 2020 / ICML 2020 (nền) | arXiv |
| 2505.23705 | Knowledge Insulating VLA (π0.5 KI) | NeurIPS 2025 | proceedings.neurips.cc; pi.website/research/knowledge_insulation; openpi issue #649 |
| 2509.22195 | VLM2VLA: Actions as Language | ICLR 2026 | proceedings.iclr.cc; vlm2vla.github.io |
| 2405.09673 | LoRA Learns Less and Forgets Less | TMLR (comment) | arXiv |
| 2502.19645 | OpenVLA-OFT | RSS 2025 | arXiv comment, openvla-oft site |
| 2603.09298 | CORAL: LoRA Experts | preprint | arXiv HTML |
| 2601.09512 | CLARE | RA-L 2026 | tum-lsy.github.io/clare |
| 2608.19589 | OrthoSkillVLA | PRCV 2026 | arXiv HTML |
| 2506.04595 | Task-aware MoILE | ACL 2025 | arXiv comment |
| 2404.13628 / 2312.09979 | MoLE / LoRAMoE | venue không ghi (comment arXiv không nêu) | arXiv abs + search index |
| 2310.05905 | TAIL | ICLR 2024 | arXiv comment |
| 2407.01531 | Sparse Diffusion Policy | CoRL 2024 | arXiv comment |
| 2010.15920 | Recovery RL | RA-L / ICRA 2021 (nền) | journal_ref |
| 2611.17394 | Agentic-Teacher | preprint | repo review `AGENTIC_SMOOTHNESS_REVIEW.md` (abs + search index) |
| 2506.19121 | CUPID | CoRL 2025 | PMLR v305; cupid-curation.github.io |
| 2505.09603 | DataMIL | ICLR 2026 | proceedings.iclr.cc; robin-lab site |
| 2502.08623 | DemInf | RSS 2025 | arXiv comment; jhejna.github.io |
| 2505.22626 | SCIZOR | preprint | arXiv |
| 2408.14037 | Re-Mix | CoRL 2024 | PMLR v270; GitHub jhejna/remix |
| 2511.14759 | π*0.6 / RECAP | preprint | pi.website/blog/pistar06 |
| 2505.23458 | CFGRL | NeurIPS 2026 | GitHub kvfrans/cfgrl; trang hội nghị |
| 2112.10751 / 1910.00177 | RvS / AWR | ICLR 2022 / preprint (nền) | arXiv |
| 2605.09410 | RePO-VLA | NeurIPS 2026 | project page (RELATED_WORK P2) |
| 2509.04259 | RL's Razor | ICLR 2026 | proceedings.iclr.cc; jyopari.github.io |
| 1802.01483 / 1612.00796 | L2-SP / EWC | ICML 2018 / PNAS (nền) | arXiv |
| 2202.10054 | LP-FT | ICLR 2022 oral (nền) | arXiv comment |
| 2311.12908 | Diffusion-DPO | CVPR 2024 | arXiv |
| 2404.04465 | Diffusion-KTO | NeurIPS 2024 | arXiv |
| 2606.05468 | FlowPRO / RPRO | preprint | project page wuyeyexvnainai.github.io/flowpro |
| 2609.21982 / 2607.01111 | CARF / FAR | preprint / CoRL 2026 | project pages (RELATED_WORK P1, P4) |
| 2411.19309 / 2506.07127 | GRAPE / APO | preprint / NeurIPS 2025 | grape-vla.github.io / arXiv comment |
| 2510.25889 | π_RL | preprint | HF model card RLinf; rlinf.org |
| 2505.22094 | ReinFlow | NeurIPS 2025 | journal_ref arXiv |
| 2507.21053 / 2505.05470 / 2409.00588 / 2509.09674 | FPO / Flow-GRPO / DPPO / SimpleVLA-RL | preprints | blog / GitHub / project sites |
| 2506.15799 | DSRL | CoRL 2025 | GitHub nakamotoo/dsrl_pi0, ajwagen/dsrl |
| 2610.05151 | Recursive Self-Improvement via Local Recovery Supervision (protocol) | preprint | RELATED_WORK L3 |

Ghi chú venue: 2312.09979 (LoRAMoE) không có venue trong comment arXiv, mình không ghi venue. CORAL, SCIZOR, FlowPRO,
π_RL, RECAP: preprint tại thời điểm kiểm.
