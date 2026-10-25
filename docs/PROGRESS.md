# RetryVLA: báo cáo tiến độ (cập nhật 2026-10-25)

> Bản tóm tắt cho người muốn nắm bức tranh lớn trong vài phút. Mọi con số lấy từ `experiments/RESULTS.md`,
> `experiments/NOTES.md`, `experiments/PLAN.md`, `runs/_analysis/*/summary.md` và lần chạy
> `scripts/analyze_baselines.py` lúc 2026-10-25 ~13:00 UTC (chỉ đọc). Chi tiết thống kê (CI, p) xem các file đó.
>
> Thuật ngữ: **RSR** = recovery success rate (tỉ lệ robot hoàn thành task khi bắt đầu từ một trạng thái thất bại,
> 56 test scenes x 10 trials, LIBERO-Recover Spatial). **Nominal** = success rate trên LIBERO-Spatial bình thường
> (10 tasks x 50 inits). **pi0** = SmolVLA gốc, **pi1** = SmolVLA sau khi distill recovery của RetryVLA.

## 1. TL;DR

- **Cái chạy được:** tìm recovery bằng world model (WM) + competence critic C^pi, kiểm chứng (verify) trong sim,
  rồi distill vào VLA: RSR **0.118 -> 0.418 ± 0.021** (pi1, 3 seeds), nominal **0.656 -> ~0.785-0.80**
  (giữ trong ngưỡng eps 2 pt so với pi1-nominal 0.796). Không cần thêm module nào lúc test.
- **"Competence" là yếu tố quyết định:** cùng pipeline, chỉ đổi mục tiêu search: competence SFT **0.417 vs goal 0.187**
  (3 seeds); progress-critic data **0.177 vs competence 0.430** (cùng số trajectory n=60). Trên LIBERO-Object (held-out,
  recipe đóng băng) competence >> goal trong search (66 / 8 failures, p = 2e-12).
- **Distill là cần thiết:** demo-only cùng init + số step chỉ 0.127; online best-of-8 trên pi1 chỉ +0.027 (n.s.);
  học từ imagination không verify (RISE-inspired) còn tệ hơn pi0 (0.068 / 0.082 / 0.113); residual RL với cùng sim
  budget đạt 0.295 nhưng nominal sụp còn **0.022**.
- **Cái chưa chạy được:** WM hiện chủ yếu là *bộ lọc / xếp hạng* trên proposal library (SmolVLA samples + primitives).
  Khi bỏ library và để WM tự plan (M11) thì kém hơn library-based search (0.398 vs 0.458 solved). Trên Object,
  WM data **không** tốt hơn random-search data ở cùng số trajectory (0.354 vs 0.362).
- **Câu hỏi B4 đã trả lời:** critic "success" (B4) thắng C^pi v2 là nhờ **nhãn continuation của pi0** (bỏ nhãn pi0 ->
  -0.12 solved, p = 0.004), tức nó vẫn là một competence critic tốt hơn -> ứng viên **C^pi v3**.

## 2. Pipeline trong một hình

![pipeline](progress_figs/pipeline_overview.png)

1. Lấy các **trạng thái thất bại thật** của pi0 (LIBERO-Recover, khôi phục chính xác snapshot sim).
2. **Đề xuất** nhiều đoạn hành động 50 bước (SmolVLA samples, perturbation, primitives, latent CEM).
3. **WM + competence critic** tưởng tượng kết quả và chấm điểm: "từ trạng thái này, pi0 có tự làm tiếp xong không?"
4. Chỉ những ứng viên tốt nhất được **verify trong simulator** (16 sim / failure); chỉ giữ thành công thật.
5. **Distill** các recovery đã verify (+50% demo) vào SmolVLA -> pi1, test RSR **không search** lúc chạy.

## 3. Timeline (M0 -> M11 + baselines)

| milestone | câu hỏi | trả lời | một con số |
|---|---|---|---|
| M0 | pi0 recover được không? | gần như không | RSR 0.118, nominal 0.656 |
| M1 | recovery từ oracle sim search có distill được không? | có, ~x2 RSR | 0.286 (oracle-CEM), 0.279 (random) |
| M2 | train latent WM (DINOv2) + critic | dùng được, critic over-confident | critic AUC 0.71 real / 0.67 imagined |
| M3 | WM + critic có tìm recovery tốt hơn random (16 sim)? | có, trên 53 hard failures | solved 0.36 vs 0.17 |
| M4 | thêm 45 WM recovery có giúp VLA? | chưa đo được (liều quá nhỏ) | 0.298 vs 0.293 |
| M5 | WM data vs random data, cùng số lượng (241) | WM tốt hơn | 0.402 vs 0.286 (+0.116) |
| M5b/c | so sánh search arms (256 failures) + sửa bug chọn subset | WM diverse thắng; round-robin = **pi1** | solved 0.547 vs 0.363; pi1 0.439 (seed 0) |
| M6 | vòng lặp pi1 -> pi2 | không hơn train thêm | pi2 0.463 vs control 0.480 |
| M7 / M8 | task yếu (stove, cookie box) | search chạy trên train nhưng không transfer; augmentation transfer nhưng lệch data | M8 overall 0.457 vs 0.504 |
| M9 | cơ chế: vì sao competence, vì sao WM, vì sao distill; 3 seeds | competence >> goal; WM ≈ real-dynamics với ~1/3 transitions | pi1 0.418 ± 0.021; comp 0.417 vs goal 0.187 |
| M10 | LIBERO-Object held-out, recipe đóng băng | distill + competence replicate; "WM data > random" không replicate | comp 0.354 vs demo-only 0.173 |
| M11 | WM có tự plan được không cần library? | chưa: kém library | wm_plan_v2 0.398 vs wm_diverse 0.458 solved |
| B2/B3/B4 (P2) | gain đến từ policy-grounding hay "bất kỳ value nào"? | policy-grounding; progress critic < random | progress 0.213 vs C^pi 0.480 solved |
| B3.2 | học từ imagination (RISE-inspired), không verify? | tệ hơn pi0 | RSR 0.068 / 0.082 / 0.113 |
| B3.3 | residual RL, cùng sim budget 3.47M | học được recovery nhưng phá nominal | RSR 0.295, nominal 0.022 |
| P3 | Pareto recovery-nominal (ratio, L_preserve) | ratio 0.5 là tốt nhất trong ràng buộc eps; L_preserve không cần | r 0.7: RSR 0.436 nhưng nominal 0.766 (< eps) |

## 4. Kết quả chính

Bảng baseline (Spatial test = **dev set** từ M7; 1 seed nếu không ghi khác). "Thêm lúc test?" = cần module thêm khi
deploy. Sim budget = số env transitions để tạo tín hiệu train.

| method | seeds | test RSR | nominal | thêm lúc test? | sim budget |
|---|---|---|---|---|---|
| pi0 (SmolVLA base) | 1 | 0.118 | 0.656 | không | 0 |
| pi1-nominal (demo SFT, init của mọi SFT) | 1 | 0.138 | 0.796 | không | 0 |
| demo-only (+6k steps) | 1 | 0.127 | 0.820 | không | 0 |
| random-data SFT (208) | 1 | 0.252 | 0.752 | không | 1.40M |
| goal-data SFT (66) | 3 | 0.187 ± 0.002 | 0.753 | không | 2.09M |
| progress-data SFT (60) | 1 | 0.177 | 0.774 | không | 2.09M |
| competence SFT n=60 | 1 | 0.430 | 0.798 | không | 2.90M |
| competence SFT n=66 | 3 | 0.417 ± 0.011 | 0.784 | không | 2.90M |
| **RetryVLA pi1 (208 verified WM recoveries)** | 3 | **0.418 ± 0.021** | **0.785** | **không** | 3.47M |
| online best-of-8 trên pi1 | 1 | 0.466 (+0.027, p 0.27) | not run | 8 VLA samples + 8 WM rollouts + critic / replan | 3.47M + 0 |
| online best-of-8 trên pi0 | 1 | 0.118 | not run | như trên | 2.21M |
| RISE-inspired cond (pool) | 1 | 0.068 | 0.718 | không | 2.21M (0 verify) |
| RISE-inspired AWR (pool) | 1 | 0.082 | 0.776 | không | 2.21M (0 verify) |
| RISE-inspired cond (VLA samples) | 1 | 0.113 | 0.770 | không | 2.21M (0 verify) |
| residual RL p25 | 1 | 0.111 | 0.400 | DINOv2-small + MLP | 0.87M |
| residual RL full | 1 | 0.295 | 0.022 | DINOv2-small + MLP | 3.47M (= PoC) |
| success-data SFT (B4) | - | pending | pending | không | 2.90M + nhãn B4 |

![main results](progress_figs/main_results.png)

Ý chính: trong các phương pháp **không cần thêm gì lúc test**, chỉ data được search bằng competence + verify trong sim
đạt mức RSR của pi1 mà vẫn giữ nominal trong eps.

**Search:** WM diverse tìm được recovery cho nhiều failure hơn random với cùng 16 sim, và ~2.8x thành công / sim (M5b):

![M5b search](progress_figs/m5b_search.png)

**Ví dụ end-to-end** (test scene drawer, không search lúc chạy): pi0, pi1-nominal, pi1-random đều 0/5; pi1-WM 3/5
(nhấc ra khỏi drawer rồi grasp lại):

![e2e](progress_figs/m5_e2e_test_keys.png)

**P3 Pareto** (seed 0; điểm r 0.3 nominal vẫn pending): tăng recovery ratio tăng RSR (0.323 / 0.375 / 0.418 / 0.436 cho
r 0.1 / 0.3 / 0.5 / 0.7) nhưng r 0.7 làm nominal rơi xuống 0.766 (-0.030 vs pi1-nominal, vượt eps). L_preserve:
RSR 0.429 / nominal 0.806, cả hai nằm trong độ dao động seed của pi1 -> không cần.

![P3](progress_figs/p3_pareto.png)

## 5. Ba câu hỏi nghiên cứu

| câu hỏi | trạng thái | bằng chứng chính |
|---|---|---|
| **Q1 Vì sao competence C^pi** (thay vì goal latent / progress / value chung)? | **proven** (Spatial, 3 seeds) + replicate search trên Object | search: competence vs goal 32/2 failures (p = 7e-8, Spatial), 66/8 (p = 2e-12, Object); SFT: 0.417 vs 0.187 (+0.230 [+0.161, +0.298]); progress critic < random trong search (0.213 vs 0.339), progress SFT 0.177 vs 0.430; B4 thắng nhờ nhãn pi0 (B4-noPi0 -0.12, p = 0.004). Object SFT comp - goal +0.112, p = 0.08 (1 seed: directional) |
| **Q2 Vì sao cần WM**? | **partial** | WM = hiệu quả acquisition: 3.1x (Spatial) / 2.2x (Object) hits / sim so với random; WM ≈ real-dynamics prefix (+0.033 [-0.049, +0.115]) với ~1/3 transitions (5.0k vs 16.5k). Nhưng: WM chưa tự plan tốt (M11), và WM data không hơn random data ở cùng số lượng trên Object |
| **Q3 Vì sao distill** (thay vì search online / RL / học từ imagination)? | **proven** | distill vs demo-only +0.292 (Spatial) / +0.181 (Object); best-of-8 online chỉ +0.027 n.s.; RISE-inspired < pi0 (nhãn tưởng tượng ~4/5 sai trên positives: V 0.81 vs real 0.185); residual RL 0.295 nhưng nominal 0.022 |

## 6. Cái KHÔNG chạy được / hạn chế (thành thật)

- **WM chủ yếu là bộ lọc trên proposal library.** Bỏ primitives (wm_no_prim) giảm -0.175 solved; WM tự plan từ noise
  (wm_plan_v2) vẫn kém library -0.059 (n.s.) và -0.64 hits / instance (p = 0.02); successes của nó giật nhất, xa
  manifold SmolVLA nhất. Retrain WM + critic trên data hành động rộng chỉ giúp +0.025 (n.s.).
- **WM data ≠ tốt hơn random data ở cùng số lượng trên Object** (0.354 vs 0.362, p = 0.93). Claim phải đổi thành
  "acquisition efficiency" (nhiều recovery hơn mỗi sim), không phải "data chất lượng hơn".
- **Nhiều baseline chỉ 1 seed** (random / progress / RISE / RL / best-of-8 / P3 / mọi số Object). Chênh < ~4 pt RSR là
  noise. Nominal trên Object (0.942 vs 0.944) cũng chỉ 1 seed, chưa có eps-test.
- **Spatial test là dev set** từ M7 (đã dùng test để chọn M7 / M8). Số "sạch" duy nhất là Object (held-out, recipe đóng băng).
- **Vòng lặp tự cải thiện (M6) không có gain** (critic cũ, data trùng task đã giỏi); task yếu (stove, cookie box) vẫn thấp.
- **Hạ tầng làm chậm:** ít nhất 6 lần /pfss EDQUOT (quota workspace dùng chung), mất tmux server nhiều lần, MuJoCo
  segfault trong sim worker, CUDA OOM trên GPU dùng chung, lỗi TMPDIR dài -> nhiều SFT phải train lại từ đầu. Code đã
  được vá (retry checkpoint, resume, respawn worker), chi tiết trong NOTES.

## 7. Bước tiếp theo

- Hoàn tất các eval đang pending: P3 r 0.3 nominal, B2 success-data SFT (RSR + nominal).
- **C^pi v3** = trainer / label pool kiểu B4 (chỉ nhãn continuation của policy sẽ chạy tiếp), sau một held-out check.
- **Thêm seeds:** 3 seeds cho các baseline 1-seed quan trọng và cho Object (eps-test nominal).
- **P4** matched-sim-budget distillation trên Object (dùng toàn bộ bank mỗi arm mua được với cùng budget) -> xem
  acquisition efficiency có thành policy tốt hơn không. **P5** cost accounting end-to-end (đang partial).
- **Generality của planner:** VLA backbone thứ hai và/hoặc WM thứ hai (claim "general" chỉ sau đó).
- **Suites:** LIBERO-Goal và LIBERO-10 với recipe đóng băng; breakdown L1-L4 của LIBERO-Recover.
- Làm WM thực sự **plan** (giảm phụ thuộc library) hoặc chứng minh coverage bottleneck trước khi thêm proposal head.
- Real-world adaptation: **chưa quyết định**.

## 8. Dung lượng

Đo bằng `timeout 1200 du -sh` (chạy song song, 2026-10-25 12:56-13:16 UTC). /pfss rất chậm: `du` trên `data` và trên
toàn repo **bị timeout sau 20 phút** (kẹt ở `data/assets`, cây scene LIBERO nhiều file nhỏ; một lần thử thêm 300 s
cũng timeout), nên hai dòng đó là **cận dưới** = tổng các phần đã đo được.

| path | dung lượng |
|---|---|
| &nbsp;&nbsp;data/recovery | 93G (4,045 files, `find -type f`) |
| &nbsp;&nbsp;data/wm | 55G |
| &nbsp;&nbsp;data/rl | 692M |
| &nbsp;&nbsp;data/splits | 97K |
| &nbsp;&nbsp;data/assets | timeout (2 tarball = ~63 MB + thư mục giải nén, chưa đo được) |
| RetryVLA/data | timeout; **≥ ~149G** |
| RetryVLA/runs | 85G |
| RetryVLA/slides | 89M |
| **RetryVLA tổng** | timeout; **≥ ~234G** (data ≥ 149G + runs 85G + slides; chưa tính code / env / assets) |

Phần lớn là recovery datasets (LeRobot, ảnh PNG) và WM latents; checkpoints SFT trung gian trong `runs/` (~2.3 GB / run)
là thứ dễ dọn nhất nếu quota lại đầy. (NOTES ngày 10-24 ghi runs 61 GB + data 145 GB -> runs đã tăng ~24 GB.)
