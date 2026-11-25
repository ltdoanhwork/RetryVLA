# Review 2026-11-25: RetryVLA trên protocol LIBERO-Goal của 2610.05151

Viết 2026-11-25 ~03:45 UTC cho buổi review chiều (UTC+7). Số liệu đến từ `runs/goal_rsi/g2.{json,md}` và
`experiments/RESULTS.md`; các rule đã pre-register ở `experiments/PLAN.md` trước khi có kết quả. Policy: M0 = pi0 3B +
LoRA (4k step, chọn trên val như paper). Recipe distill = của họ (M0 + 2k update, D0 + bank trộn đều, warmup 200,
cosine). Đánh giá: 100 val scene (1 trial / scene), 3 seed SFT (7, 8, 9).

## 1. Tóm tắt (đọc trung thực)

1. **Chưa tái tạo được +10 của paper trên nominal.** Arm tốt nhất đã đủ 3 seed (P2-3 A) bằng Original đúng 79.3. Paper
   đi từ 78 lên 88 (1 seed, McNemar p 0.076; khác split, khác framework).
2. **Data của mình có dạy recovery thật**, đo trên 168 failure state held-out của chính M0 (RSR_E8):
   - F0 (bank cũ, 47 % anchor): +13.5 pt [+6.0, +21.6], nhưng nominal −11.
   - P2-3 A (72 trajectory curated + continuation, 20 % anchor): **+10.1 pt [+5.0, +15.7] mà nominal không giảm**
     (79.3 = Original). Đây là arm đầu tiên thoả rule "recovery without nominal cost". Arm này được chọn post hoc
     trong nhiều arm, nên cần replicate (seed 10-12) trước khi claim.
3. **Prevention data D_P (P2-3 B vs A): inconclusive.** Nominal −4.0 [−13, +5], failure incidence −3.3 pt [−14, +8],
   RSR bằng nhau. D_P (alternative đã verify tại s^-) không giảm failure đo được.
4. **G2-phys đáng chú ý nhưng confounded.** 84 / 82 trên 2 seed (Original 82 / 77, Δ +3.5, CI chứa 0; seed 9 đang
   chạy). Filter vật lý chỉ giữ 18 / 132 trajectory, và 17 / 18 không có continuation (task xong ngay trong bridge),
   nên đó là data rất ít (1.6 %) và rất "sạch". Ba arm control đã pre-register và đang chạy.
5. **F1 diagnosis (11-24): không có bug pipeline.** Bridge-only open-loop (có gripper flip từ perturbation) dạy hành vi
   có hại; thêm continuation anchor thì hết hại (80 vs 63, 1 seed).

## 2. Bảng chính (val, 100 scene; Δ = scene x seed bootstrap 95 %)

| arm | recovery anchor (share) | s7 / s8 / s9 | mean | Δ vs Original | fixed / broken vs Orig | RSR_E8 (Δ) | fail. incidence* | first-attempt* |
|---|---|---|---|---|---|---|---|---|
| M0 (4k, chọn trên val) | - | 85 | - | - | - | 0.179 | 0.29 | 0.67 |
| **Original** (M0 + 2k trên D0) | 0 | 82 / 77 / 79 | **79.3** | - | - | 0.147 | 0.22 | 0.68 |
| F0: bank v1, 47 % | 34 k (47 %) | 65 / 76 / 63 | 68.0 | −11.3 [−23.3, +1.3] | 34 / 68 | 0.282 (+13.5) | - | - |
| F1: bank v1, bridge-only 4 % | 1.7 k (4 %) | 58 / 69 / 50 | 59.0 | −20.3 [−33.7, −6.3] | 23 / 84 | 0.205 (+5.4) | - | - |
| G2: curated onset + continuation, 36 | 4.9 k (11 %) | 67 / 72 / 79 | 72.7 | −6.7 [−18.3, +4.3] | 36 / 56 | 0.200 (+5.4) | 0.32 | 0.56 |
| **P2-3 A**: curated, 72 | 9.5 k (20 %) | 83 / 78 / 77 | **79.3** | +0.0 [−8.0, +8.0] | 38 / 38 | **0.248 (+10.1 [+5.0, +15.7])** | 0.30 | 0.65 |
| P2-3 B: curated 36 + D_P 36 | 8.5 k (18 %) | 75 / 71 / 80 | 75.3 | −4.0 [−12.7, +5.0] | 30 / 42 | 0.248 (+10.1) | 0.27 | 0.65 |
| G2-phys: 18 survivors | 0.6 k (1.6 %) | 84 / 82 / … | 83.0 | +3.5 [−5.0, +13.0] (2 seed) | 26 / 19 | 0.182 (+3.0) | 0.28 | 0.68 |
| 2610.05151 paper | 4.1 % teacher windows | 78 -> **88** | - | +10 (1 seed) | 18 / 8 | không đo | - | - |

\* Detector E-v1 (`retryvla/eval/goal_events.py`, frozen 11-24) chạy trên replay bit-exact của val episode: grasp_lost,
displaced, tipped, off_table, stall. **Chưa được kiểm bằng nhãn tay** (κ), nên chỉ là metric phụ.

Per task (mean 3 seed, nominal / 10):

| task | Original | G2 | P2-3 A | P2-3 B | G2-phys (2 seed) |
|---|---|---|---|---|---|
| open middle drawer | 9.7 | 3.7 | 8.0 | 8.3 | 10.0 |
| open top drawer + bowl | 8.0 | 8.0 | 7.0 | 4.7 | **2.5** |
| push plate to stove | 4.0 | 8.7 | **9.7** | 8.0 | 8.0 |
| bowl on plate | 8.7 | 7.0 | 8.3 | 8.3 | 8.0 |
| bowl on stove | 9.7 | 8.0 | 9.0 | 9.7 | 9.5 |
| bowl on cabinet | 9.0 | 7.7 | 8.0 | 9.7 | 9.5 |
| cream cheese in bowl | 7.0 | 7.0 | 7.0 | **3.3** | 8.0 |
| wine bottle on rack | 4.0 | 4.3 | 3.7 | 4.7 | **9.0** |
| wine bottle on cabinet | 9.3 | 8.3 | 8.7 | 8.7 | 8.5 |
| turn on stove | 10.0 | 10.0 | 10.0 | 10.0 | 10.0 |

Đọc nhanh:
- Recovery data giúp rõ ở 2 task Original yếu nhất: push plate 4.0 -> 8-10, và với G2-phys thêm wine rack 4 -> 9.
  Nhưng nó làm gãy task khác: middle drawer với G2, cream cheese với P2-3 B, top drawer với G2-phys (8.0 -> 2.5).
- Net nominal là tổng của gain trên task yếu trừ regression trên task đã ổn. Đây là kiểu forgetting cục bộ, không phải
  "data vô dụng".

## 3. Visual check (đã mở từng sheet; skill retryvla-visual-check)

G2 bank (curated onset, wm_diverse, shard 0, sample stratified 8):
`/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2_wm_s0/index.md`

![G2 most gripper flips ep26](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2_wm_s0/goal_search_onset_cur_s0_wm_diverse_ep26_sheet.png)
![G2 worst-jerk ep44](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2_wm_s0/goal_search_onset_cur_s0_wm_diverse_ep44_sheet.png)

> G2 (n = 8: 2 worst-jerk, 1 worst handoff, 1 flips, 4 per task):
> - Bản thân các bridge trong G2 khá mượt (curation đã bỏ được perturbation family), **nhưng phần continuation
>   của M0 thì không**.
> - ep26 (cream cheese): bridge 50 step ổn, sau đó M0 mất 280 step với **21 lần gripper flip**, lắc qua lại rồi
>   mới thành công ở step 331.
> - ep44 (bowl on plate): 9 flip và handoff jump 1.25.
> - ep35: handoff jump 1.59 (gripper open -> close đột ngột).
> - Verdict: success label đúng, nhưng các trajectory "verified" này dạy cả hành vi lóng ngóng của M0. Vì không có
>   validity mask nên chúng vẫn được imitate. Đây là giải thích hợp lý nhất cho cost nominal của G2 (−6.7) và cho
>   failure incidence cao hơn (`displaced` 9-17 / 100 vs 2-3 ở Original).

Physical filter, cái nó bỏ (worst theo từng lý do):

![rejected: seam / handoff jump ep35](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2phys_rej/goal_search_onset_cur_s0_wm_diverse_ep35_sheet.png)
![rejected: jerk ep18](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2phys_rej/goal_search_onset_cur_s0_wm_diverse_ep18_sheet.png)
![rejected: object jump ep40](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2phys_rej/goal_search_onset_cur_s0_wm_diverse_ep40_sheet.png)

Cái nó giữ:

![kept ep26 (s1)](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2phys_keep/goal_search_onset_cur_s1_wm_diverse_ep26_sheet.png)
![kept ep6 (s2)](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/visual_check/goal_g2/g2phys_keep/goal_search_onset_cur_s2_wm_diverse_ep6_sheet.png)

> Physical filter: những trajectory bị bỏ đúng là trông xấu.
> - ep35: gripper đóng / mở ở handoff (jump 1.59), M0 bắt đầu lại từ đầu.
> - ep18 (bowl on plate): bridge kéo EE lên cao 14 cm, quay ngược, rồi M0 tự làm lại toàn bộ task.
> - ep40 (cream cheese): object nhảy 5.8 cm / step khi không có finger contact (bị đẩy / văng).
> - Cái giữ lại là đoạn ngắn 17-31 step ở state đã gần goal, gripper mượt, task xong ngay trong bridge (ep6: thả
>   cream cheese vào bowl).
> - Hệ quả: filter đổi luôn *bản chất* data, từ "recovery + continuation" sang "đoạn kết thúc sạch". Nên so G2-phys
>   với G2-36 là confounded (chỉ 1.6 % anchor, gần như không có handoff). Seam rule (3 x median |Δa| của demo) là thủ
>   phạm chính (96 / 132 bị loại vì seam).

## 4. Trạng thái các thành phần mới

- **Critic swap:** CRIT-V `base` (`runs/20261124_CRITV_goal/ckpt/base_s0.pt`) đã được nối vào WMCriticRanker; mọi
  search mới sẽ dùng nó. G2 bank và D_P sinh trước đó dùng C^pi cũ (deviation; D_P không dùng critic).
- **FC labelling (a):** seed 7 / 8 có 52 / 53 rollout gán nhãn. Physical event đầu tiên sau t_last nằm trong cửa sổ onset
  ở 13 / 52 và 21 / 53 rollout (grasp_lost, tipped, displaced, stall). FC dài trung bình 23-26 step. D_F = 1,210 +
  1,400 action. Rule rebuild (s^f lệch t_on >= 20 step ở >= 30 %) KHÔNG đạt (10 % / 21 %), nên giữ entry tại t_on.
  D_F để dành cho hinge L_P và hard negative của critic.
- **Aux heads (c):** đã pre-register, code chưa viết. Chạy trên bank tốt nhất sau khi các follow-up G2-phys có kết quả.

## 5. Queue (orchestrator chạy ngay trên shell 953bb534, tmux `rv_g2_chain`, watchdog 5 lần)

Trước đó chain chạy ở local tmux của agent box, nhưng box đã đổi container nên chain chết 22:54 UTC 11-24, GPU idle
~4 h. Giờ nó chạy trên chính shell RetryVLA, dùng `det` từ trong shell.

| job | GPU | trạng thái | ETA (UTC) |
|---|---|---|---|
| g2phys_wm_s9 | 953bb534:0 | running | ~06:30 |
| g2_wm18_s7 / s8 (count control) | 953bb534:1, 8f473147:0 | running | ~06:30 |
| g2_rand_s7 | 8f473147:1 | running | ~06:30 |
| g2physns_wm s7-9 (phys filter không seam rule, 36 traj: 63 survivors / 28 entries) | queue | | ~10:00 |
| g2_wm5_s7 / s8 (anchor control ≈ G2-phys) | queue | | ~10:00-13:30 |
| g2_rand s8 / s9, d5_wm s8 / s9 | queue | | ~13:30 |
| AUX x 3 (sau khi có code + target) | chưa queue | | ~11-26 |

## 6. Bước tiếp theo (đề xuất cho review)

1. **Replicate P2-3 A** (72 trajectory curated, seed 10, 11, 12) vì nó đang là claim mạnh nhất (recovery +10 mà
   nominal giữ nguyên). Song song, quét share 36 / 72 / 108 để xem đường cong có đơn điệu không.
2. **Validity mask cho continuation.** Mask các target sau một khoảng ngắn từ handoff (ví dụ 20 step), hoặc lọc
   continuation có > 2 gripper flip. Lý do: visual check cho thấy phần continuation của M0 là nguồn hành vi xấu
   chính trong G2.
3. **Đọc G2-phys khi đủ 3 control** (no-seam @36, wm@5, wm@18). Nếu "data cuối-task sạch" là cái giúp, đó là
   hướng distill khác: dạy kết thúc chắc chắn, không dạy recovery dài.
4. **P2-3 với hinge L_P + D_F** (Algorithm 2 thật) thay cho việc chỉ thêm D_P làm positive. Rule giữ nguyên; chỉ
   chạy nếu P2-3 A replicate.
5. **Kiểm detector E-v1 bằng nhãn tay** (50 episode, κ >= 0.6) trước khi đưa failure incidence vào main table.
6. **Test split (100 scene) chỉ một lần**, với recipe đã đóng băng × 3 seed, so với Original × 3 seed và tham chiếu
   88 của paper.
