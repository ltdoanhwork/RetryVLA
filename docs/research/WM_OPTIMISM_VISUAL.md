# WM over-optimism, nhìn bằng mắt (2026-11-26)

> Câu hỏi: khi WM-guided search chọn ra một recovery bridge, WM "nhìn thấy" gì mà tự tin là sẽ thành công, trong khi sim nói thất bại?
> WM của chúng ta là **latent** (WM v1: DINOv2-small 6x6 pooled patch tokens x 2 camera (agent + wrist) + 1 robot-state token;
> V-JEPA 2-AC adapter) và **không có pixel decoder**, nên phải "nhìn" gián tiếp qua: (a) cùng ứng viên chấm trên latent mơ vs latent THẬT,
> (b) frame thật gần latent mơ nhất, (c) heatmap lỗi từng token, (d) một decoder latent→pixel huấn luyện riêng (chỉ là công cụ hiển thị).
> Dữ liệu: train split. Không có sim mới cho số liệu: chỉ replay bit-exact các bridge open-loop đã verify (9 instance x 64) để lấy frame.

Ký hiệu: **C^pi(mơ)** = critic v2 trên latent cuối do WM v1 tưởng tượng; **C^pi(thật)** = cùng critic trên latent DINOv2 THẬT sau khi chạy
bridge trong sim; **SR thật** = kết quả thật của pi0 continuation. Lỗi WM = C(mơ) − C(thật); lỗi critic = C(thật) − SR.

## 1. Exploitation curve: càng tìm nhiều trong mơ, WM càng tự tin, còn thành công thật thì đứng yên

![exploitation curve](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f1_exploitation_curve.png)

RQ3 `wm_bud_l0`, 63 instance (paired), top-4 theo J = C^pi ở mỗi ngân sách tưởng tượng N:

| N | WM v1 C(mơ), online | V-JEPA C(mơ), online | WM v1 picks: C(thật) | SR thật WM v1 / V-JEPA | FP C(mơ) WM v1 / V-JEPA | FP nếu chấm trên latent thật |
|---|---|---|---|---|---|---|
| 8 | 0.08 | 0.12 | 0.09 | 0.036 / 0.063 | 0.72 / 0.71 | 0.77 |
| 32 | 0.31 | 0.36 | 0.19 | 0.103 / 0.119 | 0.71 / 0.70 | 0.61 |
| 128 | 0.49 | 0.61 | 0.20 | 0.087 / 0.119 | 0.83 / 0.83 | 0.70 |
| 512 | 0.72 | 0.79 | 0.29 | 0.107 / 0.139 | 0.85 / 0.84 | 0.63 |

- Từ N=8 lên 512, C(mơ) tăng 9x (0.08 → 0.72) còn SR thật gần như đứng yên (0.04 → 0.11). V-JEPA 2-AC giống hệt: nó cũng bị chính search của nó khai thác.
- Mũi tên đỏ là lỗi WM, mũi tên xanh là lỗi critic. Lỗi WM tăng theo N (−0.01 / +0.09 / +0.22 / +0.33) và lớn hơn lỗi critic (+0.05 / +0.09 / +0.12 / +0.18) ở mọi N ≥ 32.
- Lưu ý: dấu x đỏ chấm chấm là C(mơ) chạy lại offline (fp32, Mesa) trên đúng các pick đó (0.62 ở N=512, so với 0.72 online bf16). Hai mũi tên dùng giá trị offline này để cùng một tập ứng viên.

## 2. Winner's curse: search chọn đúng những ứng viên mà WM sai theo hướng lạc quan

![winner's curse](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f2_winners_curse.png)

| tập ứng viên | n | gap TB C(mơ) − C(thật) | P(gap > 0.5) | \|gap\| < 0.05 | SR thật |
|---|---|---|---|---|---|
| random (không chọn, verif_m5b) | 4096 | −0.00 | 0.02 | 92 % | 0.06 |
| top-4 @ N=8 | 252 | −0.01 | 0.03 | 87 % | 0.04 |
| top-4 @ N=32 | 252 | +0.09 | 0.15 | 69 % | 0.10 |
| top-4 @ N=128 | 252 | +0.22 | 0.23 | 62 % | 0.09 |
| top-4 @ N=512 | 252 | +0.33 | 0.37 | 44 % | 0.11 |
| wm_diverse (768 imagined) | 4096 | +0.46 | 0.47 | 42 % | 0.17 |

Trên ứng viên ngẫu nhiên, lỗi WM gần như đối xứng quanh 0 (WM "tốt ngang sự thật" cho critic). Chọn top-k thì đuôi phải (gap ≈ +1:
mơ thấy thành công 100 %, thật 0 %) phình lên. Đó là winner's curse: không cần WM xấu đi, chỉ cần max trên nhiều ước lượng nhiễu.
(RQ3 không có budget 64; các budget có là 8 / 32 / 128 / 512.) V-JEPA (đường tím, critic riêng, cùng ứng viên) cũng có đuôi phải như vậy.

## 3. Gallery: 10 false positive (lỗi WM), 2 false positive (lỗi critic), 3 true positive

Mỗi hàng: frame start | frame cuối THẬT (agent, wrist) | heatmap lỗi token ‖z_mơ − z_thật‖² trên lưới 6x6 (một thang chung cho mọi hàng của mỗi camera) |
frame THẬT gần latent mơ nhất (lấy từ ngân hàng gồm mọi frame của mọi bridge đã verify trong cùng cảnh; bỏ trajectory của chính ứng viên) |
**DECODER(latent mơ)** agent / wrist | DECODER(latent thật) để xem decoder trung thực tới đâu. Caption gồm arm, họ ứng viên (heuristic từ action),
số lần đổi dấu gripper, C(mơ) / C(thật), kết quả sim, và trạng thái privileged của sim ở cuối bridge (grasped, lift, d(gripper, bowl)).

![gallery 1](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f3_gallery_1.jpg)
![gallery 2](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f3_gallery_2.jpg)
![gallery 3 (contrast)](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f3_gallery_3.jpg)

Đọc gallery:
- **Ở cả 10 FP lỗi WM, C(mơ) ≥ 0.999 còn C(thật) ≤ 0.002**, nhưng sim nói bát chưa bao giờ được cầm (grasped any = 0; gripper cách bát 6–35 cm).
- **Giấc mơ trông gần như sự thật.** Với 8/10 FP, latent cuối THẬT của chính ứng viên nằm trong top-4 frame gần latent mơ nhất (rank 0–3 / ~700),
  và frame thật gần nhất (bridge khác) gần như luôn là một frame THẤT BẠI. Decoder(latent mơ) cũng ra cảnh gần giống frame thật.
  Tức là WM không bịa ra một cảnh "bát trên đĩa"; nó sai một chút, nhưng sai đúng theo hướng mà critic thưởng.
- **Phần sai nằm ở camera cổ tay.** Lỗi token wrist của FP là 0.22–1.06, còn agent chỉ 0.06–0.17. Ảnh decoder wrist bị mờ và bát bị dịch chỗ so với ảnh thật,
  nhưng ở độ phân giải này decoder không cho thấy rõ "cú grasp tưởng tượng". Ở agent view, lỗi tập trung quanh gripper / cánh tay (#8, #9: tư thế tay mơ khác tay thật).
- FP lỗi critic (#11, #12): C(thật) = 1.00 mà vẫn fail. Ở đây chính trạng thái thật đã đánh lừa critic (gripper ngay cạnh bát nhưng chưa cầm), WM không có lỗi gì.
- Decoder là **bản tái tạo**, không phải WM render. Nó được train trên 11k cặp (frame, latent) thật của 9 cảnh này; các trajectory trong gallery được giữ ngoài tập train
  (val L1 0.014 trên [0, 1]). Ảnh decoder chỉ cho biết một decoder học trên frame thật đọc được gì từ latent, độ phân giải 6x6 token nên mờ.

## 4. Nguyên nhân, bằng số (verif_m5b: 256 instance x 64 ứng viên; random = 4096, search-selected = 12288)

![causes](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f4_causes_family_event.png)
![horizon + tokens](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f5_horizon_tokens.png)
![off-manifold](/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/docs/research/figs/wm_optimism/f6_offmanifold.png)

**(a) Search pressure là điều kiện cần.** Ở ứng viên random, lỗi WM ≈ 0 với mọi họ (|gap| ≤ 0.02). Chỉ khi đã bị chọn thì lỗi mới xuất hiện,
và độ lớn của nó phụ thuộc họ ứng viên:

| họ (search-selected) | n | lỗi WM | lỗi critic | FP C(mơ) |
|---|---|---|---|---|
| perturbation (pi0 + smooth noise, 25 % đảo gripper) | 1109 | **+0.69** | +0.06 | 0.88 |
| pi0 sample, noise 1.5 | 1957 | +0.57 | **+0.21** | 0.88 |
| primitive + CEM mutation | 2970 | +0.48 | +0.06 | 0.77 |
| pi0 sample, noise 1.0 | 4281 | +0.19 | +0.07 | 0.82 |
| primitive (nguyên bản) | 1971 | +0.11 | +0.05 | 0.69 |

Ứng viên càng "lạ" so với dữ liệu WM được huấn luyện (nhiễu, rung, nhiều lần đổi gripper), search càng dễ tìm được một ứng viên mà WM mơ ra thành công.
Primitive nguyên bản (mượt, ít trạng thái) bị khai thác ít nhất. Họ là heuristic từ action (prim = cấu trúc primitive chính xác; độ nhám bậc 2 tách
pi0 s1.0 / s1.5 / perturb, hiệu chỉnh trên nhãn collect_v1); với ứng viên của wm_diverse / wm_qd (đã bị CEM biến đổi), nhãn chỉ là gần đúng.

**(b) Gripper / grasp.** Selected có > 6 lần đổi dấu gripper: lỗi WM +0.51, lỗi critic +0.18. 0 lần đổi: +0.35 / +0.02.
Theo trạng thái THẬT cuối bridge: **"đang cầm bát"** (n=403) có lỗi critic lớn nhất (+0.28, cộng thêm lỗi WM +0.43). Ở trạng thái đó, ngay cả latent
thật cũng đánh lừa critic (cầm được ≠ đặt được). Trong các FP lỗi WM, 49 % kết thúc xa bát và 45 % đã chạm / kẹp hụt; chỉ 5 % đang thực sự cầm bát.
WM tưởng tượng ra cú grasp không có thật.

**(c) Theo horizon.** Lỗi latent tăng đều theo thời gian (FP: 0.15 ở bước 5 → 0.36 ở bước 50). Mức này **gần bằng** ứng viên TN (0.16 → 0.35), nên
FP không có lỗi trung bình lớn hơn. Còn ΔC = C(mơ) − C(thật) của FP gần 0 tới bước ~25, rồi bật lên ở cuối: 0.10 / 0.17 / 0.24 / 0.42 / **0.95** ở
bước 30 / 35 / 40 / 45 / 50. Với TN thì ΔC luôn ≈ 0. Optimism tập trung ở 1–3 bước WM cuối cùng (critic được train trên trạng thái cuối; các bước giữa
chỉ để chẩn đoán).

**(d) Ở đâu trong ảnh.** Lỗi token TB cuối bridge: wrist 0.60 (FP) vs 0.50 (TN) vs 0.43 (TP); agent 0.12 / 0.125 / 0.09. Phần dư của FP nằm ở camera cổ tay,
nơi thể hiện quan hệ gripper–bát. Lỗi eef imagined vs thật như nhau (FP 2.8 cm, TN 2.9 cm), và critic(token mơ, state thật) = C(mơ) (báo cáo RQ3).
Vậy proprio không phải nguyên nhân.

**(e) Search pressure vs critic OOD.** Latent mơ **không** lạc khỏi đa tạp trạng thái thật. Khoảng cách tới frame thật gần nhất (cùng cảnh): latent mơ
0.272, latent thật 0.270, latent mơ của FP 0.273 (f6). Vậy nguyên nhân chính không phải critic bị đưa ra ngoài phân phối bởi latent "rác". Nguyên nhân là
áp lực chọn lọc tìm được những lỗi bình thường về độ lớn nhưng nằm đúng hướng nhạy của critic, tập trung ở wrist / gripper–bát và ở cuối horizon (một kiểu adversarial example
trong không gian latent). Critic OOD là thành phần thứ hai, thật nhưng nhỏ hơn. Lỗi critic trên latent thật tăng theo N (+0.05 → +0.18) và lớn nhất ở
trạng thái hiếm "đang cầm bát" và ở pi0 noise 1.5. Ước lượng chung vẫn như RESULTS 'WM vs true': khoảng 80–85 % gap đến từ WM khi bị chọn, phần còn lại từ critic.

## Hệ quả cho thiết kế
- Mọi verifier dùng argmax trên N ứng viên tưởng tượng cần hoặc (i) xác minh nhiều pick hơn / tính đến winner's curse (shrinkage theo N), hoặc (ii) một tín hiệu
  độc lập với hướng lỗi (cross-WM, sim check cuối, critic chỉ nhìn agent view).
- Phạt ứng viên rung / nhiều lần đổi gripper và ứng viên kiểu perturbation trước khi chấm (đúng hướng G2 `curated`), và đừng tin tưởng 1–2 bước WM cuối.
- Một WM tốt hơn trung bình (V-JEPA) không đủ: lỗi ở ứng viên random vốn đã ≈ 0; vấn đề là phần đuôi bị chọn.

## Tái lập
Scripts (`scripts/wm_optimism/`): `fig_curve.py` (f1, f2; data cũ) · `cache_stats.py` (WM rollout cho mọi ứng viên verif_m5b / true_end_rq3 / collect_v1 →
`runs/_analysis/wm_optimism/cache/{rows.parquet, tok_err.npy}`) · `scripts.replay_bridges --frames_out` (9 instance gallery, NVIDIA EGL, khớp valid mask 1.000) →
`gallery_cache.py` → `train_decoder.py` (GPU, 5000 bước, 7 phút) → `fig_gallery.py` (f3) → `fig_causes.py` (f4–f6). JSON: `docs/research/figs/wm_optimism/{curve,causes,cases,decoder_log}.json`.
Media đầy đủ: `/pfss/mlde/workspaces/mlde_wsp_IAS_SAMMerge/RetryVLA/runs/_analysis/wm_optimism/figs/`.
Giới hạn: gallery / decoder / off-manifold chỉ dùng 9 instance (576 ứng viên); họ ứng viên là heuristic; C^pi giữa horizon nằm ngoài phân phối train của critic.
