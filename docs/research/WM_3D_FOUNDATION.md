# WM_3D_FOUNDATION: point cloud / 3D / foundation-model features trong world model của RetryVLA

> Câu hỏi (2026-11-24): khi train WM để tưởng tượng (imagination), có cách nào đưa POINT CLOUD / 3D vào không, và có
> paper nào gắn các FOUNDATION MODEL khác vào WM để nó học / tưởng tượng tốt hơn không? Mục tiêu cụ thể: giảm lỗi tưởng
> tượng ở chỗ quan trọng (contact, grasp, object motion) trên các trạng thái OOD mà search chọn ra.
> Loại tài liệu: research + recommendation, **không train GPU**. Mọi paper ở bảng dưới đã kiểm tra trên arXiv API (ID, tiêu
> đề, tác giả, ngày v1) + ít nhất một nguồn thứ hai (proceedings / OpenReview / CVF / PMLR / trang project / GitHub / HF).
> Số liệu trích từ paper, không chạy lại. Liên quan: `CRITIC_VARIANTS.md` (3D / CoT trong critic), RESULTS `RQ3 follow-up`
> (WM vs true), `Cross-WM disagreement`, `RGR`, `M9`, `M11`, `T2Mem`.

## 0. TL;DR

1. **Đã đo trong repo, trước khi đọc paper**: lỗi nằm ở *image tokens tưởng tượng* trên các pick do search chọn (gap WM
   +0.28..+0.46, ~80-85 % độ lạc quan), KHÔNG phải ở encoder: (a) trên TRUE state, chính DINOv2 tokens đã đọc ra được
   contact / grasp / goal atom với AUC 0.86-0.96 (CRIT-V `cot` read-out), (b) cho critic biết point cloud thật của end
   state (`pc_true`) không giúp, (c) đổi sang foundation WM lớn hơn nhiều (V-JEPA 2-AC, ViT-g 1B) vẫn bị search của chính
   nó khai thác y hệt (FP ~0.83), (d) hai WM khác nhau có lỗi tương quan (khi cả hai nói thành công, pick vẫn fail 58-74 %).
   => Đưa 3D / foundation features vào **input** hay **critic** là đòn bẩy yếu. Cái cần là làm cho **dynamics** của WM đúng
   ở vùng search khai thác.
2. **Trần (ceiling) đã đo**: tưởng tượng hoàn hảo cũng không mua được nhiều solved rate. M9 `sim_prefix_critic` (end state
   THẬT + C^pi) solved 0.443 vs `wm_diverse` 0.475 (n.s.); xếp hạng theo true end state chỉ nâng p@4 0.193 -> 0.237; một
   linear recalibrator trên embedding tưởng tượng đã gần bằng mức đó (hits 3.14 vs 3.17 / 16 verif). Mọi phương án WM ở
   đây phải được so với **recalibrator** và đo bằng **hits / sim**, không chỉ solved.
3. **Literature (2023-2026)**: 3D giúp *chất lượng dự đoán* và *policy* (DepthWorld +1.48 dB PSNR nhờ depth supervision;
   GeoMoLa bỏ pointmap -> RLBench 84.7 -> 79.6 %; Spatial Forcing align với VGGT 96.9 % vs DINOv2 94.1 % vs không align
   92.7 % LIBERO), nhưng **không paper nào đo lỗi tưởng tượng trên trạng thái do planner chọn** - đúng chỗ mình bị. Hai bài
   gần nhất với bottleneck của mình là về **data / training của WM**, không phải features: Reinforced Planning
   (2608.18669) sửa hallucination 'grasp-and-miss' bằng một vòng Dyna fine-tune WM trên rollout thất bại của planner
   (+2.3..+18.2 điểm); còn Train-Test Gap (2512.09929, ICLR 2026) thấy DAgger cho WM (OWM) **không** giúp gradient
   planning, adversarial training (AWM) mới giúp. Generic 3D WM không chuyển zero-shot sang LIBERO: theo bảng của ORD-WM,
   PointWorld zero-shot sai 321 mm (điểm di chuyển, full trajectory), fine-tune 226 mm, ORD-WM 37.6 mm.
4. **Khuyến nghị**: E1 = 2x2 (+ control) **WM-DAgger (train trên bridge do search chọn, đã verify) x auxiliary 3D heads
   (contact / grasp / object flow từ sim, áp lên latent rollout open-loop)**, đo bằng *gap WM trên pick của chính WM đó* (CPU
   replay, không cần policy) trước khi verify online. E2 (rẻ, ưu tiên thấp hơn) = mô hình 3D object-centric rigid nhỏ
   (ORD-WM-lite, pose object từ sim) làm imagination thứ hai có lỗi *ít tương quan* hơn V-JEPA, test offline trên log
   có sẵn. Không khuyến nghị: đổi encoder (RADIO / VGGT / Theia), PointWorld zero-shot, SAM 2 slots trong sim.
5. Ghi chú quan trọng cho sim: trong LIBERO, các 'teacher' foundation (Depth Anything, CoTracker / TAPIR, SAM 2, VGGT)
   **kém hơn ground truth của sim** (depth, segmentation, pose object, contact). Chúng chỉ đáng dùng khi tính đến chuyển
   sang robot thật; cho câu hỏi khoa học ở đây, dùng nhãn sim (đã có trong `data/critic3d`).

## 1. Bằng chứng nội bộ quyết định thiết kế (những gì 3D / foundation features KHÔNG thể sửa)

| quan sát (nguồn) | số | hệ quả cho câu hỏi 3D |
|---|---|---|
| RQ3 step 1 (RESULTS 'RQ3 follow-up') | gap WM (a-b) +0.28..+0.46 trên pick; random candidates a ~ b ~ SR (AUC 0.83 vs 0.85) | lỗi chỉ xuất hiện dưới **áp lực search**; latent MSE trung bình trên pick KHÔNG lớn hơn (0.32 vs 0.34) - search tìm lỗi *có hướng*, không phải lỗi lớn |
| RQ3 high-budget | gap WM -0.01 / +0.09 / +0.22 / +0.33 tại N = 8 / 32 / 128 / 512 | lỗi tăng theo budget tưởng tượng = hiện tượng exploitation (Goodhart), không phải thiếu capacity biểu diễn |
| CRIT-V `cot` read-out | contact / grasp / goal atom AUC 0.86-0.96 trên TRUE tokens, 0.72-0.91 trên tưởng tượng | DINOv2 tokens **đã mang** thông tin contact; thiếu là ở dự đoán |
| CRIT-V `pc_true` (cloud thật của end state) | Goal imag dAUC -0.012, Spatial +0.003 | upper bound của '3D đưa vào critic' = ~0 |
| RGR geometry head | AUC 0.86 trên latent thật, 0.61 trên latent tưởng tượng; filter cắt FP 0.83 -> 0.67 nhưng solved không tăng | imagined latent mất hình học; veto hình học tăng precision, không tăng solved |
| V-JEPA 2-AC (ViT-g 1B, DROID) | offline tốt hơn trên pick của WM v1 (FP 0.655 vs 0.789) nhưng search của chính nó: FP ~0.83, gap ~0.65 tại N = 512 | encoder lớn hơn / foundation hơn không chống được exploitation; đánh giá trên pick của WM khác bị **independence effect** |
| xwm (WM v1 x V-JEPA) | min-agree hits/sim 0.171 vs 0.146 (n.s.); P(fail, khi cả hai > 0.5) 0.58-0.74 | lỗi hai WM tương quan; muốn veto có ích cần mô hình có **cấu trúc khác** |
| M11 WM v2 (broad-action data) | +0.025 solved, +25 % hits/sim, n.s. | thêm data *rộng* không đủ; câu hỏi mở là data *do search chọn* |
| M9 `sim_prefix_critic` | solved 0.443 vs 0.475 (`wm_diverse`) | trần của 'WM hoàn hảo' trên solved ở budget cố định là nhỏ |
| T2Mem lead | linear recalibrator hits 3.14 vs true-end ranking 3.17 vs static 2.64 | baseline rẻ mà mọi WM mới phải vượt |

Kết luận phần 1: câu hỏi đúng không phải 'feature nào giàu hơn' mà là 'can thiệp nào làm **dynamics trên vùng search
khai thác** đúng hơn, hoặc làm lỗi của imagination thứ hai **ít tương quan** hơn'. Literature dưới đây được đọc theo
lăng kính đó.

## 2. Bảng paper đã kiểm tra

Cột 'kiểm tra' = nguồn thứ hai ngoài arXiv API. 'Code / weights' kiểm tra trên GitHub / HF lúc viết. Compute = theo paper /
README nếu có, '-' nếu không nêu.

### 2A. 3D / point-cloud world models và 3D-aware dynamics

| paper | arXiv | venue (kiểm tra) | code / weights | biểu diễn / cách tích hợp 3D | kết quả liên quan (số từ paper) | liên quan tới bottleneck của mình |
|---|---|---|---|---|---|---|
| **PointWorld** (Huang ... Fei-Fei, NVIDIA) | [2601.03782](https://arxiv.org/abs/2601.03782) | CVPR 2026 Highlight (CVF OA + supp.) | [NVlabs/PointWorld](https://github.com/NVlabs/PointWorld), [HF nvidia/PointWorld_models](https://huggingface.co/nvidia/PointWorld_models) (small / large DROID, large DROID+BEHAVIOR; NVIDIA Open Model License); DINOv3 ViT-L gated | RGB-D -> point cloud, feature DINOv3 (frozen, multi-layer) chiếu lên điểm, backbone PTv3; action = robot point flow qua URDF; dự đoán 3D flow mọi điểm, 10 bước x 0.1 s | ~2M traj / 500 h; MPC zero-shot trên Franka thật (push, deformable, articulated, tool) | trên LIBERO **không zero-shot được**: ORD-WM đo full-traj moving-point l2 321 mm (zero-shot), 226 mm (fine-tune LIBERO). Gripper DROID = Robotiq, mình = Panda |
| **ORD-WM** (Weng ... Li, HIT Shenzhen) | [2610.23156](https://arxiv.org/abs/2610.23156) | arXiv preprint (project page; paperswithcode ghi CoRL 2026, chưa thấy trên trang chính thức) | chưa thấy code (project page chỉ có video) | object point cloud -> VN-VAE (SO(3)-equivariant, SDF), dynamics transformer dự đoán SE(3) increment per object, áp giải tích; aux head khoảng cách object-gripper | **train trên LIBERO** (4 suite + 1k push): 1-step / 10-step / full-traj moving l2 1.58 / 7.71 / 37.6 mm vs PointWorld-ft 10.04 / 79.62 / 225.7 mm; MPC CEM real-time | ứng viên tốt nhất cho 'imagination 3D thứ hai' trên LIBERO; cấu trúc rigid ràng buộc không gian giả thuyết -> khó bị search khai thác theo cách 'object tự bay vào gripper' |
| **DepthWorld** (Bardhan, Sivic, Petrik) | [2610.08780](https://arxiv.org/abs/2610.08780) | CoRL 2026 (arXiv comment + LinkedIn tác giả) | [Jai2500/depthworld](https://github.com/Jai2500/depthworld) MIT; [HF jaibrdhn/depthworld](https://huggingface.co/jaibrdhn/depthworld) (SVD license; bản DPT-VGGT là CC-BY-NC) | SVD video WM dự đoán RGB + metric depth (tile cạnh nhau, VAE giữ nguyên), tùy chọn DPT pointmap head init từ VGGT | depth supervision **cải thiện cả RGB**: +1.48 dB PSNR vs RGB-only cùng budget | bằng chứng sạch nhất rằng '3D làm target phụ' giúp dự đoán; nhưng là pixel video WM (đắt), không đo dưới search |
| **GeoMoLa** (Zhang ... Guibas, Yang) | [2607.04714](https://arxiv.org/abs/2607.04714) | ICML 2026 (icml.cc poster 62176; PMLR v306 theo RELATED_WORK K3) | chưa thấy | motion latent rời rạc học bằng cách dự đoán pointmap tương lai | ablation: bỏ pointmap CALVIN 3.60 -> 3.38, RLBench 84.7 -> 79.6 %; real 53.3 vs 35.0 % | hỗ trợ 'target 3D tương lai' > 'target RGB' cho latent động lực; là policy, không phải WM để search |
| **3D-VLA** (Zhen ... Gan) | [2403.09631](https://arxiv.org/abs/2403.09631) | ICML 2024 (PMLR v235) | có (project page) | 3D-LLM + diffusion sinh goal image / goal point cloud | held-in reasoning / generation | goal-generation, không phải action-conditioned dynamics |
| **ManiGaussian** / **ManiGaussian++** | [2403.08321](https://arxiv.org/abs/2403.08321) / [2506.19842](https://arxiv.org/abs/2506.19842) | ECCV 2024 (ECVA) / arXiv | có | Gaussian world model làm loss phụ 'future scene reconstruction' cho policy | RLBench 10 task +13.1 % SR vs SOTA | aux 3D future loss cho policy, không đo imagination |
| **GWM** (Lu ... ) | [2508.17600](https://arxiv.org/abs/2508.17600) | ICCV 2025 (arXiv comment) | project page | 3D Gaussian VAE + latent DiT dự đoán propagation của Gaussian theo action | dùng làm loss phụ cho IL và neural simulator cho MBRL | 3D WM tổng quát; nặng (Gaussian splatting) |
| **GaussianDream++** | [2608.25659](https://arxiv.org/abs/2608.25659) | arXiv | - | world tokens trong VLA, head Gaussian current + future chỉ lúc train | LIBERO 98.6 %, LIBERO-Plus 87.8 % | training-only 3D head (giống option ii) nhưng cho policy |
| **TesserAct** | [2504.20995](https://arxiv.org/abs/2504.20995) | ICCV 2025 (CVF OA) | project page | video WM sinh RGB + depth + normal -> 4D scene | IDM tốt hơn video WM 2D | 3D làm output đồng thời |
| **Aether** | [2503.18945](https://arxiv.org/abs/2503.18945) | ICCV 2025 (CVF OA) | project page | video WM học chung 4D reconstruction + action-conditioned prediction + planning | zero-shot synthetic -> real | như trên, quy mô lớn |
| **ParticleFormer** (Huang ... Schwager) | [2506.23126](https://arxiv.org/abs/2506.23126) | CoRL 2025 (PMLR v305) | project page | transformer point-cloud WM, loss Chamfer hybrid, multi-material | tốt hơn GNN particle baselines về rollout + MPC | particle dynamics; cần point cloud mỗi bước |
| **PCWM** (Peri ... ) | [2404.18926](https://arxiv.org/abs/2404.18926) | ICRA 2024 (arXiv comment) | project page | Dreamer-style point-cloud world model | robust hơn RGB-D khi đổi ánh sáng / camera; sample efficiency tốt hơn | 3D chủ yếu cho **robustness thị giác**, không phải contact accuracy |
| **Point-cloud JEPA WMs** (Oberweger ... ) | [2608.29434](https://arxiv.org/abs/2608.29434) | arXiv (open weights) | có (theo abstract) | 3 thiết kế JEPA trên point cloud raycast | point-cloud WM **ngang** image-based (không vượt) | phản chứng: 3D không tự nhiên tốt hơn cho latent planning |
| **JEPA4R** | [2611.19771](https://arxiv.org/abs/2611.19771) | arXiv | [AIGeeksGroup/JEPA4R](https://github.com/AIGeeksGroup/JEPA4R) | JEPA trên point cloud depth đơn, decoder 4D | lỗi điểm 1.1 cm trên 200 bước; MetaWorld 98.4 % vs R3M 96.0 % | bài mới (11/2026), chưa có so sánh dưới search |
| **3DFlowAction** | [2506.06199](https://arxiv.org/abs/2506.06199) | arXiv | - | video diffusion sinh 3D object flow theo ngôn ngữ, GPT-4o kiểm tra | cross-embodiment | flow-as-plan, không action-conditioned |
| **3D Diffuser Actor** | [2402.10885](https://arxiv.org/abs/2402.10885) | CoRL 2024 (project page) | có | policy trên 3D scene tokens | RLBench +16.3 % | policy 3D, không WM |
| **Track2Act** / **Im2Flow2Act** / **General Flow** | [2405.01527](https://arxiv.org/abs/2405.01527) / [2407.15208](https://arxiv.org/abs/2407.15208) / [2401.11439](https://arxiv.org/abs/2401.11439) | ECCV 2024 / CoRL 2024 / CoRL 2024 | có | dự đoán point tracks / object flow (2D hoặc 3D) làm interface | zero-shot transfer từ video người (General Flow 81 % trên 18 task) | flow-as-plan; ý dùng **object flow làm target phụ** |
| **AdaptiGraph** / **GBND** (3DGS tracking + GNN) / **RoboCraft** | [2407.07889](https://arxiv.org/abs/2407.07889) / [2410.18912](https://arxiv.org/abs/2410.18912) / [2205.02909](https://arxiv.org/abs/2205.02909) | RSS 2024 (chỉ theo search, chưa mở proceedings) / CoRL 2024 (PDF) / RSS 2022 (comment) | có | particle / graph neural dynamics cho deformable | MPC trên dây, vải, plasticine | object-centric particle dynamics; LIBERO chủ yếu rigid -> ORD-WM hợp hơn |
| **D3Fields** | [2309.16118](https://arxiv.org/abs/2309.16118) | CoRL 2024 oral (comment) | có | 3D descriptor field từ DINOv2 + SAM chiếu lên 3D, theo thời gian | zero-shot rearrangement | ví dụ 'foundation features trên điểm 3D' (như PointWorld) |

### 2B. Foundation-model features / teacher trong WM (hoặc trong policy có dự đoán tương lai)

| paper | arXiv | venue (kiểm tra) | code / weights | foundation model dùng thế nào | số liệu đáng chú ý |
|---|---|---|---|---|---|
| **DINO-WM** | [2411.04983](https://arxiv.org/abs/2411.04983) | ICML 2025 (PMLR v267) | [gaoyuezhou/dino_wm](https://github.com/gaoyuezhou/dino_wm) + ckpt | frozen DINOv2 patch tokens làm state | PushT SR: DINOv2 patch 0.90, DINO CLS 0.44, R3M 0.42, ResNet 0.20 -> **spatial patch** là cái quan trọng (mình đã dùng) |
| **V-JEPA 2 / 2-AC** | [2506.09985](https://arxiv.org/abs/2506.09985) | arXiv + Meta AI publication page | weights mở (MIT) | frozen ViT-g + AC predictor 300M trên 62 h DROID | pick-and-place zero-shot 80 % (cup) / 65 % (box) vs Octo 15 / 10, Cosmos 0 / 0; 16 s / action (RTX 4090) |
| **DINO-world** ('Back to the Features') | [2507.19468](https://arxiv.org/abs/2507.19468) | arXiv (Meta) | - | predictor tương lai trên DINOv2, pretrain trên video lớn, fine-tune action | tốt trên depth / segmentation forecasting; planning được |
| **DINO-Foresight** | [2412.11673](https://arxiv.org/abs/2412.11673) | NeurIPS 2025 (comment) | có | dự báo feature VFM tương lai | forecasting semantic |
| **Theia** | [2407.20179](https://arxiv.org/abs/2407.20179) | CoRL 2024 (comment) | có (theaiinstitute) | distill nhiều VFM (ViT, DINOv2, CLIP, SAM, Depth Anything) thành 1 encoder nhỏ | vượt teacher trên CortexBench (policy) |
| **AM-RADIO / RADIOv2.5** | [2312.06709](https://arxiv.org/abs/2312.06709) / [2412.07679](https://arxiv.org/abs/2412.07679) | CVPR 2024 (comment) / arXiv | NVlabs RADIO | agglomerative (CLIP + DINOv2 + SAM) | chưa thấy WM manipulation nào báo cáo gain do RADIO |
| **Spatial Forcing** | [2510.12276](https://arxiv.org/abs/2510.12276) | arXiv | [OpenHelix-Team/Spatial-Forcing](https://github.com/OpenHelix-Team/Spatial-Forcing) | align token trung gian của VLA với **VGGT** (cosine, chỉ lúc train) | LIBERO 98.5 vs OpenVLA-OFT 97.1; ablation: no-align 92.7, SigLIP 94.0, DINOv2 94.1, VGGT 96.9; 8 H100, 150k it |
| **Geometry Forcing** | [2507.07982](https://arxiv.org/abs/2507.07982) | ICLR 2026 (proceedings) | project page | align feature video diffusion với VGGT (angular + scale) | cải thiện chất lượng + 3D consistency, kể cả action-conditioned |
| **GWM-VLA** / **FOCAL-VLA** | [2608.07619](https://arxiv.org/abs/2608.07619) / [2609.21228](https://arxiv.org/abs/2609.21228) | arXiv / arXiv | FOCAL: project page | latent WM trên token VGGT đa view / distill VGGT + Track4World làm implicit WM | robustness dưới shift (policy) |
| **DreamVLA** | [2507.04447](https://arxiv.org/abs/2507.04447) | NeurIPS 2025 (theo search, chưa mở proceedings) | có | dự đoán dynamic region + depth + DINOv2 / SAM features tương lai | CALVIN 4.44; **ablation: depth, DINO hoặc SAM làm target phụ một mình không giúp, thường làm tệ đi**; dynamic-region (vùng sắp thay đổi) giúp nhiều nhất (3.64 -> 4.32) |
| **Dyn-O** | [2507.03298](https://arxiv.org/abs/2507.03298) | NeurIPS 2025 (proceedings PDF) | - | slots trên feature Cosmos tokenizer, **SAM 2 mask** dẫn slot attention rồi anneal bỏ mask | vượt DreamerV3 về rollout accuracy (Procgen) |
| **WorldDP** | [2606.08775](https://arxiv.org/abs/2606.08775) | arXiv | project page | slots trên DINOv2 patch, SAM 2 mask làm supervision (Tversky), WM + diffusion policy | báo vật nhỏ (cube) dễ bị gộp vào robot / nền nếu không có mask |
| **SOLD** | [2410.08822](https://arxiv.org/abs/2410.08822) | ICML 2025 (PMLR v267) | có | slot-attention object-centric latent dynamics | vượt DreamerV3 / TD-MPC2 trên task relational |
| **SAM 2**, **Depth Anything V2**, **VGGT**, **DUSt3R**, **CoTracker**, **TAPIR** | [2408.00714](https://arxiv.org/abs/2408.00714), [2406.09414](https://arxiv.org/abs/2406.09414), [2503.11651](https://arxiv.org/abs/2503.11651), [2312.14132](https://arxiv.org/abs/2312.14132), [2307.07635](https://arxiv.org/abs/2307.07635), [2306.08637](https://arxiv.org/abs/2306.08637) | VGGT CVPR 2025, TAPIR ICCV 2023 (comment); còn lại theo arXiv | đều có weights | teacher: mask / depth / geometry tokens / point tracks | trong sim LIBERO đều thay được bằng ground truth (segmentation, depth, object pose) |
| **Cosmos** / **LAPA** / **Genie** | [2501.03575](https://arxiv.org/abs/2501.03575) / [2410.11758](https://arxiv.org/abs/2410.11758) / [2402.15391](https://arxiv.org/abs/2402.15391) | arXiv / ICLR 2025 (comment) / arXiv | Cosmos, LAPA mở | tokenizer video / latent action từ video không nhãn | latent action chỉ có ích khi THIẾU action; WM của mình có action thật. Cosmos-AC 0 % trong bảng V-JEPA 2 |

### 2C. WM bị planner khai thác: sửa bằng data / training (gần bottleneck nhất)

| paper | arXiv | venue (kiểm tra) | cơ chế | kết quả |
|---|---|---|---|---|
| **Closing the Train-Test Gap in WMs for Gradient-Based Planning** | [2512.09929](https://arxiv.org/abs/2512.09929) | ICLR 2026 (OpenReview By4u7V6tAx qua bib + trang tác giả) | (1) Online World Modeling = **DAgger cho WM**: chạy planner, mô phỏng thật, fine-tune WM trên trajectory đã sửa; (2) Adversarial World Modeling: FGSM trên (z, a) | trên DINO-WM: **OWM không giúp** GBP (PushT 76 = baseline, Wall 80 = baseline, PointMaze 68 -> 88); AWM +16..+34 điểm, ngang CEM với 10 % thời gian. Lưu ý: bàn về **gradient** planning; mình dùng CEM |
| **Reinforced Planning with Latent WMs** (RP1) | [2608.18669](https://arxiv.org/abs/2608.18669) | arXiv preprint | planner học được qua WM; phát hiện WM hallucinate 'grasp trượt vẫn dính cube'; **1 vòng Dyna**: deploy planner, gom rollout fail, fine-tune WM | LeWM easy / hard 92.7 / 83.4 -> 95.0 / 88.6; PLDM 85.3 / 66.6 -> 93.3 / 84.8; hallucination grasp-and-miss giảm rõ |
| **SCALE** / **AnisoWM** | [2608.16287](https://arxiv.org/abs/2608.16287) / [2609.37441](https://arxiv.org/abs/2609.37441) | arXiv / arXiv | ép khoảng cách latent tương quan với khoảng cách state đặc quyền (SCALE) / hình học latent anisotropic | cải thiện planning success so với LeWM | 
| **Agnostic System Identification** (Ross & Bagnell) | [1203.1007](https://arxiv.org/abs/1203.1007) | ICML 2012 (comment) | DAgger cho model: thêm data ở phân bố do policy / planner gây ra | đảm bảo lý thuyết; gốc của option (vi) |
| *(trong repo)* CureWM, WMPO, Ctrl-World | xem RELATED_WORK G / B | | fine-tune WM trên counterfactual / rollout của policy | CureWM: WM lạc quan trên failure counterfactual |

Đọc chung 2A-2C:
- **Features**: bằng chứng mạnh duy nhất cho encoder là 'spatial patch > global vector' (DINO-WM) - mình đã có. Chưa thấy
  bài nào cho thấy RADIO / Theia / VGGT tokens làm *action-conditioned WM* chính xác hơn DINOv2 dưới planning.
- **3D làm target phụ** giúp nhất quán (DepthWorld, GeoMoLa, Spatial / Geometry Forcing, GaussianDream++), nhưng DreamVLA
  cảnh báo: target dày, nhiều chiều (depth map, DINO / SAM features) có thể **làm hại** khi đứng một mình; target nhỏ, gắn
  với cái sắp thay đổi (dynamic region) mới giúp. => head phụ nên là **low-dim, gắn với contact / object motion**.
- **Cấu trúc** (object-centric, rigid SE(3)) giảm drift dài hạn rất mạnh trên chính LIBERO (ORD-WM 37.6 vs 226 mm).
- **Exploitation**: hai bài trực tiếp nhất cho kết quả trái chiều về DAgger cho WM (RP1 dương với CEM-like planner học
  được, 2512.09929 âm với gradient planner). Không bài nào đo trên CEM-over-library + learned success critic như mình ->
  phải tự test, và test rẻ được vì log đã có trajectory thật của các bridge do search chọn.

## 3. Các phương án cho WM của mình

Ký hiệu metric (đều tính được từ log có sẵn hoặc CPU replay, train split):
- **gap_own(N)** = mean[C(imagined end) - C(true end)] trên top-8 pick do **chính WM đó** chọn ở budget tưởng tượng N
  (N = 512 mặc định), C = C^pi v2 shipped cố định (và lặp lại với critic `base` retrained để kiểm tra robustness). True end
  latent lấy bằng `scripts/replay_bridges.py` (bridge-only, không cần policy, CPU, ~5 phút / 500 pick).
- **Cb_own** = mean C(true end) trên chính các pick đó (pick có thật sự tốt hơn không; AUC(b) ~0.85 nên là proxy của hits).
- **gap_fixed** = như RQ3 step 1 trên pick cố định (`true_end_rq3`, do WM v1 chọn). Chỉ dùng làm phụ: mọi WM mới đều
  được lợi do **independence effect** (bài học CRIT-V), nên luôn so với một **control retrained**, không bao giờ so với WM v1.
- **FP@top-k**, **p@4**, **hits / sim** ở budget verify cố định: chỉ ở bước online.

| # | phương án | cơ chế mong đợi lên bottleneck | dự đoán (của mình, trước khi chạy) | rủi ro chính | công / chi phí A100 | test rẻ, đăng ký trước được |
|---|---|---|---|---|---|---|
| (i) | **Multi-encoder input**: DINOv2 + depth / geometry tokens (depth thật của sim, hoặc VGGT / Depth Anything nếu nhắm real) | WM 'thấy' độ sâu -> dự đoán tiếp xúc đúng hơn | **thấp**: DINOv2 trên true state đã đọc contact AUC 0.95; `pc_true` cho critic phẳng; lỗi là do dynamics dưới search, không do input | phải re-render mọi bank (log chỉ lưu latent); đổi input -> phải train lại critic; VGGT CC-BY-NC | re-render CPU 2-4 h (24 workers); VGGT-1B ~3-4 A100-h cho ~250k ảnh; WM x 3 seed ~6 h | trước hết: linear probe 'contact / grasp tại t+50 từ (z_t, actions)' trên DINOv2 vs DINOv2+depth tokens; nếu probe không tăng >= 0.03 AUC thì không làm |
| (ii) | **Auxiliary 3D heads trên latent tưởng tượng**: từ latent rollout **open-loop** (không teacher forcing) dự đoán 25-d symbolic (contacts, grasp, object height / up-axis, goal atoms) + object flow (centroid theo class + 16 FPS points) tại t = 10..50; nhãn sim từ `data/critic3d` | ép predictor giữ thông tin contact / object motion trên chính trajectory nó tưởng tượng (giống DepthWorld / GeoMoLa / GaussianDream++ dùng 3D làm target phụ) | **thấp-vừa** một mình; RGR head chỉ AUC 0.61 trên latent tưởng tượng -> có chỗ để cải thiện, nhưng chỉ trên phân bố train | DreamVLA: target dày làm hại -> giữ head **low-dim**, weight nhỏ, kiểm tra val h10 không xấu đi; search vẫn có thể khai thác vùng ngoài data | 0 sim mới; ~1.2x train WM (~2 A100-h / seed) | arm A3 / A4 trong E1 |
| (iii) | **Object-centric tokens** (SAM 2 mask, trong sim = segmentation thật) + per-object 3D state | dynamics theo object giảm drift, tách vật nhỏ (bowl) khỏi robot | **vừa** cho object motion; nhưng 72 token pool 6x6 quá thô cho slot (WorldDP: vật nhỏ bị gộp) -> cần token độ phân giải cao | kiến trúc mới + re-encode DINOv2 16x16 mọi bank; critic phải đổi theo | 1-2 tuần công; ~20 A100-h | chỉ sau E1/E2; thay bằng E2 (object state đặc quyền) để biết trần trước |
| (iv) | **Đổi encoder** sang RADIO / VGGT / Theia | feature 'giàu hơn' | **thấp**: V-JEPA 2-AC (ViT-g 1B) đã cho thấy encoder mạnh hơn vẫn bị chính search của nó khai thác (FP ~0.83); không paper nào báo gain cho action-conditioned WM dưới planning | tốn re-encode + train lại WM + critic; license | ~20-30 A100-h + re-render | không khuyến nghị; nếu cần, chỉ probe như (i) |
| (v) | **PointWorld làm imagination 3D song song**, veto ứng viên bất hợp lý (cross-WM disagreement) | lỗi 3D point-flow khác cấu trúc với latent DINOv2 -> ít tương quan hơn V-JEPA | **thấp** zero-shot (321 mm full-traj trên LIBERO theo ORD-WM); fine-tune vẫn 226 mm; trần của veto hình học = RGR `rec_geo_oracle` (FP 0.658, solved không tăng) | URDF / gripper swap (Robotiq -> Panda), OSC 20 Hz -> joint trajectory, 3 pass chain cho H = 50, DINOv3 gated | rất cao (dự án riêng) | thay bằng E2 (ORD-WM-lite, rẻ hơn 1-2 bậc) |
| (vi) | **WM-DAgger**: train WM trên bridge do search chọn (true latents đã có trong `verif_m5b`, `true_end_rq3`, `goal_verif`), lặp: WM_k search -> CPU replay top-K -> thêm vào data -> WM_{k+1} | đúng chỗ lỗi: trạng thái mà search khai thác (RP1: 1 vòng Dyna giảm hallucination grasp-and-miss) | **vừa-cao nhất trong các lựa chọn**, nhưng bằng chứng trái chiều (OWM âm cho gradient planning; M11 broad-action data n.s.) | WM học thuộc lỗi của WM v1 rồi search mới tìm lỗi mới (cần >= 2 vòng); nominal drift | vòng 1: 0 sim mới, ~2 A100-h / seed; mỗi vòng sau: GPU nhẹ (search tưởng tượng) + CPU replay | arm A1 / A2 trong E1 |
| (vii) | **AWM** (FGSM trên (z, a), 2512.09929) | làm mượt landscape; phơi WM với lân cận OOD mà không cần sim | **thấp-vừa**: cơ chế nhắm gradient planning, mình dùng CEM | thêm siêu tham số epsilon | ~1.3x train WM | arm A5 tùy chọn trong E1 |
| (viii) | (không phải WM) **linear recalibrator** trên embedding C^pi của end state tưởng tượng (T2Mem lead) | sửa có hệ thống phần lạc quan | đã đo: hits 3.14 vs 3.17 (true end) vs 2.64 | chưa held-out | ~0 | **baseline bắt buộc** cho mọi so sánh online |

## 4. Khuyến nghị: 2 thí nghiệm (đăng ký trước; train split; test split không đụng)

### E1 (chính): WM-DAgger x auxiliary 3D heads, đo bằng self-exploitation

**Câu hỏi**: train WM trên trạng thái mà search đã khai thác (vi) và / hoặc ép latent tưởng tượng mang contact / object
motion (ii) có làm **search của chính WM đó** ít tìm được lỗi tưởng tượng hơn không?

**Data (đã có, kiểm tra 2026-11-24)**:
- `data/wm/collect_v1`: 144 instance, 4,608 bridge (data train của WM v1), **không trùng** 63 instance RQ3.
- `data/wm/verif_m5b`: 256 instance x 64 bridge đã verify, mỗi file có z [64, 11, 72, 384] TRUE (mỗi 5 step), 16 bridge
  / arm (`random`, `wm_diverse`, `wm_noise`, `wm_qd`). Bỏ 63 instance RQ3 -> 193 instance: 3,088 bridge `random` +
  9,264 bridge do search của WM v1 chọn.
- `data/critic3d/{collect_v1, verif_m5b}`: nhãn sim cho cùng các bridge (sym [K, 11, 25], labelled point cloud [K, 6, 512]).
- Held-out: 63 instance RQ3 (`data/wm/true_end_rq3`: 1,920 pick do WM v1 chọn ở N = 8..512, true z + outcome; 218 thành công).

**Arms** (cùng recipe M2 `train_wm_v1`, cùng số step, train lại từ đầu, 3 seed; ~1.5 h / seed đo được ở M2, +20 % với heads):

| arm | data | loss |
|---|---|---|
| A0 control | collect_v1 | WM v1 recipe (retrained, seed mới) |
| A1 +random (volume control) | collect_v1 + 3,088 bridge `random` | như A0 |
| A2 +search (DAgger, volume-matched) | collect_v1 + 3,088 bridge search-chọn (lấy mẫu phân tầng theo arm / instance) | như A0 |
| A2+ (dose) | collect_v1 + 9,264 bridge search-chọn | như A0 |
| A3 3D-aux | collect_v1 | + heads trên latent rollout open-loop: sym 25-d (BCE / MSE) + object flow (centroid theo class + 16 FPS points, Huber), weight 0.1 |
| A4 DAgger + 3D-aux | như A2 | như A3 |
| A5 (tùy chọn) AWM | collect_v1 | + FGSM trên (z, a), eps theo 2512.09929 |

So sánh pre-registered: **A2 vs A1** (search-chọn vs thêm data bất kỳ), **A3 vs A0** (3D-aux), **A4 vs A2** (cộng thêm).
Critic: C^pi v2 shipped cố định cho mọi arm (thay đổi duy nhất là WM); lặp lại với critic `base` của CRIT-V.

**Bước đo**:
1. Offline tức thì: val h1 / h4 / h10 ratio (in-distribution), gap_fixed trên `true_end_rq3` (phụ, có independence effect).
2. **Self-exploitation (primary)**: với mỗi arm-seed, chạy pipeline `wm_bud_l0` / `wm_unc_l0` **chỉ trong tưởng tượng** trên
   63 instance held-out (cùng pool / CEM seed như RQ3 sweep, paired), lấy top-8 ở N = 32 / 128 / 512, CPU replay bridge
   (`scripts/replay_bridges.py`, 63 x 8 x 3 budget ~ 1.5k pick / arm-seed, ~15 phút CPU 24 worker) -> gap_own(N), Cb_own(N).
   Không cần policy continuation, không cần GPU lớn (SmolVLA sample pool + WM scoring).
3. Online (chỉ khi qua gate): verify bằng pi continuation, xem dưới.

**Decision rules (đăng ký trước)**, unit = instance, instance-cluster bootstrap 1,000 rep, seed-mean:
- **PASS -> online**: arm X so với control tương ứng có d gap_own(512) <= -0.10 (A0 dự kiến ~+0.33, tức giảm ~30 %) với
  95 % CI < 0, cùng dấu ở 3/3 seed, **và** d Cb_own(512) >= +0.03 với CI > 0, **và** val h10 ratio không xấu hơn A0 quá 5 %
  tương đối. (Chỉ giảm gap mà Cb_own không tăng = WM chỉ bi quan hơn, không tìm pick tốt hơn -> không tính là pass.)
- **PROMISING -> thêm 1 vòng DAgger (WM_1 search -> replay -> retrain), không online**: d gap_own <= -0.05 với CI < 0 nhưng
  Cb_own CI chứa 0.
- **CLOSE nhánh WM-data / 3D-aux**: |d gap_own(512)| < 0.05 cho cả A2 vs A1 và A3 vs A0 -> ghi 'ở quy mô này, data do search
  chọn và 3D-aux không giảm self-exploitation'; chuyển sang recalibrator + verify nhiều hơn. Trước khi close: chạy
  `retryvla-idea-review` (bắt buộc theo skill).
- **Online gate** (nếu PASS): paired search 127 instance (shards 0-3 của M5b; 63 RQ3 + 64 mới), pipeline `wm_unc_l0`, 8
  verification / instance, cùng pool / continuation seed; arms: X, A0, A0 + recalibrator (viii). Primary = **hits / instance**
  (sign-flip clustered p < 0.05 vs A0 **và** không thua recalibrator), secondary = solved (McNemar, chỉ yêu cầu không xấu đi;
  n = 127 không đủ power cho chênh solved < ~8 pt, và M9 `sim_prefix_critic` cho thấy trần solved nhỏ). Nominal không bị
  ảnh hưởng vì không distill trong E1.

**Chi phí**: 7 arm x 3 seed x ~1.8 A100-h ~ 38 A100-h train (A5 bỏ được -> ~32) + ~7 GPU-h search tưởng tượng + ~5 h CPU
replay. Disk: chỉ thêm z của pick (~1.5k x 11 x 72 x 384 fp16 ~ 0.9 GB / arm-seed -> ghi pooled 2x384 nếu chật quota).
Online gate: ~127 x 8 x 3 arm verify ~ 1 ngày trên 1 GPU (theo `VERIFICATION_COST.md`).

### E2 (rẻ, ưu tiên 2): imagination 3D object-centric (ORD-WM-lite, state đặc quyền) làm partner ít tương quan

**Câu hỏi**: một mô hình dynamics có **cấu trúc khác hẳn** (object-level rigid motion trên 3D state đặc quyền của sim, kiểu
ORD-WM) có lỗi ít tương quan với WM v1 hơn V-JEPA không, và veto theo nó có giảm FP trên pick của search không?

- **Mô hình**: token per-object (centroid + PCA up-axis + class từ labelled cloud t = 0 trong `data/critic3d`, cộng sym t = 0)
  + token gripper (eef pose, width từ q) + 50 x 7 action -> per-object delta centroid / up-axis tại t = 10..50 + grasp /
  contact flags và BDDL goal atoms tại t = 50. Transformer nhỏ (~5M param), train < 1 A100-h trên ~28k bridge (mọi bank
  critic3d trừ 63 instance held-out). Không cần render, không cần foundation model.
- **Dùng**: J = C_v1 chỉ khi ObjDyn đồng ý trên predicate liên quan task (object grasped / lifted / gần đích); hoặc
  J = min(C_v1, P_ObjDyn(goal atoms)).
- **Test offline** (held-out: `true_end_rq3` 1,920 pick có outcome + 63 instance trong `verif_m5b`):
  (a) độc lập: lỗi ObjDyn (centroid error tại t = 50) trên pick do WM v1 chọn / trên candidate `random` <= 1.2
  (nếu lỗi của nó cũng tăng trên pick thì nó không phải partner độc lập);
  (b) P(fail | C_v1 > 0.5 và ObjDyn đồng ý) <= 0.50 (xwm: 0.58-0.74);
  (c) FP@top4 sau veto <= FP của xwm-min - 0.05 với retention thành công thật >= 70 %;
  (d) phải hơn veto hình học kiểu RGR (`rec_geo`) về precision ở cùng retention.
- **Decision**: (a)-(d) đạt -> thêm làm arm rẻ trong online gate của E1; không đạt -> close 'veto 3D dynamics không làm
  lỗi bớt tương quan'. **Trần**: RGR `rec_geo_oracle` (FP 0.658, solved không tăng) -> kỳ vọng gain chủ yếu ở hits / sim,
  không ở solved; đó là lý do E2 xếp sau E1.
- Nếu E2 pass và muốn chuyển sang real: thay state đặc quyền bằng SAM 2 mask + depth (FoundationStereo / Depth Anything)
  như ORD-WM làm ngoài thực tế; khi đó mới cần foundation teachers.

## 5. Những gì KHÔNG khuyến nghị (và vì sao)

- **Đổi / ghép encoder** (RADIO, Theia, VGGT, SigLIP) trước khi có probe dương: bằng chứng nội bộ (contact AUC 0.95 trên
  true DINOv2 tokens; V-JEPA không chống exploitation) + literature chỉ có bằng chứng cho policy (Spatial Forcing), không
  cho action-conditioned WM dưới planning.
- **PointWorld zero-shot** làm veto: sai số trên LIBERO quá lớn (ORD-WM: 321 mm full-traj), chi phí tích hợp rất cao.
- **Latent-action models** (LAPA, Genie): giải bài toán thiếu action; mình có action thật.
- **Pixel / Gaussian video WM** (DepthWorld, GWM, TesserAct, Cosmos): mỗi candidate đắt hơn latent WM nhiều bậc (V-JEPA
  đã ~200x; video diffusion còn hơn), trong khi search của mình cần 256-768 candidate / instance.
- **3D vào critic**: đã test (CRIT-V), phẳng.

## 6. Ghi chú kiểm tra nguồn

- arXiv API (export.arxiv.org, 2026-11-24): toàn bộ ID trong bảng 2A-2C trả về đúng tiêu đề / tác giả / ngày v1.
- Nguồn thứ hai: CVF Open Access (PointWorld supp., TesserAct, Aether), PMLR (DINO-WM v267, SOLD v267, ParticleFormer v305,
  3D-VLA v235), ECVA (ManiGaussian), ICML virtual (GeoMoLa poster 62176), ICLR proceedings (Geometry Forcing),
  NeurIPS proceedings (Dyn-O), arXiv comment (Theia, D3Fields, PCWM, VGGT, TAPIR, LAPA, GWM, Im2Flow2Act, Track2Act,
  DepthWorld, AM-RADIO, DINO-Foresight, Agnostic SysID), GitHub / HF (PointWorld, DepthWorld, Spatial Forcing, DINO-WM,
  JEPA4R). **Chưa xác minh đầy đủ**: venue của DreamVLA (NeurIPS 2025) và AdaptiGraph (RSS 2024) chỉ qua kết quả tìm kiếm;
  ORD-WM 'CoRL 2026' chỉ thấy trên paperswithcode (để là arXiv preprint); 2512.09929 ICLR 2026 qua bib + trang tác giả
  (OpenReview id By4u7V6tAx), chưa mở trang OpenReview.
- Số liệu ORD-WM vs PointWorld lấy từ Table 1 bản HTML arXiv của ORD-WM (đo của tác giả ORD-WM, không phải của NVIDIA).
