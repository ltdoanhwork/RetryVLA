# Review 2026-11-26: RetryVLA trên protocol LIBERO-Goal (2610.05151), cập nhật

Viết 2026-11-26 ~10:35 UTC. Tiếp nối `REVIEW_2026-11-25_GOAL.md`. Protocol: 100 val scene, M0 = pi0 + LoRA 4k, recipe của
paper (M0 + 2k update, D0 + bank trộn đều). Δ = scene x seed bootstrap 95 %, paired theo seed với Original (Original
có 5 seed 7-11: 82 / 77 / 79 / 76 / 81 = **79.0**, SD 2.5). Paper: 78 -> 88 (1 seed).

## 1. Tóm tắt

1. **Chưa có arm nào vượt Original với CI loại trừ 0.** Ba arm tốt nhất đều quanh +2 đến +4:
   - strict-phys (46 survivor): 83.7, +4.3 [−5, +13];
   - G2-phys (18 strict survivor): 83.0, +3.7 [−4, +12];
   - G2-physns (bỏ seam rule, 36): 81.7, +2.3 [−5, +11], kèm **recovery +9.9 [+3.6, +17.1]**.
2. **Seed 9 dập tắt hai kết quả 2-seed đẹp:**
   - G2_wm@5: 86 / 85 -> 74 (mean 81.7, +2.3);
   - G2_wm@18: 76 / 82 -> 72.
   - Trước đó, P2-3 A (79.3, recovery +10) đã không replicate trên seed mới (73.7).
3. **G2-phys so với cùng share (G2_wm@5): +1.3 [−8, +12].** Rule "G2-phys gain is a share effect" ĐẠT. Bộ lọc vật
   lý không thêm gì đo được ngoài việc dùng ít data, nhưng G2-phys **ổn định hơn** (82-84 ở mọi seed so với 74-86).
4. **Mẫu chung:**
   - share recovery 1-2 % thì nominal xấp xỉ hoặc nhỉnh hơn Original;
   - share 11-20 % thì nominal mất 0-9 điểm;
   - recovery trên failure held-out tăng +2 đến +14 tuỳ share / filter.
   Nghĩa là trade-off nominal vs recovery vẫn còn đó; bộ lọc vật lý và share nhỏ chỉ dịch điểm cân bằng.

## 2. Bảng chính (val, 100 scene)

| arm | share | seeds | val | mean | Δ vs Original (paired) | own-failure RSR (Δ) |
|---|---|---|---|---|---|---|
| Original | 0 | 7-11 | 82 / 77 / 79 / 76 / 81 | 79.0 | - | 0.147 |
| G2 (curated onset + cont., 36) | 11 % | 7-9 | 67 / 72 / 79 | 72.7 | −6.7 [−18, +4] | 0.200 (+5.4) |
| P2-3 A (72) | 20 % | 7-9 | 83 / 78 / 77 | 79.3 | +0.0 [−8, +8] | 0.248 (+10.1) |
| P2-3 A replication | 20 % | 10-12 | 75 / 75 / 71 | 73.7 | −3.5 [−11.5, +4.5] (paired s10-11) | 0.226 |
| P2-3 B (36 + D_P 36) | 18 % | 7-9 | 75 / 71 / 80 | 75.3 | −4.0 [−13, +5] | 0.248 (+10.1) |
| G2_wm@5 | ~1.5 % | 7-9 | 86 / 85 / 74 | 81.7 | +2.3 [−7, +11] | 0.217 (+6.5) |
| G2_wm@18 | ~5 % | 7-9 | 76 / 82 / 72 | 76.7 | −2.7 [−12, +8] | 0.164 (+1.2) |
| **G2-phys (18 strict)** | 1.6 % | 7-11 | 84 / 82 / 83 / 80 / 81 | **82.0** | **+3.0 [−3.8, +9.8]** (5 seed) | 0.183 (+3.6, s7-9) |
| **G2-physns (36, bỏ seam rule)** | ~10 % | 7-9 | 82 / 82 / 81 | 81.7 | +2.3 [−5, +11] | **0.246 (+9.9)** |
| **strict-phys (46)** | ~3 % | 7-11 | 80 / 87 / 84 / 78 / 81 | **82.0** | **+3.0 [−4.0, +10.2]** (5 seed) | 0.167 (+2.0, s7-9) |
| G3-c20 (cap continuation 20) | ~8 % | 7-9 | 62 / 72 / 77 | 70.3 | −9.0 vs P2-3 A | 0.224 |
| G2 random (36) | 11 % | 7-9 | 80 / 76 / 80 | 78.7 | −0.7 | 0.218 (+7.1) |
| paper (cross-paper) | 4.1 % teacher | 1 | 78 -> 88 | - | +10 | không đo |

Seed mới của G4 (11-26): G2_wm@5 s9 74 (wine rack 0 / 10), G2_wm@18 s9 72, Original s10 76, s11 81.

## 2b. Kết quả rule G4 (cập nhật 17:25 UTC, 5 seed paired)

- **"beats Original" (Δ >= +3, CI lower bound > 0): KHÔNG đạt.** G2-phys và strict-phys đều +3.0, nhưng CI là
  [−3.8, +9.8] và [−4.0, +10.2].
- **"matches theirs" (Δ >= +5, CI > 0): KHÔNG đạt.**
- Seed 10 / 11 kéo hai arm về gần Original: G2-phys 80 / 81, strict-phys 78 / 81, so với 82-87 ở seed 7-9. Trung bình
  3 seed (83.0 / 83.7) có phần do may mắn seed.
- Gain tập trung ở task yếu: push plate 4.2 -> 7.8 / 8.0, wine rack 4.2 -> 6.8 / 5.2. Regression ở top drawer + bowl
  (7.8 -> 4.0 G2-phys, 6.2 strict-phys).

## 3. Đọc trung thực

- Hiệu ứng thật, nếu có, nhỏ hơn MDE ở n = 100 x 3 seed (~7 pt). Ba arm phys đều > Original ở mọi seed đã chạy,
  nhưng chênh lệch giữa seed của Original (76-82) cũng gần bằng gain.
- 5 seed (sau G4) có MDE paired ~5-6 pt. Nếu G2-phys / strict-phys giữ +4, CI có thể sát 0 nhưng có lẽ vẫn chưa loại
  trừ được.
- Kết luận có thể bảo vệ ngay bây giờ:
  - recovery data từ failure tự nhiên, verify bằng sim, dạy recovery thật: RSR held-out +6 đến +14 ở nhiều arm, CI > 0;
  - recovery data giữ được nominal khi share nhỏ (≤ 2-3 %) hoặc khi lọc vật lý chặt;
  - lợi ích +10 nominal của paper chưa tái tạo được.
- Mâu thuẫn còn lại: arm có recovery tốt nhất (G2-physns +9.9) không phải arm có nominal tốt nhất (strict-phys
  83.7, recovery +2).

## 4. Đang chạy (shell 7485771f `vla_RetryVLA_DoanhLT_2g_c`, 2 GPU, lane tmux rv_g4_g0 / rv_g4_g1)

| vòng | GPU 0 | GPU 1 | xong (UTC) |
|---|---|---|---|
| 1 | G2_wm@5 s9 = 74 | G2_wm@18 s9 = 72 | 07:04 (xong) |
| 2 | Original s10 = 76 | Original s11 = 81 | 10:09 (xong) |
| 3 | G2-phys s10 = 80 | G2-phys s11 = 81 | 12:42 (xong) |
| 4 | strict-phys s10 = 78 | strict-phys s11 = 81 | 15:16 (xong) |
| 5 | share sweep 5 (chạy), rồi 36 (s7) | share sweep 18 (chạy, s7) | ~17:55 / ~20:30 |

Shell tự kill khi cả hai lane xong (watcher `rv_g4_watch`). E8 / failure events của các run G4 chỉ chạy nếu còn GPU.

## 5. Bước tiếp theo (đề xuất)

1. Sau vòng 4: rule "beats Original" trên 5 seed paired cho G2-phys và strict-phys. Nếu một arm qua thì đóng băng
   recipe đó và dùng test split (100 scene) đúng một lần, × 3 seed, cùng Original.
2. Nếu không arm nào qua: viết kết quả là "recovery gain không tốn nominal ở share nhỏ" (claim khiêm tốn, có CI),
   không claim nominal gain.
3. Hướng còn mở cho nominal: chỉ dùng recovery trên task yếu (push plate, wine rack: 4 / 10 ở Original), hoặc
   task-balanced share; per-task, đây là nơi mọi arm recovery đều giúp.
