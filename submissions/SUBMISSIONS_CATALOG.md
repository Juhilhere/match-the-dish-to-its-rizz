# Submissions Catalog & Leaderboard Progression

This catalog records all verified submission files generated during the competition iterations, with their exact configuration, weights, Sinkhorn temperature, SHA256 checksum, and official Kaggle Public Leaderboard scores.

| Rank | Identifier | LB Score | Approach / Strategy | Key Hyperparameters | Rows | SHA256 Hash |
| :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **1** | `best_0.97938/submission.csv` | **0.97938** | SigLIP2 4-View TTA + Weighted 4-Way Ensemble + Log-Sinkhorn Balancing | $w_{\text{text}}=0.45, w_{\text{proto}}=0.25, w_{\text{ridge}}=0.10, w_{\text{probe}}=0.20$, $T=0.095$ | 303 | `e174d86a3d4e0cbe...` |
| **2** | `text_proto_0.97872/submission.csv` | **0.97872** | Pure Zero-Shot Text Embeddings + Visual Class Prototypes | $w_{\text{text}}=0.50, w_{\text{proto}}=0.50$, $T=0.110$ | 303 | `1dcab705a60f6702...` |
| **3** | `blend_5way_0.97738/submission.csv` | **0.97738** | 5-Way Balanced Multimodal Ensemble (Text + Proto + Ridge + Logistic + k-NN) | $w_{\text{t}}=0.30, w_{\text{p}}=0.30, w_{\text{r}}=0.20, w_{\text{log}}=0.10, w_{\text{knn}}=0.10$, $T=0.110$ | 303 | `b7c1e955fd0c5f4c...` |
| **4** | `v4_all_labeled_0.97674/submission_t085.csv` | **0.97674** | 3,010-Image Labeled Pool (Train + Validation Ground Truth) + Sharp Temperature | $w_{\text{t}}=0.40, w_{\text{p}}=0.35, w_{\text{r}}=0.15, w_{\text{log}}=0.10$, $T=0.085$ | 303 | `0b9e91def77e98d6...` |
| **5** | `v2_siglip2_flip_0.97283/submission.csv` | **0.97283** | SigLIP2 SO400M with Horizontal Flip TTA + Grid Search Blend | $w_{\text{t}}=0.45, w_{\text{p}}=0.35, w_{\text{log}}=0.10, w_{\text{knn}}=0.10$, $T=0.125$ | 303 | `dd2afeb4eada18a9...` |
| **6** | `friend_baseline_0.97105/submission.csv` | **0.97105** | Friend's Documented Baseline: Zero-Shot Text Cosine Similarities + Sinkhorn | Pure Text Prompts, Friend's Normalization, $T=0.125$ | 303 | `fa99f69046fb6c61...` |
| **7** | `v1_baseline_0.93793/submission.csv` | **0.93793** | Initial Baseline: Standard CLIP ViT-B/16 Zero-Shot Retrieval | Uncalibrated Cosine Ranking | 303 | `d5ed38ddd43ac125...` |

---

### Integrity Verification

All submission CSVs adhere to the competition format:
- Exactly **303 queries** (one per test query, 3 queries per category across 101 categories).
- Exactly **20 unique space-delimited image IDs** retrieved per query.
- Validated with [`data/validate_submission.py`](file:///D:/Scaler/Kaggle/data/validate_submission.py).
