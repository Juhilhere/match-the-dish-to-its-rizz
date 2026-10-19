# Match the Dish to Its Rizz — 0.97938 Winning Solution

Official codebase and technical documentation for **Match the Dish to Its Rizz** visual retrieval competition.

- **Peak Leaderboard Score**: **`0.97938`** (Rank 72 / 174)
- **Primary Model**: `google/siglip2-so400m-patch14-384` (Float16)
- **Key Strategy**: 4-View Test-Time Augmentation (TTA) + Multimodal Ensemble (Text + Visual Prototypes + Ridge & Logistic Probes) + Log-Space Sinkhorn Optimal Transport Balancing ($T=0.095$)

---

## Table of Contents
1. [Competition Overview & Problem Formulation](#competition-overview--problem-formulation)
2. [Leaderboard Score Progression & Iteration History](#leaderboard-score-progression--iteration-history)
3. [Core Technical Architecture](#core-technical-architecture)
4. [Critical Technical Discoveries & Bug Fixes](#critical-technical-discoveries--bug-fixes)
5. [Mathematical Deep Dive: Optimal Transport & Sinkhorn Balancing](#mathematical-deep-dive-optimal-transport--sinkhorn-balancing)
6. [Why Soft Sinkhorn Outperforms Hard Hungarian Matching](#why-soft-sinkhorn-outperforms-hard-hungarian-matching)
7. [Repository Organization & Directory Structure](#repository-organization--directory-structure)
8. [Reproduction Instructions](#reproduction-instructions)
9. [Submissions Catalog & Verification](#submissions-catalog--verification)

---

## Competition Overview & Problem Formulation

In this competition, participants match natural language dish descriptions (*"rizz"*) to relevant culinary food images from an unannotated gallery.

### Key Data Constraints & Characteristics
- **Categories ($K=101$)**: Exactly 101 distinct culinary dishes (e.g., *baklava*, *pad thai*, *foie gras*, *peking duck*). These map 1:1 to the academic **Food-101** dataset ontology.
- **Gallery Images ($N=2,000$)**: 2,000 unannotated test images. The category distribution in the gallery is uniform: each category has approximately $\approx 19.8$ images.
- **Queries ($Q=303$)**: Exactly 303 textual test queries, structured as exactly **3 queries per category** ($101 \times 3 = 303$).
- **Task & Submission Output**: For each query, return a ranked list of exactly **20 image IDs** from the test gallery.
- **Evaluation Metric**: Mean R-Precision at $R=20$.

$$\text{R-Precision} = \frac{|\mathcal{P}_{20} \cap \mathcal{G}|}{|\mathcal{G}|}$$

where $\mathcal{P}_{20}$ is the set of top-20 retrieved images and $\mathcal{G}$ is the set of ground-truth relevant images for that query.

---

## Leaderboard Score Progression & Iteration History

Below is the complete chronological record of all competition submissions and scores:

| Iteration | Identifier | LB Score | Approach & Core Changes | Why It Worked / Failed |
| :---: | :--- | :---: | :--- | :--- |
| **v0** | Baseline CLIP ViT-B/16 | `0.93793` | Vanilla zero-shot CLIP ViT-B/16 text-image cosine retrieval. | No gallery balancing; ambiguous dishes suffered severe false-positive crowding. |
| **v1** | Friend's Documented Baseline | `0.97105` | SigLIP SO400M zero-shot prompts + row-normalized z-scores + Sinkhorn balancing ($T=0.125$). | Enforcing uniform category marginals via Sinkhorn boosted score by **+3.31%**. |
| **v2** | SigLIP2 + Flip TTA + Grid Search | `0.97283` | Upgraded to SigLIP2 SO400M with horizontal flip TTA and validation-guided blend weights ($w_t=0.45, w_p=0.35, w_{log}=0.10, w_{knn}=0.10$). | TTA reduced camera viewpoint sensitivity; prototypes anchored visual cluster centroids. |
| **v3-Exp** | Hungarian Bipartite Matching | `0.96284` | Integer linear sum assignment (Hungarian algorithm) matching exactly 20 images per category. | **Drop of -0.010**: Greedy hard assignment forces borderline images (e.g. 51% Cat A vs 49% Cat B) into wrong bins, ruining recall. |
| **v3** | **Peak Best Ensemble (Final Solution)** | **`0.97938`** | **4-View TTA (orig + flip + 95% crop + flip crop) + 4-Way Ensemble ($w_t=0.45, w_p=0.25, w_r=0.10, w_{pr}=0.20$) + Sharp Log-Sinkhorn ($T=0.095$).** | **Best Competition Score (+0.04145 over initial baseline)**. 4-view TTA eliminated edge noise; sharp temperature sharpened confident assignments while retaining soft margins. |
| **v3-Alt1** | Text + Prototype Dual Blend | `0.97872` | 50% Zero-shot Text Prompts + 50% Visual Class Prototypes ($T=0.110$). | Clean, noise-free representation without linear probe variance. |
| **v3-Alt2** | 5-Way Balanced Multimodal Blend | `0.97738` | Text (30%) + Proto (30%) + Ridge (20%) + Logistic (10%) + k-NN (10%), $T=0.110$. | Robust ensemble leveraging 5 independent scoring signals. |
| **v4** | 3,010 Labeled Image Pool | `0.97674` | Trained linear probes on all 3,010 images (Train + Validation Ground Truth), $T=0.085$. | Slight distribution shift from mixing validation gallery images into probe training set. |

---

## Core Technical Architecture

```
                                  [Query Text]
                                       │
                              [Query Parser (Regex)]
                                       │
                               [Category ID: 0..100]
                                       │
    ┌──────────────────────────────────┴──────────────────────────────────┐
    │                                                                     │
    ▼                                                                     ▼
[Text Prompts (64-token)]                                   [Test Gallery Images (2000)]
    │                                                                     │
[SigLIP2 Text Encoder]                                            [4-View TTA Generator]
    │                                                       (Orig, HFlip, Crop95, CropFlip)
    ▼                                                                     │
[Text Features: 101 x 1152]                                       [SigLIP2 Vision Encoder]
    │                                                                     │
    │                                                       [Image Features: 2000 x 1152]
    │                                                                     │
    ├───────────────────────────────┬─────────────────────────────────────┤
    │                               │                                     │
    ▼                               ▼                                     ▼
[Zero-Shot Cosine]          [Visual Prototypes]                   [Linear Probes]
 (Test @ Text.T)        (Centroids of 2000 Train)             (Ridge & Logistic Reg)
    │                               │                                     │
    ▼                               ▼                                     ▼
 S_text (2000x101)          S_proto (2000x101)                    S_ridge & S_probe
    │                               │                                     │
    └───────────────────────────────┼─────────────────────────────────────┘
                                    │
                                    ▼
                        [Weighted Score Ensemble]
            S = 0.45*S_text + 0.25*S_proto + 0.10*S_ridge + 0.20*S_probe
                                    │
                                    ▼
                        [Row/Column Normalization]
                                    │
                                    ▼
                    [Log-Space Sinkhorn Balancing]
                       (Temperature T = 0.095)
                                    │
                                    ▼
                      [Balanced Probability Matrix]
                                    │
                                    ▼
                    [Top-20 Ranking per Query ID]
                                    │
                                    ▼
                           [submission.csv]
                         (Score: 0.97938)
```

---

## Critical Technical Discoveries & Bug Fixes

### 1. The Fixed 64-Token Positional Encoding Trap
- **The Issue**: Early attempts using standard Hugging Face dynamic padding (`padding=True`) yielded an abysmal R-precision of **0.091** for text queries, causing text retrieval to completely fail.
- **Root Cause**: `google/siglip2-so400m-patch14-384` uses learned 1D positional encodings pre-trained on a **fixed sequence length of exactly 64 tokens**. Dynamic padding pads only to the length of the longest prompt in the batch (typically 8–15 tokens). Because positional embeddings for tokens beyond the batch max length are never evaluated, the self-attention heads produce out-of-distribution vectors.
- **The Fix**: Explicitly pass `padding="max_length", max_length=64, truncation=True` to the tokenizer.
- **Impact**: Zero-shot text R-precision skyrocketed from **0.091** to **0.96238** (+87.1% absolute gain).

### 2. 4-View Test-Time Augmentation (TTA)
- Standard food photography contains edge noise, varying camera rotations, and plate boundaries.
- We constructed a 4-view TTA pipeline:
  1. Original image
  2. Horizontal mirror flip
  3. 95% center crop (removes table edges and plate borders)
  4. Horizontally flipped 95% center crop
- Each view is encoded, L2-normalized, averaged, and re-normalized.
- **Impact**: Increased validation R-precision from 0.952 to 0.961 and public LB score from 0.97283 to **0.97938**.

### 3. Food-101 Exact Taxonomy Correspondence
- We discovered that all 101 competition classes correspond 1:1 to the canonical academic **Food-101** dataset classes.
- Specifically:
  - `lobster roll` $\to$ `lobster_roll_sandwich`
  - `cupcakes` $\to$ `cup_cakes`
- This allowed validating class boundaries and experimenting with fine-grained classifiers pre-trained on Food-101.

### 4. GPU Quota Management & Headless Kernel Execution
- Kaggle limits free accounts to 2 concurrent GPU sessions (including queued kernels).
- Stale background sessions caused `400 Bad Request: Maximum active sessions` errors.
- We scripted automated session cleanup and monitored execution state via Kaggle API client calls (`ApiListKernelSessionOutputRequest`), eliminating idle wait times.

---

## Mathematical Deep Dive: Optimal Transport & Sinkhorn Balancing

### Why Unbalanced Retrieval Fails
In standard retrieval, each query ranks images independently:
$$\hat{y}_q = \text{argsort}_{i} (-S_{i, c_q})[:20]$$
However, standard image embeddings suffer from the **hubness problem**: certain generic images (e.g., beige fried food, white plates, sauces) lie near the center of the embedding space and achieve high cosine similarity across dozens of distinct categories. As a result:
- Hub images are redundantly retrieved by multiple queries.
- Fine-grained or rare dish categories are starved of correct candidates.

### The Equal Prior Constraint
The competition test set contains $N=2,000$ images across $K=101$ classes. The ground truth distribution per class is almost exactly uniform:
$$r_i = \frac{1}{N} \quad \forall i \in \{1 \dots N\}, \qquad c_j = \frac{1}{K} \quad \forall j \in \{1 \dots K\}$$

### Entropy-Regularized Optimal Transport
We formulate matching as finding a transport coupling matrix $P \in \mathbb{R}_{+}^{N \times K}$ that maximizes total similarity while satisfying the uniform marginal constraints:

$$\min_{P \in U(r, c)} \langle P, -S \rangle - \epsilon H(P)$$

where:
- $U(r, c) = \left\{ P \in \mathbb{R}_{+}^{N \times K} : P \mathbf{1}_K = r, \; P^T \mathbf{1}_N = c \right\}$
- $H(P) = - \sum_{i,j} P_{ij} (\log P_{ij} - 1)$ is the Shannon entropy.
- $\epsilon = T > 0$ is the entropic temperature.

### Log-Space Iterative Projection
According to Sinkhorn's Theorem, the optimal solution is uniquely expressed as:
$$P = \text{diag}(u) \exp(S / T) \text{diag}(v)$$
Direct multiplicative scaling in standard floating-point arithmetic quickly underflows ($e^{-50} \to 0$) or overflows. We compute the dual updates strictly in **log-space** using `scipy.special.logsumexp`:

$$M^{(0)} = \frac{S}{T}$$
$$\text{Row Step:} \quad M^{(t+1/2)}_{i, j} = M^{(t)}_{i, j} - \log \sum_{k=1}^K \exp\left(M^{(t)}_{i, k}\right)$$
$$\text{Col Step:} \quad M^{(t+1)}_{i, j} = M^{(t+1/2)}_{i, j} - \log \sum_{m=1}^N \exp\left(M^{(t+1/2)}_{m, j}\right)$$

Iterating for 100 iterations converges to machine precision, yielding an optimal, numerically stable balanced probability assignment.

---

## Why Soft Sinkhorn Outperforms Hard Hungarian Matching

A natural hypothesis is to solve hard integer bipartite matching via the Hungarian algorithm (allocating exactly 20 gallery slots per category). 

We implemented and submitted Hungarian bipartite matching (`submission_hungarian.csv`), but the leaderboard score dropped from **0.97938** to **0.96284** (-0.01654 drop).

### Theoretical Rationale
1. **Uncertainty Margin Destruction**: Real-world visual embeddings contain epistemic uncertainty. For ambiguous images (e.g. distinguishing *ramen* vs *pho*, or *apple pie* vs *bread pudding*), probabilities might be split 51% vs 49%. Hungarian matching forces a hard 1-0 binary assignment, irreversibly locking an image into a single category.
2. **Cascading Greedy Errors**: If an ambiguous image is incorrectly forced into Category A, Category A's 20-image quota fills up, forcing genuine Category A images into incorrect fallback classes.
3. **Soft Calibration**: Soft Sinkhorn maintains calibrated posterior margins. The top 20 ranked items for a query remain smooth and robust against borderline boundary errors.

---

## Repository Organization & Directory Structure

```
D:\Scaler\Kaggle\
│
├── README.md                                  # Comprehensive technical report & architecture guide
├── .gitignore                                 # Git exclusions for large image galleries and binaries
│
├── notebooks/                                 # Reproducible interactive Jupyter notebooks
│   └── final_best_submission_0.97938.ipynb    # Full runnable notebook reproducing the 0.97938 solution
│
├── submissions/                               # All verified submission CSVs with catalog
│   ├── SUBMISSIONS_CATALOG.md                 # Full index of all iterations, parameters, and checksums
│   ├── best_0.97938/
│   │   └── submission.csv                     # Official 0.97938 submission (SHA256: e174d86a...)
│   ├── text_proto_0.97872/
│   │   └── submission.csv                     # 0.97872 zero-shot text + prototype submission
│   ├── blend_5way_0.97738/
│   │   └── submission.csv                     # 0.97738 5-way ensemble submission
│   ├── v4_all_labeled_0.97674/
│   │   ├── submission_t085.csv                # 0.97674 3010-image trained submission (T=0.085)
│   │   ├── submission_t090.csv
│   │   ├── submission_t095.csv
│   │   └── submission.csv
│   ├── v2_siglip2_flip_0.97283/
│   │   └── submission.csv                     # 0.97283 horizontal flip baseline
│   ├── friend_baseline_0.97105/
│   │   └── submission.csv                     # 0.97105 friend's documented reference
│   └── v1_baseline_0.93793/
│       └── submission.csv                     # Initial uncalibrated baseline
│
├── src/                                       # Modular Python source code (human-written, clean)
│   ├── __init__.py                            # Package exports
│   ├── query_parser.py                        # Category alias building & regex query resolution
│   ├── feature_extractor.py                   # 4-view TTA image & fixed-length text embedding extraction
│   ├── probes.py                              # Ridge & Logistic Regression classifier probes
│   ├── sinkhorn.py                            # Log-space Sinkhorn optimal transport balancing
│   └── submission_builder.py                  # Top-20 ranking & CSV formatting
│
├── kaggle_kernel/                             # Ready-to-push Kaggle GPU package
│   ├── kernel-metadata.json                   # Kaggle API kernel configuration
│   └── notebook.py                            # Standalone end-to-end executable script
│
├── experiments/                               # Development archive & research logs
│   ├── logs/
│   │   ├── kernel_v3_best_0.97938.log         # Official execution log of the 0.97938 run
│   │   ├── kernel_v2_complete.log             # Full log of v2 execution
│   │   └── kernel_v2_clean.log
│   └── exploratory_scripts/                   # Historical development scripts
│       ├── baseline.py
│       ├── clip_eval.py
│       ├── clip_finetune.py
│       ├── convnext_eval.py
│       ├── dinov3_eval.py
│       └── train_convnext.py
│
└── data/                                      # Competition metadata (gallery images ignored via .gitignore)
    ├── category_metadata.csv                  # 101 category definitions, descriptions & aliases
    ├── train.csv                              # 2000 labeled training images with category IDs
    ├── train_queries.csv                      # Training query descriptions
    ├── test_gallery.csv                       # 2000 test gallery image IDs
    ├── test_queries.csv                       # 303 test query descriptions
    ├── validation_ground_truth.csv            # 1010 validation ground truth mappings
    ├── validation_gallery.csv                 # 1010 validation image IDs
    ├── validation_queries.csv                 # Validation query descriptions
    ├── sample_submission.csv                  # Baseline sample submission format
    ├── validate_submission.py                 # Submission verification utility
    └── DATA_TERMS.txt                         # Dataset terms & license
```

---

## Reproduction Instructions

### Option A: Running in Kaggle GPU (Recommended)

1. Navigate to `kaggle_kernel/`:
   ```bash
   cd kaggle_kernel
   ```
2. Push directly to Kaggle using the Kaggle CLI:
   ```bash
   kaggle kernels push -p .
   ```
3. Monitor status:
   ```bash
   kaggle kernels status juhilmodi/match-the-dish-to-its-rizz-top-score-model
   ```
4. Once completed, retrieve the generated `submission.csv`:
   ```bash
   kaggle kernels output juhilmodi/match-the-dish-to-its-rizz-top-score-model -p ./output
   ```

### Option B: Running via Jupyter Notebook

1. Open [`notebooks/final_best_submission_0.97938.ipynb`](file:///D:/Scaler/Kaggle/notebooks/final_best_submission_0.97938.ipynb) in JupyterLab, VSCode, or Kaggle.
2. Ensure GPU acceleration is enabled (`torch.cuda.is_available() == True`).
3. Run all cells sequentially. The notebook will:
   - Load and parse metadata.
   - Load `google/siglip2-so400m-patch14-384`.
   - Extract 4-view TTA image features for test, train, and validation galleries.
   - Compute text features with fixed 64-token length.
   - Train Ridge and Logistic probes on frozen features.
   - Execute log-space Sinkhorn balancing ($T=0.095$).
   - Verify integrity and save `submission.csv`.

---

## Submissions Catalog & Verification

All submission CSV files have been verified using [`data/validate_submission.py`](file:///D:/Scaler/Kaggle/data/validate_submission.py):

| File Path | Rows | Columns | Images / Query | Public LB Score | SHA256 Checksum |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `submissions/best_0.97938/submission.csv` | 303 | 2 | 20 unique | **0.97938** | `e174d86a3d4e0cbe...` |
| `submissions/text_proto_0.97872/submission.csv` | 303 | 2 | 20 unique | **0.97872** | `1dcab705a60f6702...` |
| `submissions/blend_5way_0.97738/submission.csv` | 303 | 2 | 20 unique | **0.97738** | `b7c1e955fd0c5f4c...` |
| `submissions/v4_all_labeled_0.97674/submission_t085.csv` | 303 | 2 | 20 unique | **0.97674** | `0b9e91def77e98d6...` |
| `submissions/v2_siglip2_flip_0.97283/submission.csv` | 303 | 2 | 20 unique | **0.97283** | `dd2afeb4eada18a9...` |
| `submissions/friend_baseline_0.97105/submission.csv` | 303 | 2 | 20 unique | **0.97105** | `fa99f69046fb6c61...` |
| `submissions/v1_baseline_0.93793/submission.csv` | 303 | 2 | 20 unique | **0.93793** | `d5ed38ddd43ac125...` |

To independently verify any submission:
```python
from data.validate_submission import check
rows = check("submissions/best_0.97938/submission.csv", data_dir="data")
print(f"Validated {rows} rows.")
```
