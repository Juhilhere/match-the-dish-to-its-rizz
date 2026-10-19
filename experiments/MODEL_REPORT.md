# Match the Dish to Its Rizz — model audit and submission record

## Competition contract

- Competition: `match-the-dish-to-its-rizz`
- Task: retrieve the gallery images matching each dish description.
- Metric: mean R-precision; the first `R` predictions are scored, where `R` is 19 or 20 for a test dish.
- Submission contract: all 303 test queries, exactly 20 distinct IDs per row, IDs from `test_gallery.csv`.
- The competition data terms restrict the images to this course. The model uses only the supplied competition data and generic pretrained vision models.

## Folder audit

The workspace contains the competition archive and extracted data, three image galleries, metadata/query/ground-truth CSVs, the official submission validator, baseline EfficientNet and CLIP/SigLIP/DINO/ConvNeXt experiments, cached embeddings, checkpoints, and generated submissions. Current gallery counts are 2,000 train images, 1,010 validation images, and 2,000 test images. There are 101 dish categories, 2,000 training labels, 202 validation queries, and 303 test queries.

## Selected model

`vit_base_patch16_siglip_224.v2_webli` is a generic pretrained SigLIP image backbone. The final local candidate uses the backbone with the classification head and final two transformer blocks fine-tuned on the 2,000 supplied training images. The best checkpoint was epoch 3. Its logits are blended 60/40 with a frozen-backbone SigLIP linear probe, then a balanced gallery assignment is applied before producing each query's ranking.

The balanced assignment uses the known gallery structure (20 slots per category) and 20 low-score dummy slots to absorb the 19-vs-20 category-count difference. Query category extraction is deterministic from the supplied category metadata, alternative names, and training query labels.

## Local evidence

| Candidate | Held-out validation result |
|---|---:|
| Frozen SigLIP linear probe, unrestricted | 0.846 |
| Frozen SigLIP linear probe, balanced assignment | 0.901 |
| Fine-tuned SigLIP checkpoint, balanced assignment | 0.915 |
| Fine-tuned/frozen SigLIP blend, balanced assignment | **0.920** |

The validation score uses the released validation gallery and ground truth only for model selection; it is not a Kaggle leaderboard score.

## Candidate artifact

- File: `siglip_finetuned_blend_submission.csv`
- SHA-256: `638800B256C16841FBA6273717B5E6BCC46272256D4FEF6DF9075DE1FCF9B246`
- `python data/validate_submission.py siglip_finetuned_blend_submission.csv --data-dir data`: `OK: 303 rows`
- Kaggle submission dry-run: all four file checks passed; no upload has occurred.

## Reproduction

1. `python finetune_siglip.py`
2. `python extract_finetuned_test.py`
3. `python build_finetuned_submission.py`
4. `python data/validate_submission.py siglip_finetuned_blend_submission.csv --data-dir data`

The final Kaggle upload remains a separate account-changing action and has not been performed.
