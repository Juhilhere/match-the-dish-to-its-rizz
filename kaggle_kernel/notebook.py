import os
import sys
import re
import time
import hashlib
import warnings
import random
from pathlib import Path

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn.functional as F
from scipy.special import logsumexp
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, RidgeClassifier

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if Path('/kaggle/input').exists():
    matches = list(Path('/kaggle/input').rglob('category_metadata.csv'))
    if matches:
        ROOT = matches[0].parent
    else:
        ROOT = Path('/kaggle/input/match-the-dish-to-its-rizz')
    OUTPUT = Path('/kaggle/working')
else:
    ROOT = Path('data')
    OUTPUT = Path('.')

print(f"Dataset root: {ROOT.resolve()}")
print(f"Output directory: {OUTPUT.resolve()}")

cats = pd.read_csv(ROOT / 'category_metadata.csv')
cidx = {c: i for i, c in enumerate(cats.category_id)}
idx2cid = {i: c for c, i in cidx.items()}
N_CATS = len(cats)

train_df = pd.read_csv(ROOT / 'train.csv')
train_df['y'] = train_df.category_id.map(cidx)
train_queries = pd.read_csv(ROOT / 'train_queries.csv')

test_queries = pd.read_csv(ROOT / 'test_queries.csv')
test_gallery = pd.read_csv(ROOT / 'test_gallery.csv')
test_ids = test_gallery.image_id.tolist()
N_TEST = len(test_ids)
tid_map = {x: i for i, x in enumerate(test_ids)}

val_gt_path = ROOT / 'validation_ground_truth.csv'
HAS_VAL = val_gt_path.exists()

aliases = {}
for _, r in cats.iterrows():
    a = {r.category_name.lower()}
    if isinstance(r.alternative_names, str) and r.alternative_names.strip():
        a |= {x.strip().lower() for x in re.split(r'[,|;/]', r.alternative_names) if x.strip()}
    aliases[r.category_id] = a

for _, r in train_queries.iterrows():
    aliases.setdefault(r.category_id, set()).add(r.category_name.lower())

def parse_query(text):
    hits = []
    t = text.lower()
    for cid, aa in aliases.items():
        for a in aa:
            if re.search(r'(?<![a-z])' + re.escape(a) + r'(?![a-z])', t):
                hits.append((len(a), cid))
    return max(hits)[1] if hits else None

query_categories = {}
for _, q in test_queries.iterrows():
    cid = parse_query(q.description)
    assert cid is not None, f"Failed to parse query: {q.description}"
    query_categories[q.query_id] = cid

print(f"Parsed {len(query_categories)} test queries")

from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip2-so400m-patch14-384"
print(f"Loading {MODEL_NAME}...")
t_load = time.time()
model = AutoModel.from_pretrained(MODEL_NAME, dtype=torch.float16).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(MODEL_NAME)
print(f"Loaded in {time.time() - t_load:.1f}s")

def extract_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        if out.last_hidden_state.dim() == 3:
            return out.last_hidden_state[:, 0]
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return extract_tensor(out[0])
    return out

def extract_gallery_features(gallery_dir, image_ids, batch_size=16, desc="gallery"):
    print(f"Extracting {len(image_ids)} images from {desc} (4-view TTA)...")
    all_feats = []
    t0 = time.time()
    
    for start in range(0, len(image_ids), batch_size):
        end = min(start + batch_size, len(image_ids))
        batch_ids = image_ids[start:end]
        
        v1_list, v2_list, v3_list, v4_list = [], [], [], []
        for iid in batch_ids:
            path = gallery_dir / f"{iid}.jpg"
            img = Image.open(path)
            img = ImageOps.exif_transpose(img).convert('RGB')
            w, h = img.size
            crop_box = (int(w * 0.05), int(h * 0.05), int(w * 0.95), int(h * 0.95))
            cropped = img.crop(crop_box).resize((w, h), Image.Resampling.BICUBIC)
            
            v1_list.append(img)
            v2_list.append(img.transpose(Image.FLIP_LEFT_RIGHT))
            v3_list.append(cropped)
            v4_list.append(cropped.transpose(Image.FLIP_LEFT_RIGHT))
            
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                inp1 = processor(images=v1_list, return_tensors="pt", padding=True).to(DEVICE)
                f1 = F.normalize(extract_tensor(model.get_image_features(**inp1)).float(), dim=-1)
                
                inp2 = processor(images=v2_list, return_tensors="pt", padding=True).to(DEVICE)
                f2 = F.normalize(extract_tensor(model.get_image_features(**inp2)).float(), dim=-1)
                
                inp3 = processor(images=v3_list, return_tensors="pt", padding=True).to(DEVICE)
                f3 = F.normalize(extract_tensor(model.get_image_features(**inp3)).float(), dim=-1)
                
                inp4 = processor(images=v4_list, return_tensors="pt", padding=True).to(DEVICE)
                f4 = F.normalize(extract_tensor(model.get_image_features(**inp4)).float(), dim=-1)
                
                f_avg = F.normalize((f1 + f2 + f3 + f4) / 4.0, dim=-1)
                all_feats.append(f_avg.cpu())
                
        if (start // batch_size) % 15 == 0 or end == len(image_ids):
            elapsed = time.time() - t0
            rate = end / max(elapsed, 0.01)
            print(f"  [{desc}] {end}/{len(image_ids)} ({rate:.1f} imgs/s)")
            
    feats = torch.cat(all_feats, dim=0)
    print(f"[{desc}] Finished in {time.time() - t0:.1f}s, shape: {feats.shape}")
    return feats

test_feats = extract_gallery_features(ROOT / 'test_gallery', test_ids, batch_size=16, desc="test_gallery")

train_ids = train_df.image_id.tolist()
train_feats = extract_gallery_features(ROOT / 'train_gallery', train_ids, batch_size=16, desc="train_gallery")
train_labels = train_df.y.to_numpy()

cat_text_list = []
for _, row in cats.iterrows():
    name = row.category_name
    desc = row.description
    
    prompts = [
        f"a photo of {name}.",
        f"a photograph of {name}.",
        f"an image of {name}.",
        f"{name}.",
        f"a picture of {name}.",
        f"food: {name}.",
    ]
    if isinstance(desc, str) and desc.strip():
        prompts.append(f"a photo of {name}, {desc}.")
        prompts.append(f"{name}: {desc}.")
        prompts.append(f"an image of {desc}, known as {name}.")
    
    if isinstance(row.alternative_names, str) and row.alternative_names.strip():
        for alt in re.split(r'[,|;/]', row.alternative_names):
            alt = alt.strip()
            if alt:
                prompts.append(f"a photo of {alt}.")
                prompts.append(f"{alt}.")
                
    if isinstance(row.train_paraphrases, str) and row.train_paraphrases.strip():
        for p in row.train_paraphrases.split('|'):
            p = p.strip()
            if p:
                prompts.append(p)
                
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=torch.float16):
            inp = processor(text=prompts, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(DEVICE)
            tf = extract_tensor(model.get_text_features(**inp)).float()
            tf = F.normalize(tf, dim=-1)
            c_feat = F.normalize(tf.mean(dim=0, keepdim=True), dim=-1)
            cat_text_list.append(c_feat.cpu())

cat_text_feats = torch.cat(cat_text_list, dim=0)
print(f"Category text features shape: {cat_text_feats.shape}")

cat_proto_list = []
for k in range(N_CATS):
    mask = (train_labels == k)
    if mask.sum() > 0:
        proto = train_feats[mask].mean(dim=0, keepdim=True)
        proto = F.normalize(proto, dim=-1)
    else:
        proto = cat_text_feats[k:k+1]
    cat_proto_list.append(proto)

cat_proto_feats = torch.cat(cat_proto_list, dim=0)
print(f"Category visual prototypes shape: {cat_proto_feats.shape}")

S_text = (test_feats @ cat_text_feats.T).numpy()
S_proto = (test_feats @ cat_proto_feats.T).numpy()

all_cos = (test_feats @ train_feats.T).numpy()
S_knn = np.zeros((N_TEST, N_CATS), dtype=np.float32)
for k in range(N_CATS):
    mask = (train_labels == k)
    k_sims = all_cos[:, mask]
    top3 = np.sort(k_sims, axis=1)[:, -3:]
    S_knn[:, k] = top3.mean(axis=1)

X_train = train_feats.numpy()
y_train = train_labels
X_test = test_feats.numpy()

clf = LogisticRegression(C=2.5, max_iter=800, solver='lbfgs', multi_class='multinomial', random_state=SEED)
clf.fit(X_train, y_train)
print(f"Logistic Regression train acc: {clf.score(X_train, y_train)*100:.2f}%")
S_probe = clf.decision_function(X_test)

ridge = RidgeClassifier(alpha=1.5, random_state=SEED)
ridge.fit(X_train, y_train)
print(f"Ridge train acc: {ridge.score(X_train, y_train)*100:.2f}%")
S_ridge = ridge.decision_function(X_test)

def normalize_scores(S, mode='friend'):
    if mode == 'friend':
        return (S - S.mean(axis=1, keepdims=True)) / (S.std() + 1e-8)
    else:
        return (S - S.mean(axis=1, keepdims=True)) / (S.std(axis=1, keepdims=True) + 1e-8)

def sinkhorn_log(S, temperature=0.125, iterations=100):
    M = S / temperature
    for _ in range(iterations):
        M = M - logsumexp(M, axis=1, keepdims=True)
        M = M - logsumexp(M, axis=0, keepdims=True)
    return M

def build_submission_df(scores, test_ids, query_categories, test_queries):
    rows = []
    for _, q in test_queries.iterrows():
        cid = query_categories[q.query_id]
        k = cidx[cid]
        top20_indices = np.argsort(-scores[:, k])[:20]
        rows.append({
            'query_id': q.query_id,
            'image_ids': ' '.join(test_ids[idx] for idx in top20_indices)
        })
    return pd.DataFrame(rows)

best_weights = (0.35, 0.35, 0.15, 0.15)
best_temp = 0.110

if HAS_VAL:
    val_gt = pd.read_csv(val_gt_path)
    val_gallery_df = pd.read_csv(ROOT / 'validation_gallery.csv')
    val_queries_df = pd.read_csv(ROOT / 'validation_queries.csv')
    val_ids = val_gallery_df.image_id.tolist()
    N_VAL = len(val_ids)
    val_id_map = {x: i for i, x in enumerate(val_ids)}
    
    val_labels = np.full(N_VAL, -1)
    val_query_cats = {}
    for _, r in val_gt.iterrows():
        for x in r.relevant_image_ids.split():
            val_labels[val_id_map[x]] = cidx[r.category_id]
        val_query_cats[r.query_id] = r.category_id

    def eval_val_rprecision(scores):
        rp_list = []
        for _, q in val_queries_df.iterrows():
            cid = val_query_cats.get(q.query_id)
            if cid is None:
                continue
            k = cidx[cid]
            order = np.argsort(-scores[:, k])
            relevant = set(np.where(val_labels == k)[0])
            R = len(relevant)
            if R == 0:
                continue
            hits = sum(1 for idx in order[:R] if idx in relevant)
            rp_list.append(hits / R)
        return np.mean(rp_list)

    val_feats = extract_gallery_features(ROOT / 'validation_gallery', val_ids, batch_size=16, desc="val_gallery")
    
    val_S_text = (val_feats @ cat_text_feats.T).numpy()
    val_S_proto = (val_feats @ cat_proto_feats.T).numpy()
    
    val_cos = (val_feats @ train_feats.T).numpy()
    val_S_knn = np.zeros((N_VAL, N_CATS), dtype=np.float32)
    for k in range(N_CATS):
        mask = (train_labels == k)
        k_sims = val_cos[:, mask]
        val_S_knn[:, k] = np.sort(k_sims, axis=1)[:, -3:].mean(axis=1)
        
    val_S_probe = clf.decision_function(val_feats.numpy())
    val_S_ridge = ridge.decision_function(val_feats.numpy())
    
    vn_text = normalize_scores(val_S_text, 'friend')
    vn_proto = normalize_scores(val_S_proto, 'friend')
    vn_knn = normalize_scores(val_S_knn, 'friend')
    vn_probe = normalize_scores(val_S_probe, 'friend')
    vn_ridge = normalize_scores(val_S_ridge, 'friend')
    
    rp_text = eval_val_rprecision(sinkhorn_log(vn_text, 0.125))
    rp_proto = eval_val_rprecision(sinkhorn_log(vn_proto, 0.125))
    rp_knn = eval_val_rprecision(sinkhorn_log(vn_knn, 0.125))
    rp_probe = eval_val_rprecision(sinkhorn_log(vn_probe, 0.125))
    rp_ridge = eval_val_rprecision(sinkhorn_log(vn_ridge, 0.125))
    
    print(f"Validation R-Precision:")
    print(f"  Text (proper max_length): {rp_text:.5f}")
    print(f"  Visual Prototypes:       {rp_proto:.5f}")
    print(f"  K-NN Matching:           {rp_knn:.5f}")
    print(f"  Logistic Probe:          {rp_probe:.5f}")
    print(f"  Ridge Probe:             {rp_ridge:.5f}")
    
    grid_best_rp = 0.0
    for w_t in [0.25, 0.35, 0.45]:
        for w_p in [0.25, 0.35, 0.45]:
            for w_r in [0.10, 0.15, 0.20]:
                w_pr = 1.0 - (w_t + w_p + w_r)
                if w_pr < 0:
                    continue
                v_combo = w_t * vn_text + w_p * vn_proto + w_r * vn_ridge + w_pr * vn_probe
                for temp in [0.08, 0.095, 0.110, 0.125]:
                    v_bal = sinkhorn_log(v_combo, temp)
                    rp = eval_val_rprecision(v_bal)
                    if rp > grid_best_rp:
                        grid_best_rp = rp
                        best_weights = (w_t, w_p, w_r, w_pr)
                        best_temp = temp
                        
    print(f"BEST VAL R-PRECISION: {grid_best_rp:.5f}")
    print(f"Best Config: w_text={best_weights[0]:.2f}, w_proto={best_weights[1]:.2f}, w_ridge={best_weights[2]:.2f}, w_probe={best_weights[3]:.2f}, Temp={best_temp:.3f}")

Sn_text = normalize_scores(S_text, 'friend')
Sn_proto = normalize_scores(S_proto, 'friend')
Sn_knn = normalize_scores(S_knn, 'friend')
Sn_probe = normalize_scores(S_probe, 'friend')
Sn_ridge = normalize_scores(S_ridge, 'friend')

submissions_dict = {}

w_t, w_p, w_r, w_pr = best_weights
combo_best = w_t * Sn_text + w_p * Sn_proto + w_r * Sn_ridge + w_pr * Sn_probe
bal_best = sinkhorn_log(combo_best, best_temp)
df_primary = build_submission_df(bal_best, test_ids, query_categories, test_queries)
df_primary.to_csv(OUTPUT / 'submission.csv', index=False)
submissions_dict['submission.csv'] = df_primary
print(f"Generated submission.csv: w_t={w_t:.2f}, w_p={w_p:.2f}, w_r={w_r:.2f}, w_pr={w_pr:.2f}, T={best_temp}")

combo_text_proto = 0.50 * Sn_text + 0.50 * Sn_proto
bal_text_proto = sinkhorn_log(combo_text_proto, 0.110)
df_text_proto = build_submission_df(bal_text_proto, test_ids, query_categories, test_queries)
df_text_proto.to_csv(OUTPUT / 'submission_text_proto.csv', index=False)
submissions_dict['submission_text_proto.csv'] = df_text_proto

combo_all = 0.30 * Sn_text + 0.30 * Sn_proto + 0.20 * Sn_ridge + 0.10 * Sn_probe + 0.10 * Sn_knn
bal_all = sinkhorn_log(combo_all, 0.110)
df_all = build_submission_df(bal_all, test_ids, query_categories, test_queries)
df_all.to_csv(OUTPUT / 'submission_5way.csv', index=False)
submissions_dict['submission_5way.csv'] = df_all

sys.path.insert(0, str(ROOT))
try:
    from validate_submission import check
    for fname, df in submissions_dict.items():
        fpath = OUTPUT / fname
        valid_rows = check(str(fpath), data_dir=str(ROOT))
        file_sha = hashlib.sha256(fpath.read_bytes()).hexdigest()[:16]
        print(f"{fname}: VALID ({valid_rows} rows, sha256={file_sha})")
except Exception as e:
    for fname, df in submissions_dict.items():
        assert len(df) == 303, f"{fname} row mismatch"
        print(f"{fname}: {len(df)} rows")

print("Done. Files ready in /kaggle/working/")
