import numpy as np
import pandas as pd

def build_submission(scores, test_ids, query_categories, test_queries, top_k=20, category_to_idx=None):
    rows = []
    for _, q in test_queries.iterrows():
        cid = query_categories[q.query_id]
        k = category_to_idx[cid] if category_to_idx is not None else cid
        top_indices = np.argsort(-scores[:, k])[:top_k]
        rows.append({
            'query_id': q.query_id,
            'image_ids': ' '.join(test_ids[idx] for idx in top_indices)
        })
    return pd.DataFrame(rows)
