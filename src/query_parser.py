import re
import pandas as pd

def build_aliases(cats_df, train_queries_df=None):
    aliases = {}
    for _, r in cats_df.iterrows():
        a = {r.category_name.lower()}
        if isinstance(r.alternative_names, str) and r.alternative_names.strip():
            a |= {x.strip().lower() for x in re.split(r'[,|;/]', r.alternative_names) if x.strip()}
        aliases[r.category_id] = a
    if train_queries_df is not None:
        for _, r in train_queries_df.iterrows():
            aliases.setdefault(r.category_id, set()).add(r.category_name.lower())
    return aliases

def parse_query(text, aliases):
    hits = []
    t = text.lower()
    for cid, aa in aliases.items():
        for a in aa:
            if re.search(r'(?<![a-z])' + re.escape(a) + r'(?![a-z])', t):
                hits.append((len(a), cid))
    return max(hits)[1] if hits else None

def map_queries(queries_df, aliases):
    mapping = {}
    for _, q in queries_df.iterrows():
        cid = parse_query(q.description, aliases)
        if cid is None:
            raise ValueError(f"Failed to parse query: {q.description}")
        mapping[q.query_id] = cid
    return mapping
