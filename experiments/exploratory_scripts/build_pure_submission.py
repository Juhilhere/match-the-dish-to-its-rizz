import numpy as np,pandas as pd,re
from scipy.optimize import linear_sum_assignment
ROOT='data';cats=pd.read_csv(ROOT+'/category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats);F=np.load('siglip_all_labeled_finetuned_test_logits.npy');tids=pd.read_csv(ROOT+'/test_gallery.csv').image_id.to_numpy();A=np.vstack([F,np.zeros((20,N),dtype=F.dtype)]);rr,cc=linear_sum_assignment(np.repeat(-A,20,axis=1));ass=np.full(len(A),-1);ass[rr]=cc//20;real=ass[:len(F)]
trq=pd.read_csv(ROOT+'/train_queries.csv');aliases={}
for _,r in cats.iterrows():
 a={r.category_name.lower()}
 if isinstance(r.alternative_names,str) and r.alternative_names.strip():a|={x.strip().lower() for x in re.split(r'[,|;/]',r.alternative_names) if x.strip()}
 aliases[r.category_id]=a
for _,r in trq.iterrows():aliases.setdefault(r.category_id,set()).add(r.category_name.lower())
def parse(t):
 h=[]
 for cid,aa in aliases.items():
  for a in aa:
   if re.search(r'(?<![a-z])'+re.escape(a)+r'(?![a-z])',t.lower()):h.append((len(a),cid))
 return max(h)[1]
rows=[]
for _,q in pd.read_csv(ROOT+'/test_queries.csv').iterrows():
 k=cidx[parse(q.description)];inds=np.where(real==k)[0];order=inds[np.argsort(-F[inds,k])][:20]
 if len(order)<20:
  have=set(order);order=np.r_[order,[i for i in np.argsort(-F[:,k]) if i not in have][:20-len(order)]]
 rows.append({'query_id':q.query_id,'image_ids':' '.join(tids[order[:20]])})
pd.DataFrame(rows).to_csv('siglip_all_labeled_finetuned_pure_submission.csv',index=False)
