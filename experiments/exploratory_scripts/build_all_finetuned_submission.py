import numpy as np,pandas as pd,re
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
ROOT='data';cats=pd.read_csv(ROOT+'/category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats);z=np.load('siglip_timm_embeddings.npz');X,V,y,vi=z['X'],z['V'],z['y'],z['vi'].astype(str);T=np.load('siglip_timm_test_embeddings.npy');F=np.load('siglip_all_labeled_finetuned_test_logits.npy');gt=pd.read_csv(ROOT+'/validation_ground_truth.csv');vy=np.full(len(vi),-1)
for _,r in gt.iterrows():
 for x in r.relevant_image_ids.split():vy[np.where(vi==x)[0][0]]=cidx[r.category_id]
L=LogisticRegression(C=3,max_iter=500,n_jobs=-1).fit(np.r_[X,V],np.r_[y,vy]).decision_function(T);S=.6*F+.4*L;tids=pd.read_csv(ROOT+'/test_gallery.csv').image_id.to_numpy();A=np.vstack([S,np.zeros((20,N),dtype=S.dtype)]);rr,cc=linear_sum_assignment(np.repeat(-A,20,axis=1));ass=np.full(len(A),-1);ass[rr]=cc//20;real=ass[:len(T)];print('counts',np.bincount(real,minlength=N).tolist())
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
 k=cidx[parse(q.description)];inds=np.where(real==k)[0];order=inds[np.argsort(-S[inds,k])][:20]
 if len(order)<20:
  have=set(order);order=np.r_[order,[i for i in np.argsort(-S[:,k]) if i not in have][:20-len(order)]]
 rows.append({'query_id':q.query_id,'image_ids':' '.join(tids[order[:20]])})
pd.DataFrame(rows).to_csv('siglip_all_labeled_finetuned_submission.csv',index=False)
