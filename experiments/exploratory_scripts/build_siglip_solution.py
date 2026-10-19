import timm,torch,numpy as np,pandas as pd,re,json,time
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from timm.data import resolve_data_config,create_transform
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats)
# cache/reuse train embeddings; test uses same frozen generic model
z=np.load('siglip_timm_embeddings.npz');X=z['X'];y=z['y'];
m=timm.create_model('vit_base_patch16_siglip_224.v2_webli',pretrained=True,num_classes=0).to(D).eval();tf=create_transform(**resolve_data_config({},model=m))
class G(Dataset):
 def __init__(self,ids):self.ids=list(ids)
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):return tf(Image.open(ROOT/'test_gallery'/f'{self.ids[i]}.jpg').convert('RGB')),self.ids[i]
tids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist(); out=[];oi=[]
with torch.no_grad():
 for j,(xx,ii) in enumerate(DataLoader(G(tids),batch_size=32,num_workers=0)):
  out.append(m(xx.to(D)).float().cpu());oi+=list(ii)
  if j%10==0: print('test batch',j,flush=True)
T=torch.cat(out).numpy();T/=np.linalg.norm(T,axis=1,keepdims=True);np.save('siglip_timm_test_embeddings.npy',T)
clf=LogisticRegression(C=3,max_iter=500,n_jobs=-1).fit(X,y);S=clf.decision_function(T)
# Assign 20 slots/category, padding with 20 dummy rows. Dummy score is zero after per-row normalization of scores.
# Work in a score scale where real image evidence dominates the dummy baseline.
S2=np.vstack([S,np.zeros((20,N),dtype=S.dtype)])
rr,cc=linear_sum_assignment(np.repeat(-S2,20,axis=1));assigned=np.full(len(S2),-1);assigned[rr]=cc//20
real_assigned=assigned[:len(T)];
print('assigned counts',np.bincount(real_assigned[real_assigned>=0],minlength=N).tolist(),flush=True)
# Parse every test query to its exact category using metadata names/aliases and training query labels.
aliases={}
for _,r in cats.iterrows():
 a={r.category_name.lower()}
 if isinstance(r.alternative_names,str) and r.alternative_names.strip():a|={x.strip().lower() for x in re.split(r'[,|;/]',r.alternative_names) if x.strip()}
 aliases[r.category_id]=a
trq=pd.read_csv(ROOT/'train_queries.csv')
for _,r in trq.iterrows():aliases.setdefault(r.category_id,set()).add(r.category_name.lower())
def parse(t):
 hits=[];t=t.lower()
 for cid,aa in aliases.items():
  for a in aa:
   if re.search(r'(?<![a-z])'+re.escape(a)+r'(?![a-z])',t):hits.append((len(a),cid))
 return max(hits)[1] if hits else None
tq=pd.read_csv(ROOT/'test_queries.csv');rows=[];missing=[]
for _,q in tq.iterrows():
 cid=parse(q.description)
 if cid is None:missing.append((q.query_id,q.description));continue
 k=cidx[cid];inds=np.where(real_assigned==k)[0];order=inds[np.argsort(-S[inds,k])][:20]
 # If dummy quota left fewer than 20 real images, append highest-scoring unassigned images as a guarded fallback.
 if len(order)<20:
  extra=[i for i in np.argsort(-S[:,k]) if i not in set(order)][:20-len(order)];order=np.r_[order,extra]
 rows.append({'query_id':q.query_id,'image_ids':' '.join(oi[i] for i in order[:20])})
print('queries',len(rows),'missing',missing,flush=True)
out=pd.DataFrame(rows);out.to_csv('siglip_balanced_submission.csv',index=False);print('wrote siglip_balanced_submission.csv',flush=True)
