import torch, numpy as np, pandas as pd, time
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import CLIPModel,CLIPProcessor
ROOT=Path('data'); D='cuda' if torch.cuda.is_available() else 'cpu'
class G(Dataset):
 def __init__(self,ids,split):self.ids=list(ids);self.split=split
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i]; return Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'),iid
def coll(b):return [x[0] for x in b],[x[1] for x in b]
def enc(m,p,ids,split,bs=8):
 out=[];oi=[]; t=time.time()
 with torch.no_grad():
  for j,(ims,iids) in enumerate(DataLoader(G(ids,split),batch_size=bs,num_workers=0,collate_fn=coll)):
   z=m.get_image_features(**p(images=ims,return_tensors='pt').to(D)); z=z/z.norm(dim=-1,keepdim=True); out.append(z.float().cpu()); oi+=iids
   if j%20==0: print(split,j,len(ids)//bs,flush=True)
 print('encoded',split,time.time()-t,flush=True); return torch.cat(out).numpy(),oi
print('load',D,flush=True);m=CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(D).eval();p=CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14');print('loaded',flush=True)
tr=pd.read_csv(ROOT/'train.csv');va=pd.read_csv(ROOT/'validation_gallery.csv');cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};y=tr.category_id.map(cidx).to_numpy();
X,xi=enc(m,p,tr.image_id.tolist(),'train');V,vi=enc(m,p,va.image_id.tolist(),'validation');np.savez('clip_large_embeddings.npz',X=X,V=V,y=y,xi=np.array(xi),vi=np.array(vi))
gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vq=pd.read_csv(ROOT/'validation_queries.csv');truth={r.query_id:r.relevant_image_ids.split() for _,r in gt.iterrows()};
def score(S):
 vals=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];rank=[vi[i] for i in np.argsort(-S[:,k])]; vals.append(sum(x in set(truth[q.query_id]) for x in rank[:10])/10)
 return float(np.mean(vals))
for typ in ['centroid','max','mean3','mean5','mean10','logreg']:
 if typ=='logreg':
  from sklearn.linear_model import LogisticRegression
  clf=LogisticRegression(C=3,max_iter=500,multi_class='multinomial',n_jobs=-1).fit(X,y); S=clf.decision_function(V)
 else:
  S=np.zeros((len(V),len(cats)))
  for k in range(len(cats)):
   sim=V@X[y==k].T
   if typ=='centroid':a=X[y==k].mean(0);a/=np.linalg.norm(a);S[:,k]=V@a
   elif typ=='max':S[:,k]=sim.max(1)
   else:S[:,k]=np.sort(sim,axis=1)[:,-int(typ[4:]):].mean(1)
 print(typ,score(S),flush=True)
