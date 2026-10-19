import timm, torch, numpy as np, pandas as pd, time
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from timm.data import resolve_data_config,create_transform
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu'
class G(Dataset):
 def __init__(self,ids,split,tf):self.ids=list(ids);self.split=split;self.tf=tf
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i];return self.tf(Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB')),iid
def enc(m,tf,ids,split):
 o=[];oi=[]
 with torch.no_grad():
  for j,(x,ii) in enumerate(DataLoader(G(ids,split,tf),batch_size=64,num_workers=0)):
   o.append(m(x.to(D)).float().cpu());oi+=list(ii)
   if j%10==0:print(split,j,flush=True)
 return torch.cat(o).numpy(),oi
print('load',flush=True);m=timm.create_model('vit_small_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0).to(D).eval();tf=create_transform(**resolve_data_config({},model=m));print('loaded',D,flush=True)
tr=pd.read_csv(ROOT/'train.csv');va=pd.read_csv(ROOT/'validation_gallery.csv');cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};y=tr.category_id.map(cidx).to_numpy();X,xi=enc(m,tf,tr.image_id.tolist(),'train');V,vi=enc(m,tf,va.image_id.tolist(),'validation');X/=np.linalg.norm(X,axis=1,keepdims=True);V/=np.linalg.norm(V,axis=1,keepdims=True);np.savez('dinov3_embeddings.npz',X=X,V=V,y=y,xi=np.array(xi),vi=np.array(vi));gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vq=pd.read_csv(ROOT/'validation_queries.csv');truth={r.query_id:r.relevant_image_ids.split() for _,r in gt.iterrows()}
def score(S):
 a=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];rank=[vi[i] for i in np.argsort(-S[:,k])];a.append(sum(x in set(truth[q.query_id]) for x in rank[:10])/10)
 return float(np.mean(a))
for typ in ['centroid','max','mean3','mean5','mean10','logreg']:
 if typ=='logreg':
  from sklearn.linear_model import LogisticRegression; S=LogisticRegression(C=3,max_iter=500,multi_class='multinomial',n_jobs=-1).fit(X,y).decision_function(V)
 else:
  S=np.zeros((len(V),len(cats)))
  for k in range(len(cats)):
   sim=V@X[y==k].T
   if typ=='centroid':a=X[y==k].mean(0);a/=np.linalg.norm(a);S[:,k]=V@a
   elif typ=='max':S[:,k]=sim.max(1)
   else:S[:,k]=np.sort(sim,axis=1)[:,-int(typ[4:]):].mean(1)
 print(typ,score(S),flush=True)
