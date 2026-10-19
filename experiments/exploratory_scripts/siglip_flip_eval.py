import timm,torch,numpy as np,pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from timm.data import resolve_data_config,create_transform
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linear_sum_assignment
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vq=pd.read_csv(ROOT/'validation_queries.csv');truth={r.query_id:r.relevant_image_ids.split() for _,r in gt.iterrows()}
m=timm.create_model('vit_base_patch16_siglip_224.v2_webli',pretrained=True,num_classes=0).to(D).eval();base=create_transform(**resolve_data_config({},model=m));flip=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.RandomHorizontalFlip(p=1),transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
# use base config transform and explicit flip based on base transform impossible; construct common transform from config
norm=transforms.Normalize([.5,.5,.5],[.5,.5,.5]);tf=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),norm]);tff=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.RandomHorizontalFlip(p=1),transforms.ToTensor(),norm])
ids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist()
class G(Dataset):
 def __len__(self):return len(ids)
 def __getitem__(self,i):
  im=Image.open(ROOT/'validation_gallery'/f'{ids[i]}.jpg').convert('RGB');return tf(im),tff(im)
o=[];f=[]
with torch.no_grad():
 for j,(x,xx) in enumerate(DataLoader(G(),batch_size=32,num_workers=0)):
  o.append(m(x.to(D)).float().cpu());f.append(m(xx.to(D)).float().cpu());
V=torch.cat(o).numpy();F=torch.cat(f).numpy();V/=np.linalg.norm(V,axis=1,keepdims=True);F/=np.linalg.norm(F,axis=1,keepdims=True);z=np.load('siglip_timm_embeddings.npz');clf=LogisticRegression(C=3,max_iter=500,n_jobs=-1).fit(z['X'],z['y']);A=clf.decision_function(V);B=clf.decision_function(F)
def score(S,b=False):
 if b:rr,cc=linear_sum_assignment(np.repeat(-S,10,axis=1));ass=np.full(len(S),-1);ass[rr]=cc//10
 out=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];ord=np.argsort(-S[:,k]) if not b else np.argsort(np.where(ass==k,-S[:,k],1e9));out.append(sum(x in set(truth[q.query_id]) for x in ids[ord[:10]])/10)
 return np.mean(out)
for a in [0,.25,.5,.75,1]:print(a,score((1-a)*A+a*B),score((1-a)*A+a*B,1))
