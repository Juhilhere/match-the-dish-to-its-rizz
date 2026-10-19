import timm,torch,numpy as np,pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats)
class G(Dataset):
 def __init__(self,ids,sp):self.ids=list(ids);self.sp=sp;self.tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):return self.tf(Image.open(ROOT/f'{self.sp}_gallery'/f'{self.ids[i]}.jpg').convert('RGB')),self.ids[i]
m=timm.create_model('convnext_tiny.fb_in22k',pretrained=False,num_classes=N).to(D);ck=torch.load('convnext_finetuned_best.pt',map_location=D);m.load_state_dict(ck['model']);m.eval();ids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist();o=[]
with torch.no_grad():
 for x,_ in DataLoader(G(ids,'validation'),batch_size=128):o.append(m(x.to(D)).float().cpu())
S=torch.cat(o).numpy();np.save('convnext_finetuned_val_logits.npy',S);print('loaded checkpoint',ck.get('epoch'),ck.get('score'))
# evaluate raw/balanced and blends
vi=np.array(ids);gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vq=pd.read_csv(ROOT/'validation_queries.csv');truth={r.query_id:r.relevant_image_ids.split() for _,r in gt.iterrows()}
z=np.load('clip_embeddings.npz');X,V,y=z['X'],z['V'],z['y'];from sklearn.linear_model import LogisticRegression;L=LogisticRegression(C=3,max_iter=500,n_jobs=-1).fit(X,y).decision_function(V)
def score(A,b=False):
 if b: rr,cc=linear_sum_assignment(np.repeat(-A,10,axis=1));ass=np.full(len(A),-1);ass[rr]=cc//10
 a=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];ord=np.argsort(-A[:,k]) if not b else np.argsort(np.where(ass==k,-A[:,k],1e9));a.append(sum(x in set(truth[q.query_id]) for x in vi[ord[:10]])/10)
 return np.mean(a)
for nm,A in [('fine',S),('clip',L),('blend',.5*S+.5*L),('blend2',.3*S+.7*L)]:print(nm,score(A),score(A,1))
