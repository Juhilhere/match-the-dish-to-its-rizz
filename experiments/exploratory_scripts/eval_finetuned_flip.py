import timm,torch,numpy as np,pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';N=len(pd.read_csv(ROOT/'category_metadata.csv'));ids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist();norm=transforms.Normalize([.5,.5,.5],[.5,.5,.5]);tf=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),norm]);tff=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.RandomHorizontalFlip(p=1),transforms.ToTensor(),norm])
class G(Dataset):
 def __len__(self):return len(ids)
 def __getitem__(self,i):
  im=Image.open(ROOT/'validation_gallery'/f'{ids[i]}.jpg').convert('RGB');return tf(im),tff(im)
m=timm.create_model('vit_base_patch16_siglip_224.v2_webli',pretrained=False,num_classes=N).to(D);m.load_state_dict(torch.load('siglip_finetuned_best.pt',map_location=D)['model']);m.eval();A=[];B=[]
with torch.no_grad():
 for x,y in DataLoader(G(),batch_size=64,num_workers=0):A.append(m(x.to(D)).float().cpu());B.append(m(y.to(D)).float().cpu())
A=torch.cat(A).numpy();B=torch.cat(B).numpy();np.save('siglip_finetuned_val_flip_logits.npy',B)
# eval against validation ground truth
cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vq=pd.read_csv(ROOT/'validation_queries.csv');truth={r.query_id:r.relevant_image_ids.split() for _,r in gt.iterrows()};vi=np.array(ids)
def score(S):
 rr,cc=linear_sum_assignment(np.repeat(-S,10,axis=1));ass=np.full(len(S),-1);ass[rr]=cc//10;z=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];o=np.argsort(np.where(ass==k,-S[:,k],1e9));z.append(sum(x in set(truth[q.query_id]) for x in vi[o[:10]])/10)
 return np.mean(z)
for w in [0,.2,.4,.5,.6,.8,1]:print(w,score((1-w)*A+w*B),flush=True)
