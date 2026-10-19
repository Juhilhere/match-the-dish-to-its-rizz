import torch,numpy as np,pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import CLIPModel,CLIPProcessor
ROOT=Path(__file__).resolve().parent/'data'
class G(Dataset):
 def __init__(self,ids,split):self.ids=list(ids);self.split=split
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i];return Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'),iid
def coll(b):return [x[0] for x in b],[x[1] for x in b]
def enc(m,p,ids,split,d):
 o=[];oi=[]
 with torch.no_grad():
  for ims,iids in DataLoader(G(ids,split),batch_size=32,num_workers=0,collate_fn=coll):
   z=m.get_image_features(**p(images=ims,return_tensors='pt').to(d));z=z/z.norm(dim=-1,keepdim=True);o.append(z.cpu());oi+=iids
 return torch.cat(o).numpy(),oi
def score(r,gt):return float(np.mean([sum(x in set(v) for x in r[q][:len(v)])/len(v) for q,v in gt.items()]))
def main():
 d='cuda' if torch.cuda.is_available() else 'cpu';m=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(d);m.load_state_dict(torch.load('clip_finetuned.pt',map_location=d));m.eval();p=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32');cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)}
 tr=pd.read_csv(ROOT/'train.csv');X,tri=enc(m,p,tr.image_id.tolist(),'train',d);y=tr.category_id.map(cidx).to_numpy();V,vi=enc(m,p,pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist(),'validation',d);sim=V@X.T;gtd=pd.read_csv(ROOT/'validation_ground_truth.csv');gt={r.query_id:r.relevant_image_ids.split() for _,r in gtd.iterrows()};vq=pd.read_csv(ROOT/'validation_queries.csv')
 for typ in ['max','mean3','mean5','mean10']:
  S=np.zeros((len(V),len(cats)))
  for k in range(len(cats)):
   a=np.sort(sim[:,y==k],axis=1)
   if typ=='max':S[:,k]=a[:,-1]
   else:S[:,k]=a[:,-int(typ[4:]):].mean(1)
  r={}
  for _,q in vq.iterrows():
   k=cidx[gtd.loc[gtd.query_id.eq(q.query_id),'category_id'].iloc[0]];r[q.query_id]=[vi[i] for i in np.argsort(-S[:,k])]
  print(typ,score(r,gt))

if __name__=='__main__':main()
