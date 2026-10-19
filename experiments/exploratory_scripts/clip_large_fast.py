import torch,numpy as np,pandas as pd,time
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import CLIPModel,CLIPProcessor
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';dtype=torch.float16 if D=='cuda' else torch.float32
class G(Dataset):
 def __init__(self,ids,sp):self.ids=list(ids);self.sp=sp
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):return Image.open(ROOT/f'{self.sp}_gallery'/f'{self.ids[i]}.jpg').convert('RGB'),self.ids[i]
def coll(b):return [x[0] for x in b],[x[1] for x in b]
 def enc(m,p,ids,sp):
 o=[];ii=[];t=time.time();dl=DataLoader(G(ids,sp),batch_size=4,num_workers=0,collate_fn=coll)
 with torch.inference_mode():
  for j,(ims,s) in enumerate(dl):
   inp=p(images=ims,return_tensors='pt'); inp={k:v.to(D) for k,v in inp.items()}
   z=m.get_image_features(**inp);z=z.float();z=z/z.norm(dim=-1,keepdim=True);o.append(z.cpu());ii+=list(s)
   if j%10==0:print(sp,j,'/',len(dl),round(time.time()-t,1),flush=True)
 return torch.cat(o).numpy(),ii
print('loading',flush=True);m=CLIPModel.from_pretrained('openai/clip-vit-large-patch14',torch_dtype=dtype,local_files_only=True).to(D).eval();p=CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14',local_files_only=True);print('loaded',flush=True)
tr=pd.read_csv(ROOT/'train.csv');va=pd.read_csv(ROOT/'validation_gallery.csv');cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};y=tr.category_id.map(cidx).to_numpy();X,xi=enc(m,p,tr.image_id,'train');V,vi=enc(m,p,va.image_id,'validation');np.savez('clip_large_embeddings.npz',X=X,V=V,y=y,xi=np.array(xi),vi=np.array(vi));print('saved',X.shape,V.shape,flush=True)
