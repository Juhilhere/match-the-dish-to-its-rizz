import torch,numpy as np,pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import SiglipModel,SiglipProcessor
ROOT=Path(__file__).resolve().parent/'data'
class G(Dataset):
 def __init__(self,ids,split): self.ids=list(ids); self.split=split
 def __len__(self): return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i]; return Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'),iid
def coll(b): return [x[0] for x in b],[x[1] for x in b]
def enc(m,p,ids,split,d):
 o=[];oi=[]
 with torch.no_grad():
  for ims,iids in DataLoader(G(ids,split),batch_size=32,num_workers=0,collate_fn=coll):
   z=m.get_image_features(**p(images=ims,return_tensors='pt').to(d)); z=z/z.norm(dim=-1,keepdim=True); o.append(z.cpu());oi+=iids
 return torch.cat(o).numpy(),oi
def score(r,gt): return float(np.mean([sum(x in set(v) for x in r[q][:len(v)])/len(v) for q,v in gt.items()]))
def main():
 d='cuda' if torch.cuda.is_available() else 'cpu'; m=SiglipModel.from_pretrained('google/siglip-base-patch16-224').to(d).eval(); p=SiglipProcessor.from_pretrained('google/siglip-base-patch16-224')
 cats=pd.read_csv(ROOT/'category_metadata.csv'); names=cats.category_name.tolist(); cids=cats.category_id.tolist(); cidx={c:i for i,c in enumerate(cids)}
 prompts=[f'a photo of {n}' for n in names]
 with torch.no_grad():
  tx=m.get_text_features(**p(text=prompts,return_tensors='pt',padding=True).to(d));tx=tx/tx.norm(dim=-1,keepdim=True)
 vids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist(); V,vids=enc(m,p,vids,'validation',d); S=V@tx.cpu().numpy().T
 gt_df=pd.read_csv(ROOT/'validation_ground_truth.csv');gt={r.query_id:r.relevant_image_ids.split() for _,r in gt_df.iterrows()};vq=pd.read_csv(ROOT/'validation_queries.csv');ranks={}
 for _,q in vq.iterrows():
  k=cidx[gt_df.loc[gt_df.query_id.eq(q.query_id),'category_id'].iloc[0]];ranks[q.query_id]=[vids[i] for i in np.argsort(-S[:,k])]
 print('siglip category',score(ranks,gt))
 with torch.no_grad():
  q=m.get_text_features(**p(text=vq.description.tolist(),return_tensors='pt',padding=True).to(d));q=q/q.norm(dim=-1,keepdim=True)
 Q=q.cpu().numpy();ranks={}
 for j,(_,row) in enumerate(vq.iterrows()):ranks[row.query_id]=[vids[i] for i in np.argsort(-(V@Q[j]))]
 print('siglip query',score(ranks,gt))
 tids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist();T,tids=enc(m,p,tids,'test',d); tq=pd.read_csv(ROOT/'test_queries.csv')
 with torch.no_grad():
  q=m.get_text_features(**p(text=tq.description.tolist(),return_tensors='pt',padding=True).to(d));q=q/q.norm(dim=-1,keepdim=True)
 Q=q.cpu().numpy();rows=[]
 for j,(_,row) in enumerate(tq.iterrows()):
  order=np.argsort(-(T@Q[j]))[:20];rows.append({'query_id':row.query_id,'image_ids':' '.join(tids[i] for i in order)})
 pd.DataFrame(rows).to_csv('siglip_submission.csv',index=False);print('wrote siglip_submission.csv')
if __name__=='__main__':main()

