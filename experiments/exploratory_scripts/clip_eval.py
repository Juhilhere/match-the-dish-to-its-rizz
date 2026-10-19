import re, torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import CLIPModel, CLIPProcessor
ROOT=Path(__file__).resolve().parent/'data'
class G(Dataset):
 def __init__(self,ids,split):
  self.ids=list(ids); self.split=split; self.tf=None
 def __len__(self): return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i]; return Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'),iid
def score(ranks,gt): return float(np.mean([sum(x in set(v) for x in ranks[q][:len(v)])/len(v) for q,v in gt.items()]))
def main():
 device='cuda' if torch.cuda.is_available() else 'cpu'; model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(device).eval(); proc=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
 cats=pd.read_csv(ROOT/'category_metadata.csv'); names=cats.category_name.tolist(); cids=cats.category_id.tolist(); cidx={c:i for i,c in enumerate(cids)}
 prompts=sum(([f'a photo of {n}',f'a picture of {n}',f'a food photograph of {n}',f'{n} dish'] for n in names),[])
 with torch.no_grad():
  te=proc(text=prompts,return_tensors='pt',padding=True).to(device); tx=model.get_text_features(**te); tx=tx/tx.norm(dim=-1,keepdim=True); tx=tx.reshape(len(names),4,-1).mean(1); tx=tx/tx.norm(dim=-1,keepdim=True)
 ids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist(); ds=G(ids,'validation')
 def coll(b): return [x[0] for x in b],[x[1] for x in b]
 il=[]; outids=[]
 with torch.no_grad():
  for ims,iids in DataLoader(ds,batch_size=64,num_workers=0,collate_fn=coll):
   inp=proc(images=ims,return_tensors='pt').to(device); z=model.get_image_features(**inp); z=z/z.norm(dim=-1,keepdim=True); il.append(z.cpu()); outids+=iids
 ie=torch.cat(il).numpy(); sims=ie@tx.cpu().numpy().T
 vq=pd.read_csv(ROOT/'validation_queries.csv'); gt_df=pd.read_csv(ROOT/'validation_ground_truth.csv'); gt={r.query_id:r.relevant_image_ids.split() for _,r in gt_df.iterrows()}
 ranks1={}; ranks2={}; qemb=[]
 with torch.no_grad():
  for st in range(0,len(vq),64):
   desc=vq.description.iloc[st:st+64].tolist(); inp=proc(text=desc,return_tensors='pt',padding=True).to(device); q=model.get_text_features(**inp); q=q/q.norm(dim=-1,keepdim=True); qemb.append(q.cpu())
 qemb=torch.cat(qemb).numpy()
 for j,(_,q) in enumerate(vq.iterrows()):
  cid=gt_df.loc[gt_df.query_id.eq(q.query_id),'category_id'].iloc[0]; k=cidx[cid]; ranks1[q.query_id]=[outids[i] for i in np.argsort(-sims[:,k])]; ranks2[q.query_id]=[outids[i] for i in np.argsort(-(ie@qemb[j]))]
 print('clip category val Rprecision',score(ranks1,gt)); print('clip query val Rprecision',score(ranks2,gt))
 np.save('clip_val_image.npy',ie); np.save('clip_val_ids.npy',np.array(outids))
 # test rankings using category prompt and query embedding
 tids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist(); il=[]; outids=[]; ds=G(tids,'test')
 with torch.no_grad():
  for ims,iids in DataLoader(ds,batch_size=64,num_workers=0,collate_fn=coll):
   inp=proc(images=ims,return_tensors='pt').to(device); z=model.get_image_features(**inp); z=z/z.norm(dim=-1,keepdim=True); il.append(z.cpu()); outids+=iids
 tie=torch.cat(il).numpy(); tq=pd.read_csv(ROOT/'test_queries.csv'); rows=[]; qemb=[]
 with torch.no_grad():
  for st in range(0,len(tq),64):
   inp=proc(text=tq.description.iloc[st:st+64].tolist(),return_tensors='pt',padding=True).to(device); q=model.get_text_features(**inp); q=q/q.norm(dim=-1,keepdim=True); qemb.append(q.cpu())
 qemb=torch.cat(qemb).numpy()
 for j,(_,q) in enumerate(tq.iterrows()):
  order=np.argsort(-(tie@qemb[j]))[:20]; rows.append({'query_id':q.query_id,'image_ids':' '.join(outids[i] for i in order)})
 pd.DataFrame(rows).to_csv('clip_submission.csv',index=False)
 print('wrote clip_submission.csv')
if __name__=='__main__': main()

