import torch, numpy as np, pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from transformers import CLIPModel,CLIPProcessor
from sklearn.linear_model import LogisticRegression
ROOT=Path(__file__).resolve().parent/'data'
class G(Dataset):
 def __init__(self,ids,split): self.ids=list(ids); self.split=split
 def __len__(self): return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i]; return Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'),iid
def coll(b): return [x[0] for x in b],[x[1] for x in b]
def enc(model,proc,ids,split,device):
 out=[]; oi=[]
 with torch.no_grad():
  for ims,iids in DataLoader(G(ids,split),batch_size=64,num_workers=0,collate_fn=coll):
   z=model.get_image_features(**proc(images=ims,return_tensors='pt').to(device)); z=z/z.norm(dim=-1,keepdim=True); out.append(z.cpu()); oi+=iids
 return torch.cat(out).numpy(),oi
def score(ranks,gt): return float(np.mean([sum(x in set(v) for x in ranks[q][:len(v)])/len(v) for q,v in gt.items()]))
def main():
 d='cuda' if torch.cuda.is_available() else 'cpu'; model=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(d).eval(); proc=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
 cats=pd.read_csv(ROOT/'category_metadata.csv'); cidx={c:i for i,c in enumerate(cats.category_id)}
 tr=pd.read_csv(ROOT/'train.csv'); tr_ids=tr.image_id.tolist(); y=tr.category_id.map(cidx).to_numpy(); X,ids=enc(model,proc,tr_ids,'train',d)
 vids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist(); V,vids=enc(model,proc,vids,'validation',d)
 gt_df=pd.read_csv(ROOT/'validation_ground_truth.csv'); gt={r.query_id:r.relevant_image_ids.split() for _,r in gt_df.iterrows()}; vq=pd.read_csv(ROOT/'validation_queries.csv')
 means=np.stack([X[y==k].mean(0) for k in range(len(cats))]); means/=np.linalg.norm(means,axis=1,keepdims=True)
 centroid=V@means.T
 # max and top-k train similarity per class
 sim=V@X.T; knn=np.zeros((len(V),len(cats)))
 for k in range(len(cats)):
  vals=np.sort(sim[:,y==k],axis=1)[:,-5:]; knn[:,k]=vals.mean(1)
 clf=LogisticRegression(C=3,max_iter=300,multi_class='multinomial',n_jobs=-1).fit(X,y); logit=clf.decision_function(V)
 for nm,S in [('centroid',centroid),('knn5',knn),('logreg',logit),('blend',.5*centroid+.5*logit/np.std(logit)*np.std(centroid))]:
  ranks={}
  for _,q in vq.iterrows():
   k=cidx[gt_df.loc[gt_df.query_id.eq(q.query_id),'category_id'].iloc[0]]; ranks[q.query_id]=[vids[i] for i in np.argsort(-S[:,k])]
  print(nm,score(ranks,gt))
 # test images and query embedding, write best probe and query variants
 tids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist(); T,tids=enc(model,proc,tids,'test',d); tsim=T@X.T; tknn=np.zeros((len(T),len(cats)))
 for k in range(len(cats)): tknn[:,k]=np.sort(tsim[:,y==k],axis=1)[:,-5:].mean(1)
 tcent=T@means.T; tlog=clf.decision_function(T); tq=pd.read_csv(ROOT/'test_queries.csv'); rows=[]
 for _,q in tq.iterrows():
  # exact lexical category is used to select the category; probe scores rank images within it
  text=q.description.lower(); cand=[(len(n),i) for i,n in enumerate(cats.category_name.str.lower()) if n in text]; k=max(cand)[1] if cand else int(tcent.mean(0).argmax())
  S=.6*tknn[:,k]+.4*tcent[:,k]; order=np.argsort(-S)[:20]; rows.append({'query_id':q.query_id,'image_ids':' '.join(tids[i] for i in order)})
 pd.DataFrame(rows).to_csv('probe_submission.csv',index=False); print('wrote probe_submission.csv')
 np.savez('clip_embeddings.npz',X=X,y=y,V=V,ids=np.array(ids),vids=np.array(vids),means=means)
if __name__=='__main__': main()

