import argparse, random, re
from pathlib import Path
import numpy as np, pandas as pd, torch
from PIL import Image
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
ROOT=Path(__file__).resolve().parent/'data'
def seed_all(s=42):
 random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
class FoodDS(Dataset):
 def __init__(self,df,root,train=False):
  self.df=df.reset_index(drop=True); self.root=Path(root)
  self.tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.72,1)) if train else transforms.Resize(256),transforms.RandomHorizontalFlip() if train else transforms.CenterCrop(224),transforms.ColorJitter(.15,.15,.15,.05) if train else transforms.Lambda(lambda x:x),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self): return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i]; im=Image.open(self.root/'train_gallery'/f'{r.image_id}.jpg').convert('RGB'); return self.tf(im),int(r.label)
class GalleryDS(Dataset):
 def __init__(self,ids,root,split):
  self.ids=list(ids); self.root=Path(root); self.split=split; self.tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 def __len__(self): return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i]; im=Image.open(self.root/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'); return self.tf(im),iid
def build_model(n):
 m=models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT); m.classifier[1]=nn.Linear(m.classifier[1].in_features,n); return m
def predict(model,loader,device):
 model.eval(); out=[]; ids=[]
 with torch.no_grad():
  for x,ii in loader: out.append(torch.softmax(model(x.to(device)),1).cpu()); ids.extend(ii)
 return torch.cat(out).numpy(),ids
def lexical_category(desc,cats):
 t=desc.lower(); choices=[]
 for _,r in cats.iterrows():
  names=[r.category_name]
  if isinstance(r.alternative_names,str) and r.alternative_names.strip(): names+=re.split(r'[,|;/]',r.alternative_names)
  for name in names:
   name=name.strip().lower()
   if name and re.search(r'(?<![a-z])'+re.escape(name)+r'(?![a-z])',t): choices.append((len(name),r.category_id))
 return max(choices)[1] if choices else None
def score_rank(ranks,gt):
 return float(np.mean([sum(x in set(ids) for x in ranks[q][:len(ids)])/len(ids) for q,ids in gt.items()]))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--epochs',type=int,default=12); ap.add_argument('--batch',type=int,default=64); ap.add_argument('--out',default='baseline_submission.csv'); args=ap.parse_args(); seed_all(42); device='cuda' if torch.cuda.is_available() else 'cpu'; print('device',device)
 cats=pd.read_csv(ROOT/'category_metadata.csv'); c2i={c:i for i,c in enumerate(cats.category_id)}; tr=pd.read_csv(ROOT/'train.csv'); tr['label']=tr.category_id.map(c2i)
 valmask=tr.groupby('label',group_keys=False).apply(lambda x:x.sample(frac=.2,random_state=42)).index; va=tr.loc[valmask]; trn=tr.drop(valmask)
 train_loader=DataLoader(FoodDS(trn,ROOT,True),batch_size=args.batch,shuffle=True,num_workers=0,pin_memory=True); hold_loader=DataLoader(FoodDS(va,ROOT,False),batch_size=args.batch*2,shuffle=False,num_workers=0,pin_memory=True)
 model=build_model(len(cats)).to(device)
 for p in model.features.parameters(): p.requires_grad=False
 opt=torch.optim.AdamW(model.classifier.parameters(),lr=2e-3,weight_decay=1e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs); lossfn=nn.CrossEntropyLoss(label_smoothing=.05)
 for ep in range(args.epochs):
  if ep==2:
   for p in model.features.parameters(): p.requires_grad=True
   opt=torch.optim.AdamW(model.parameters(),lr=2e-4,weight_decay=2e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs-2)
  model.train(); total=correct=0
  for x,y in train_loader:
   opt.zero_grad(set_to_none=True); z=model(x.to(device)); l=lossfn(z,y.to(device)); l.backward(); opt.step(); correct+=(z.argmax(1).cpu()==y).sum().item(); total+=len(y)
  sch.step(); model.eval(); hc=0
  with torch.no_grad():
   for x,y in hold_loader: hc+=(model(x.to(device)).argmax(1).cpu()==y).sum().item()
  print(f'ep {ep+1}/{args.epochs} train_acc {correct/total:.3f} hold_acc {hc/len(va):.3f}')
 torch.save(model.state_dict(),'baseline_model.pt')
 vq=pd.read_csv(ROOT/'validation_queries.csv'); gt_df=pd.read_csv(ROOT/'validation_ground_truth.csv'); gt={r.query_id:r.relevant_image_ids.split() for _,r in gt_df.iterrows()}; vids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist(); vp,vids=predict(model,DataLoader(GalleryDS(vids,ROOT,'validation'),batch_size=args.batch*2,num_workers=0),device); cidx={cid:i for i,cid in enumerate(cats.category_id)}; ranks={}; lexgood=0
 for _,q in vq.iterrows():
  cid=lexical_category(q.description,cats); lexgood+=cid==gt_df.loc[gt_df.query_id.eq(q.query_id),'category_id'].iloc[0]
  if cid is None: cid=cats.category_id.iloc[int(vp.mean(0).argmax())]
  ranks[q.query_id]=[vids[i] for i in np.argsort(-vp[:,cidx[cid]])]
 print('lexical query category accuracy',lexgood/len(vq),'val Rprecision',score_rank(ranks,gt))
 tq=pd.read_csv(ROOT/'test_queries.csv'); tids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist(); tp,tids=predict(model,DataLoader(GalleryDS(tids,ROOT,'test'),batch_size=args.batch*2,num_workers=0),device); rows=[]
 for _,q in tq.iterrows():
  cid=lexical_category(q.description,cats)
  if cid is None: cid=cats.category_id.iloc[int(tp.mean(0).argmax())]
  order=np.argsort(-tp[:,cidx[cid]])[:20]; rows.append({'query_id':q.query_id,'image_ids':' '.join(tids[i] for i in order)})
 pd.DataFrame(rows).to_csv(args.out,index=False); print('wrote',args.out)
if __name__=='__main__': main()

