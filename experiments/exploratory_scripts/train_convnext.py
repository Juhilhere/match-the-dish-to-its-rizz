import timm,torch,random,numpy as np,pandas as pd,time,json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
ROOT=Path('data'); D='cuda' if torch.cuda.is_available() else 'cpu'; SEED=42
random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats)
tr=pd.read_csv(ROOT/'train.csv'); tr['y']=tr.category_id.map(cidx)
val_ids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist(); gt=pd.read_csv(ROOT/'validation_ground_truth.csv'); vtruth={r.query_id:r.relevant_image_ids.split() for _,r in gt.iterrows()}; vq=pd.read_csv(ROOT/'validation_queries.csv')
# label for each validation gallery image can be recovered from ground truth membership; all 10 images per category
val_y=np.full(len(val_ids),-1); vidx={x:i for i,x in enumerate(val_ids)}
for _,r in gt.iterrows():
 k=cidx[r.category_id]
 for x in r.relevant_image_ids.split(): val_y[vidx[x]]=k
assert (val_y>=0).all()
train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(0.65,1.0),ratio=(0.8,1.25)),transforms.RandomHorizontalFlip(),transforms.RandAugment(num_ops=2,magnitude=7),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
eval_tf=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class DS(Dataset):
 def __init__(self,ids,sp,tf,ys=None):self.ids=list(ids);self.sp=sp;self.tf=tf;self.ys=ys
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  x=self.tf(Image.open(ROOT/f'{self.sp}_gallery'/f'{self.ids[i]}.jpg').convert('RGB'))
  return (x,int(self.ys[i])) if self.ys is not None else (x,self.ids[i])
train_dl=DataLoader(DS(tr.image_id,'train',train_tf,tr.y.to_numpy()),batch_size=64,shuffle=True,num_workers=0,pin_memory=True)
val_dl=DataLoader(DS(val_ids,'validation',eval_tf,val_y),batch_size=128,shuffle=False,num_workers=0,pin_memory=True)
m=timm.create_model('convnext_tiny.fb_in22k',pretrained=True,num_classes=N).to(D)
# warmup classifier, then full fine-tune
for p in m.parameters():p.requires_grad=False
for p in m.head.parameters():p.requires_grad=True
opt=torch.optim.AdamW(m.head.parameters(),lr=3e-3,weight_decay=1e-4)
lossfn=torch.nn.CrossEntropyLoss(label_smoothing=.05)
def evaluate():
 m.eval();allz=[];ally=[]
 with torch.no_grad():
  for x,y in val_dl: allz.append(m(x.to(D)).float().cpu());ally.append(y)
 Z=torch.cat(allz).numpy();Y=torch.cat(ally).numpy();acc=(Z.argmax(1)==Y).mean();
 vals=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];order=np.argsort(-Z[:,k]);vals.append(np.mean(Y[order[:10]]==k))
 return float(acc),float(np.mean(vals)),Z
best=(-1,None);history=[]
for ep in range(1,26):
 if ep==4:
  for p in m.parameters():p.requires_grad=True
  opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=2e-4)
  sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=22)
 m.train();tot=cor=0
 for x,y in train_dl:
  opt.zero_grad(set_to_none=True);z=m(x.to(D));l=lossfn(z,y.to(D));l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();tot+=len(y);cor+=(z.argmax(1).cpu()==y).sum().item()
 if ep>=4:sch.step()
 acc,sc,Z=evaluate(); history.append({'epoch':ep,'train_acc':cor/tot,'val_acc':acc,'rprecision':sc});print(json.dumps(history[-1]),flush=True)
 if sc>best[0]:best=(sc,ep);torch.save({'model':m.state_dict(),'epoch':ep,'score':sc,'history':history},'convnext_finetuned_best.pt')
print('BEST',best);pd.DataFrame(history).to_csv('convnext_finetune_history.csv',index=False)
