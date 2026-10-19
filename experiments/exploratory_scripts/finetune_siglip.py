import timm,torch,random,numpy as np,pandas as pd,json,time
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from scipy.optimize import linear_sum_assignment
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats);tr=pd.read_csv(ROOT/'train.csv');tr['y']=tr.category_id.map(cidx)
vids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist();gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vq=pd.read_csv(ROOT/'validation_queries.csv');vy=np.full(len(vids),-1);vm={x:i for i,x in enumerate(vids)}
for _,r in gt.iterrows():
 for x in r.relevant_image_ids.split():vy[vm[x]]=cidx[r.category_id]
train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.7,1.0),ratio=(.85,1.18)),transforms.RandomHorizontalFlip(),transforms.RandAugment(num_ops=2,magnitude=5),transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
eval_tf=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
class DS(Dataset):
 def __init__(self,ids,sp,tf,ys):self.ids=list(ids);self.sp=sp;self.tf=tf;self.ys=ys
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):return self.tf(Image.open(ROOT/f'{self.sp}_gallery'/f'{self.ids[i]}.jpg').convert('RGB')),int(self.ys[i])
train_dl=DataLoader(DS(tr.image_id,'train',train_tf,tr.y.to_numpy()),batch_size=32,shuffle=True,num_workers=0,pin_memory=True);val_dl=DataLoader(DS(vids,'validation',eval_tf,vy),batch_size=64,shuffle=False,num_workers=0,pin_memory=True)
m=timm.create_model('vit_base_patch16_siglip_224.v2_webli',pretrained=True,num_classes=N).to(D)
for p in m.parameters():p.requires_grad=False
for p in m.head.parameters():p.requires_grad=True
opt=torch.optim.AdamW(m.head.parameters(),lr=2e-3,weight_decay=1e-4);lossfn=torch.nn.CrossEntropyLoss(label_smoothing=.05)
def ev():
 m.eval();z=[];yy=[]
 with torch.no_grad():
  for x,y in val_dl:z.append(m(x.to(D)).float().cpu());yy.append(y)
 Z=torch.cat(z).numpy();Y=torch.cat(yy).numpy();acc=float((Z.argmax(1)==Y).mean());a=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];ord=np.argsort(-Z[:,k]);a.append(sum(Y[ord[:10]]==k)/10)
 # balanced assignment based on logits
 rr,cc=linear_sum_assignment(np.repeat(-Z,10,axis=1));ass=np.full(len(Z),-1);ass[rr]=cc//10;b=[]
 for _,q in vq.iterrows():
  k=cidx[gt.loc[gt.query_id.eq(q.query_id),'category_id'].iloc[0]];ord=np.argsort(np.where(ass==k,-Z[:,k],1e9));b.append(sum(Y[ord[:10]]==k)/10)
 return acc,float(np.mean(a)),float(np.mean(b)),Z
hist=[];best=(-1,None)
for ep in range(1,9):
 if ep==3:
  for block in m.blocks[-2:]:
   for p in block.parameters():p.requires_grad=True
  for p in m.norm.parameters():p.requires_grad=True
  opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-5,weight_decay=2e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=6)
 m.train();tot=cor=0
 for x,y in train_dl:
  opt.zero_grad(set_to_none=True);z=m(x.to(D));l=lossfn(z,y.to(D));l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();tot+=len(y);cor+=(z.argmax(1).cpu()==y).sum().item()
 if ep>=3:sch.step()
 acc,raw,bal,Z=ev();r={'epoch':ep,'train_acc':cor/tot,'val_acc':acc,'raw':raw,'balanced':bal};hist.append(r);print(json.dumps(r),flush=True)
 if bal>best[0]:best=(bal,ep);torch.save({'model':m.state_dict(),'epoch':ep,'score':bal,'history':hist},'siglip_finetuned_best.pt');np.save('siglip_finetuned_val_logits.npy',Z)
print('BEST',best,flush=True);pd.DataFrame(hist).to_csv('siglip_finetune_history.csv',index=False)
