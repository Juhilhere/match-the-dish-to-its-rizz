import timm,torch,random,numpy as np,pandas as pd,json
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';SEED=42;random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};N=len(cats)
tr=pd.read_csv(ROOT/'train.csv');tr['y']=tr.category_id.map(cidx);vi=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist();gt=pd.read_csv(ROOT/'validation_ground_truth.csv');vm={x:i for i,x in enumerate(vi)};vy=np.full(len(vi),-1)
for _,r in gt.iterrows():
 for x in r.relevant_image_ids.split():vy[vm[x]]=cidx[r.category_id]
ids=tr.image_id.tolist()+vi;ys=np.r_[tr.y.to_numpy(),vy];sps=['train']*len(tr)+['validation']*len(vi)
train_tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.7,1.0),ratio=(.85,1.18)),transforms.RandomHorizontalFlip(),transforms.RandAugment(num_ops=2,magnitude=5),transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
class DS(Dataset):
 def __len__(self):return len(ids)
 def __getitem__(self,i):return train_tf(Image.open(ROOT/f'{sps[i]}_gallery'/f'{ids[i]}.jpg').convert('RGB')),int(ys[i])
dl=DataLoader(DS(),batch_size=32,shuffle=True,num_workers=0,pin_memory=True)
m=timm.create_model('vit_base_patch16_siglip_224.v2_webli',pretrained=False,num_classes=N).to(D);m.load_state_dict(torch.load('siglip_finetuned_best.pt',map_location=D)['model'])
for p in m.parameters():p.requires_grad=False
for b in m.blocks[-2:]:
 for p in b.parameters():p.requires_grad=True
for p in m.norm.parameters():p.requires_grad=True
for p in m.head.parameters():p.requires_grad=True
opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-5,weight_decay=2e-4);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=3);lossfn=torch.nn.CrossEntropyLoss(label_smoothing=.05);hist=[]
for ep in range(1,4):
 m.train();tot=cor=0
 for x,y in dl:
  opt.zero_grad(set_to_none=True);z=m(x.to(D));l=lossfn(z,y.to(D));l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();tot+=len(y);cor+=(z.argmax(1).cpu()==y).sum().item()
 sch.step();r={'epoch':ep,'train_acc':cor/tot};hist.append(r);print(json.dumps(r),flush=True)
torch.save({'model':m.state_dict(),'history':hist,'source':'siglip_finetuned_best.pt + train + released validation labels'},'siglip_all_labeled_finetuned.pt')
