import torch,numpy as np,pandas as pd,random
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from transformers import CLIPModel,CLIPProcessor
ROOT=Path(__file__).resolve().parent/'data'
class Train(Dataset):
 def __init__(self,df):
  self.df=df.reset_index(drop=True);self.tf=transforms.Compose([transforms.RandomResizedCrop(224,scale=(.65,1)),transforms.RandomHorizontalFlip()])
 def __len__(self):return len(self.df)
 def __getitem__(self,i):
  r=self.df.iloc[i];return self.tf(Image.open(ROOT/'train_gallery'/f'{r.image_id}.jpg').convert('RGB')),int(r.k)
class Gal(Dataset):
 def __init__(self,ids,split):self.ids=list(ids);self.split=split
 def __len__(self):return len(self.ids)
 def __getitem__(self,i):
  iid=self.ids[i];return Image.open(ROOT/f'{self.split}_gallery'/f'{iid}.jpg').convert('RGB'),iid
def coll(b):return [x[0] for x in b],[x[1] for x in b]
def score(r,gt):return float(np.mean([sum(x in set(v) for x in r[q][:len(v)])/len(v) for q,v in gt.items()]))
def main():
 torch.manual_seed(42);d='cuda' if torch.cuda.is_available() else 'cpu';m=CLIPModel.from_pretrained('openai/clip-vit-base-patch32').to(d);p=CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32');m.eval()
 cats=pd.read_csv(ROOT/'category_metadata.csv');cidx={c:i for i,c in enumerate(cats.category_id)};names=cats.category_name.tolist();prompts=sum(([f'a photo of {n}',f'a picture of {n}',f'a food photograph of {n}',f'{n} dish'] for n in names),[])
 with torch.no_grad():
  te=p(text=prompts,return_tensors='pt',padding=True).to(d);tx=m.get_text_features(**te);tx=tx/tx.norm(dim=-1,keepdim=True);tx=tx.reshape(len(names),4,-1).mean(1);tx=tx/tx.norm(dim=-1,keepdim=True)
 tr=pd.read_csv(ROOT/'train.csv');tr['k']=tr.category_id.map(cidx); loader=DataLoader(Train(tr),batch_size=32,shuffle=True,num_workers=0,collate_fn=lambda b:( [x[0] for x in b],torch.tensor([x[1] for x in b]) ))
 for p0 in m.parameters():p0.requires_grad=False
 for layer in m.vision_model.encoder.layers[-4:]:
  for p0 in layer.parameters():p0.requires_grad=True
 for p0 in m.visual_projection.parameters():p0.requires_grad=True
 params=[p0 for p0 in m.parameters() if p0.requires_grad];opt=torch.optim.AdamW(params,lr=2e-6,weight_decay=.01);sc=torch.cuda.amp.GradScaler(enabled=(d=='cuda'));lossfn=torch.nn.CrossEntropyLoss(label_smoothing=.05)
 for ep in range(5):
  m.train();tot=cor=0
  for ims,y in loader:
   inp=p(images=ims,return_tensors='pt').to(d);opt.zero_grad(set_to_none=True)
   with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=(d=='cuda')):
    z=m.get_image_features(**inp);z=z/z.norm(dim=-1,keepdim=True);logits=m.logit_scale.exp()*z@tx.T;loss=lossfn(logits,y.to(d))
   sc.scale(loss).backward();sc.step(opt);sc.update();tot+=len(y);cor+=(logits.argmax(1).cpu()==y).sum().item()
  print('ep',ep+1,'acc',cor/tot)
 # eval galleries

 def enc(ids,split):
  out=[];oi=[];m.eval()
  with torch.no_grad():
   for ims,iids in DataLoader(Gal(ids,split),batch_size=32,num_workers=0,collate_fn=coll):
    z=m.get_image_features(**p(images=ims,return_tensors='pt').to(d));z=z/z.norm(dim=-1,keepdim=True);out.append(z.cpu());oi+=iids
  return torch.cat(out).numpy(),oi
 vids=pd.read_csv(ROOT/'validation_gallery.csv').image_id.tolist();V,vids=enc(vids,'validation');S=V@tx.detach().cpu().numpy().T;gtd=pd.read_csv(ROOT/'validation_ground_truth.csv');gt={r.query_id:r.relevant_image_ids.split() for _,r in gtd.iterrows()};vq=pd.read_csv(ROOT/'validation_queries.csv');r={}
 for _,q in vq.iterrows():
  k=cidx[gtd.loc[gtd.query_id.eq(q.query_id),'category_id'].iloc[0]];r[q.query_id]=[vids[i] for i in np.argsort(-S[:,k])]
 print('finetuned val',score(r,gt));torch.save(m.state_dict(),'clip_finetuned.pt')
 tids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist();T,tids=enc(tids,'test');tq=pd.read_csv(ROOT/'test_queries.csv');qemb=[]
 with torch.no_grad():
  for st in range(0,len(tq),32):
   q=m.get_text_features(**p(text=tq.description.iloc[st:st+32].tolist(),return_tensors='pt',padding=True).to(d));q=q/q.norm(dim=-1,keepdim=True);qemb.append(q.cpu())
 qemb=torch.cat(qemb).numpy();rows=[]
 for j,(_,q) in enumerate(tq.iterrows()):
  order=np.argsort(-(T@qemb[j]))[:20];rows.append({'query_id':q.query_id,'image_ids':' '.join(tids[i] for i in order)})
 pd.DataFrame(rows).to_csv('clip_finetuned_submission.csv',index=False);print('wrote clip_finetuned_submission.csv')
if __name__=='__main__':main()

