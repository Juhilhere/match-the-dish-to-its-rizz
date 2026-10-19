import timm,torch,numpy as np,pandas as pd
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
ROOT=Path('data');D='cuda' if torch.cuda.is_available() else 'cpu';N=len(pd.read_csv(ROOT/'category_metadata.csv'));ids=pd.read_csv(ROOT/'test_gallery.csv').image_id.tolist();tf=transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.5,.5,.5],[.5,.5,.5])])
class G(Dataset):
 def __len__(self):return len(ids)
 def __getitem__(self,i):return tf(Image.open(ROOT/'test_gallery'/f'{ids[i]}.jpg').convert('RGB'))
m=timm.create_model('vit_base_patch16_siglip_224.v2_webli',pretrained=False,num_classes=N).to(D);ck=torch.load('siglip_all_labeled_finetuned.pt',map_location=D);m.load_state_dict(ck['model']);m.eval();o=[]
with torch.no_grad():
 for j,x in enumerate(DataLoader(G(),batch_size=64,num_workers=0)):
  o.append(m(x.to(D)).float().cpu());print(j,flush=True)
np.save('siglip_all_labeled_finetuned_test_logits.npy',torch.cat(o).numpy());print('saved',ck.get('epoch'),ck.get('score'),flush=True)
