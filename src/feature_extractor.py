import time
import re
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
import torch
import torch.nn.functional as F

ImageFile.LOAD_TRUNCATED_IMAGES = True

def extract_tensor(out):
    if isinstance(out, torch.Tensor):
        return out
    if hasattr(out, 'pooler_output') and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, 'last_hidden_state') and out.last_hidden_state is not None:
        if out.last_hidden_state.dim() == 3:
            return out.last_hidden_state[:, 0]
        return out.last_hidden_state
    if isinstance(out, (tuple, list)):
        return extract_tensor(out[0])
    return out

def get_4views(img, crop_ratio=0.05):
    img = ImageOps.exif_transpose(img).convert('RGB')
    w, h = img.size
    crop_box = (int(w * crop_ratio), int(h * crop_ratio), int(w * (1.0 - crop_ratio)), int(h * (1.0 - crop_ratio)))
    cropped = img.crop(crop_box).resize((w, h), Image.Resampling.BICUBIC)
    return [img, img.transpose(Image.FLIP_LEFT_RIGHT), cropped, cropped.transpose(Image.FLIP_LEFT_RIGHT)]

def extract_image_features(image_paths, model, processor, device, batch_size=16):
    all_feats = []
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        b_paths = image_paths[start:end]
        v1_list, v2_list, v3_list, v4_list = [], [], [], []
        for path in b_paths:
            img = Image.open(path)
            views = get_4views(img)
            v1_list.append(views[0])
            v2_list.append(views[1])
            v3_list.append(views[2])
            v4_list.append(views[3])
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                inp1 = processor(images=v1_list, return_tensors="pt", padding=True).to(device)
                f1 = F.normalize(extract_tensor(model.get_image_features(**inp1)).float(), dim=-1)
                
                inp2 = processor(images=v2_list, return_tensors="pt", padding=True).to(device)
                f2 = F.normalize(extract_tensor(model.get_image_features(**inp2)).float(), dim=-1)
                
                inp3 = processor(images=v3_list, return_tensors="pt", padding=True).to(device)
                f3 = F.normalize(extract_tensor(model.get_image_features(**inp3)).float(), dim=-1)
                
                inp4 = processor(images=v4_list, return_tensors="pt", padding=True).to(device)
                f4 = F.normalize(extract_tensor(model.get_image_features(**inp4)).float(), dim=-1)
                
                f_avg = F.normalize((f1 + f2 + f3 + f4) / 4.0, dim=-1)
                all_feats.append(f_avg.cpu())
    return torch.cat(all_feats, dim=0)

def extract_text_features(cats_df, model, processor, device):
    cat_text_list = []
    for _, row in cats_df.iterrows():
        name = row.category_name
        desc = row.description
        prompts = [
            f"a photo of {name}.",
            f"a photograph of {name}.",
            f"an image of {name}.",
            f"{name}.",
            f"a picture of {name}.",
            f"food: {name}.",
        ]
        if isinstance(desc, str) and desc.strip():
            prompts.append(f"a photo of {name}, {desc}.")
            prompts.append(f"{name}: {desc}.")
            prompts.append(f"an image of {desc}, known as {name}.")
        if isinstance(row.alternative_names, str) and row.alternative_names.strip():
            for alt in re.split(r'[,|;/]', row.alternative_names):
                alt = alt.strip()
                if alt:
                    prompts.append(f"a photo of {alt}.")
                    prompts.append(f"{alt}.")
        if isinstance(row.train_paraphrases, str) and row.train_paraphrases.strip():
            for p in row.train_paraphrases.split('|'):
                p = p.strip()
                if p:
                    prompts.append(p)
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.float16):
                inp = processor(text=prompts, return_tensors="pt", padding="max_length", max_length=64, truncation=True).to(device)
                tf = extract_tensor(model.get_text_features(**inp)).float()
                tf = F.normalize(tf, dim=-1)
                c_feat = F.normalize(tf.mean(dim=0, keepdim=True), dim=-1)
                cat_text_list.append(c_feat.cpu())
    return torch.cat(cat_text_list, dim=0)

def compute_visual_prototypes(train_feats, train_labels, cat_text_feats, num_categories):
    cat_proto_list = []
    for k in range(num_categories):
        mask = (train_labels == k)
        if mask.sum() > 0:
            proto = train_feats[mask].mean(dim=0, keepdim=True)
            proto = F.normalize(proto, dim=-1)
        else:
            proto = cat_text_feats[k:k+1]
        cat_proto_list.append(proto)
    return torch.cat(cat_proto_list, dim=0)
