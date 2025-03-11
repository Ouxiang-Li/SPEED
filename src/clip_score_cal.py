import os, sys, re, pdb
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import argparse
import torch
import torch_fidelity
import pandas as pd

from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import CLIPModel, CLIPProcessor, CLIPTokenizer


class Generate_Dataset(Dataset):
    def __init__(self, path, content, sub_root):
        super().__init__()
        root_path = os.path.join(path, content, sub_root)
        self.content = content
        self.images = [os.path.join(root_path, name) for name in os.listdir(root_path)]
        if content == 'coco':
            df = pd.read_csv("data/mscoco.csv")
            self.texts = [df.loc[df['image_id'].isin([int(os.path.basename(x).replace('COCO_val2014_', '').split('.')[0])]), 'text'].tolist()[0] for x in self.images]
        else:
            self.texts = [('_').join(x.split('/')[-1].split('_')[:-1]) for x in self.images]
    
    def __len__(self,):
        return len(self.images)
    
    def __getitem__(self, idx):
        return {'text': self.texts[idx], 'image': self.images[idx], 'content': self.content}


class CLIP_Score():
    def __init__(self, version='openai/clip-vit-large-patch14', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = CLIPModel.from_pretrained(version)
        self.processor = CLIPProcessor.from_pretrained(version)
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.device = device
        self.model = self.model.to(self.device)
    
    def __call__(self, dataloader):
        out_score = 0
        for item in dataloader:
            out_score_matrix  = self.model_output(images=item['image'], texts=item['text'])
            out_score += out_score_matrix.mean().item() 
        return out_score / len(dataloader)
    
    def model_output(self, images, texts):
        torch.cuda.empty_cache()
        images_feats = self.processor(images=[Image.open(img) for img in images], return_tensors="pt").to('cuda')
        images_feats = self.model.get_image_features(**images_feats)

        texts_feats = self.tokenizer(texts, padding=True, truncation=True, max_length=77, return_tensors="pt",).to('cuda')
        texts_feats = self.model.get_text_features(**texts_feats)

        images_feats = images_feats / images_feats.norm(dim=1, p=2, keepdim=True)
        texts_feats = texts_feats / texts_feats.norm(dim=1, p=2, keepdim=True)
        score = (images_feats * texts_feats).sum(-1)
        return score


def find_root_paths(root_dir, sub_root):
    return sorted(
        list({os.path.abspath(os.path.join(dirpath, '..')) 
                for dirpath, dirnames, _ in os.walk(root_dir) if sub_root in dirnames})
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--contents', type=str)
    parser.add_argument('--root_path', type=str)
    parser.add_argument('--sub_root', type=str, default='edit')
    parser.add_argument('--pretrained_path', type=str)
    args = parser.parse_args()

    contents = [item.strip() for item in args.contents.split(',')]
    root_paths = find_root_paths(args.root_path, args.sub_root)

    CS_calculator = CLIP_Score()

    for root_path in root_paths:
        try:
            save_txt = os.path.join(root_path, 'record_metrics.txt')
            if not os.path.exists(save_txt): 
                with open(save_txt, 'a') as f:
                    f.writelines('*************************** \n')
                    f.writelines(f'Calculating the metrics for {root_path} \n')

            with open(save_txt, 'r') as f:  
                txt_content = f.read()
            for content in tqdm(contents):
                if content + ':' in txt_content: continue
                dataset = Generate_Dataset(root_path, content, args.sub_root)
                dataloader = DataLoader(dataset, batch_size=10)
                CS = CS_calculator(dataloader)
                FIDELITY = torch_fidelity.calculate_metrics(
                    input1=os.path.join(root_path, content, args.sub_root), 
                    input2=os.path.join(args.pretrained_path, content, 'original') if content != 'coco' else "data/pretrain/coco/coco/original", 
                    cuda=True, 
                    fid=True, 
                    verbose=False,
                )
                with open(save_txt, 'a') as f:
                    f.writelines(f"{content}: CS is {CS * 100}, FID is {FIDELITY['frechet_inception_distance']} \n")
        except Exception as e:
            pass