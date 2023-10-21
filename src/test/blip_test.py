import os
import numpy as np
import json

import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from models.blip import blip_retrieval, blip_itm
from utils.utils import getR1Accuary, getR5Accuary


def load_image(image, image_size, device, batch=16):
    print('start load image')
    images = []
    for i in range(batch):
        raw_image = Image.open(image[i]).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
        img = transform(raw_image).to(device)
        images.append(img)
    
    imgs = torch.stack(images)

    return imgs


def I2IRetrieval(ori_images, pair_images, ckpt_path, device, batch):
    model = blip_retrieval(pretrained=ckpt_path, image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
    model.eval()
    model.to(device)

    ori_feat = model.visual_encoder(ori_images)
    ori_embed = model.vision_proj(ori_feat)
    ori_embed = F.normalize(ori_embed,dim=-1)

    ske_feat =  model.visual_encoder(pair_images)
    ske_embed = model.vision_proj(ske_feat)  
    ske_embed = F.normalize(ske_embed,dim=-1)    

    prob = torch.softmax(ori_embed.view(b, -1) @ ske_embed.view(batch, -1).permute(1, 0), dim=-1)

    return prob


def T2IRetrieval(ori_images, text_caption, ckpt_path, device):
    model = blip_itm(pretrained=ckpt_path, image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
    model.eval()
    model.to(device)

    prob = model(ori_images, text_caption, match_head='itc')

    return prob


if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ckpt_path = 'model_large_retrieval_coco.pth'
    pair = json.load(open('fscoco/test.json', 'r'))
    
    r1 = []
    r5 = []
    rang = 80
    b = int(len(pair)/rang)
    
    for i in tqdm(range(rang)):
        ori_image=[]
        text_list=[]
        sketch_image=[]
        
        for j in range(b):
            caption_path = os.path.join('fscoco/', 'text/'+pair[i*b+j]['caption'])
            image_path = os.path.join('fscoco/', 'images/'+pair[i*b+j]['image'])
            # sketch_path = os.path.join('fscoco/', 'mosaic/'+pair[i*b+j]['image'])
            
            f = open(caption_path, 'r')
            caption = f.readline().replace('\n', '')
            text_list.append(caption)
            ori_image.append(image_path)
            # sketch_image.append(sketch_path)

        ori_images = load_image(ori_image, 224, device, b)
        # sketch_images = load_image(sketch_image, 224, device, b)

        prob = T2IRetrieval(ori_images, text_list, ckpt_path, device)
        # prob = I2IRetrieval(ori_images, sketch_images, ckpt_path, device, b)

        r1.append(getR1Accuary(prob))
        r5.append(getR5Accuary(prob))


    resr1 = sum(r1)/len(r1)
    resr5 = sum(r5)/len(r5)
    print(resr1)
    print(resr5)




