import os
import json
import sys
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm

from prompt_model import Prompt_BLIP
from BLIP.models import blip_retrieval
from src.utils import getR1Accuary, getR5Accuary


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Prompt_BLIP test.')

    # project settings
    parser.add_argument('--resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--device', default='cuda:0')

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image or style2image.')
    parser.add_argument("--test_dataset_path", type=str, default='fscoco/')
    parser.add_argument("--test_json_path", type=str, default='fscoco/test.json')
    parser.add_argument("--batch_size", type=int, default=24)

    # model settings
    parser.add_argument('--model', type=str, default='prompt', help='prompt-blip or blip-retrieval.')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args


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


def S2IRetrieval(args, model, ori_images, pair_images):
    if args.model == 'prompt':
        ori_feat = model(ori_images, mode='image')
        ske_feat =  model(pair_images, mode='image')  

        prob = torch.softmax(ske_feat.view(args.batch_size, -1) @ ori_feat.view(args.batch_size, -1).permute(1, 0), dim=-1)
    
    else:
        ori_feat = model.visual_encoder(ori_images)
        ori_embed = model.vision_proj(ori_feat)
        ori_embed = F.normalize(ori_embed,dim=-1)

        ske_feat =  model.visual_encoder(pair_images)
        ske_embed = model.vision_proj(ske_feat)  
        ske_embed = F.normalize(ske_embed,dim=-1)    

        prob = torch.softmax(ske_embed.view(args.batch_size, -1) @ ori_embed.view(args.batch_size, -1).permute(1, 0), dim=-1)

    return prob


def T2IRetrieval(args, model, ori_images, text_caption):

    prob = model(ori_images, text_caption, match_head='itc')

    return prob


if __name__ == "__main__":
    args = parse_args()
    pair = json.load(open(args.test_json_path, 'r'))
    
    if args.model == 'prompt':
        model = Prompt_BLIP(args)
    else:
        model = blip_retrieval(pretrained=args.resume, image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)

    model.eval()
    model.to(args.device)

    r1 = []
    r5 = []
    rang = int(len(pair)/args.batch_size)
    
    for i in tqdm(range(rang)):
        ori_image=[]
        text_list=[]
        sketch_image=[]
        
        if args.type == 'style2image':
            for j in range(args.batch_size):
                image_path = os.path.join(args.test_dataset_path, 'images/'+pair[i*args.batch_size+j]['image'])
                sketch_path = os.path.join(args.test_dataset_path, 'sketch/'+pair[i*args.batch_size+j]['image'])

                ori_image.append(image_path)
                sketch_image.append(sketch_path)
            
            ori_images = load_image(ori_image, 224, args.device, args.batch_size)
            sketch_images = load_image(sketch_image, 224, args.device, args.batch_size)

            prob = S2IRetrieval(args, model, ori_images, sketch_images)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

        else:
            for j in range(args.batch_size):
                caption_path = os.path.join(args.test_dataset_path, 'text/'+pair[i*args.batch_size+j]['caption'])
                image_path = os.path.join(args.test_dataset_path, 'images/'+pair[i*args.batch_size+j]['image'])

                f = open(caption_path, 'r')
                caption = f.readline().replace('\n', '')
                text_list.append(caption)
                ori_image.append(image_path)

            ori_images = load_image(ori_image, 224, args.device, args.batch_size)

            prob = T2IRetrieval(args, model, ori_images, text_list)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    resr1 = sum(r1)/len(r1)
    resr5 = sum(r5)/len(r5)
    print(resr1)
    print(resr5)
