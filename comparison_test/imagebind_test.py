import torch
import argparse
import sys
import json
import os
import time
from tqdm import tqdm
from open_clip.factory import image_transform
from torch.utils.data import DataLoader

from src.utils import setup_seed, getR1Accuary, getR5Accuary
from src.dataset import I2ITestDataset, T2ITestDataset
from ImageBind.imagebind import data, ModalityType, imagebind_model
from prompt_model import Prompt_ImageBind


image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Prompt_ImageBind or Origin_ImageBind test on DSR dataset.')

    # project settings
    parser.add_argument('--origin_resume', default='', type=str, help='load origin model checkpoint from given path')
    parser.add_argument('--prompt_resume', default='', type=str, help='load prompt model checkpoint from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image or style2image.')
    parser.add_argument("--style", type=str, default='sketch', help='choose sketch, art or mosaic.')
    parser.add_argument("--test_dataset_path", type=str, default='DSR/')
    parser.add_argument("--test_json_path", type=str, default='DSR/test.json')
    parser.add_argument("--batch_size", type=int, default=24)

    # model settings
    parser.add_argument('--model', type=str, default='prompt', help='prompt-imagebind or imagebind-huge.')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args


def S2IRetrieval(args, model, ori_feat, pair_feat):
    t1 = time.time()

    if args.model == 'prompt':
        ori_feat = model(ori_feat, dtype='image')
        ske_feat = model(pair_feat, mode='image')  

        prob = torch.softmax(ske_feat @ ori_feat.T, dim=-1)
    
    else:
        with torch.no_grad():
            ori_feat = model(ori_feat)
            ske_feat = model(pair_feat)
        
        prob = torch.softmax(ske_feat[ModalityType.VISION] @ ori_feat[ModalityType.VISION].T, dim=-1)
    
    t2 = time.time()
    print('inference a batch costs {}ms'.format((t2-t1)*1000))

    return prob


def T2IRetrieval(args, model, ori_feat, pair_feat):
    t1 = time.time()

    if args.model == 'prompt':
        ori_feat = model(ori_feat, dtype='image')
        ske_feat = model(pair_feat, mode='text')
    
    else:
        with torch.no_grad():
            ori_feat = model(ori_feat)
            ske_feat = model(pair_feat)
        
    prob = torch.softmax(ske_feat[ModalityType.TEXT] @ ori_feat[ModalityType.VISION].T, dim=-1)

    t2 = time.time()
    print('inference a batch costs {}ms'.format((t2-t1)*1000))

    return prob


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    pair = json.load(open(args.test_json_path, 'r'))

    if args.model == 'prompt':
        model = Prompt_ImageBind(args)
        model.load_state_dict(torch.load(args.prompt_resume))
    else:
        model = imagebind_model.imagebind_huge(args.origin_resume)

    model.eval()
    model.to(args.device)
    
    r1 = []
    r5 = []
    rang = int(len(pair)/args.batch_size)

    pre_process_val = image_transform(224, True, image_mean, image_std)

    if args.type == 'text2image':
        test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)
    else:
        test_dataset = I2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)
    
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=16, shuffle=False, drop_last=True)
    
    if args.model == 'prompt':  # prompt_imagebind
        if args.type == 'style2image':
            for data in enumerate(tqdm(test_loader)):
                original_image = data[1][0].to(args.device, non_blocking=True)
                retrival_image = data[1][1].to(args.device, non_blocking=True)

                prob = S2IRetrieval(args, model, original_image, retrival_image)

        else:
            for data in enumerate(tqdm(test_loader)):
                caption = data[1][0]
                image = data[1][1].to(args.device, non_blocking=True)

                prob = T2IRetrieval(args, model, image, caption)

    else:   # imagebind_huge
        for i in tqdm(range(rang)):
            ori_image=[]
            text_list=[]
            sketch_image=[]

            if args.type == 'style2image':
                for j in range(args.batch_size):
                    image_path = os.path.join(args.test_dataset_path, 'images/'+pair[i*args.batch_size+j]['image'])
                    sketch_path = os.path.join(args.test_dataset_path, '{}/'.format(args.style)+pair[i*args.batch_size+j]['image'])
                    ori_image.append(image_path)
                    sketch_image.append(sketch_path)
                
                input1 = {ModalityType.VISION: data.load_and_transform_vision_data(ori_image, args.device)}
                input2 = {ModalityType.VISION: data.load_and_transform_vision_data(sketch_image, args.device)}

                prob = S2IRetrieval(args, model, input1, input2)
            
            else:
                for j in range(args.batch_size):
                    caption_path = os.path.join(args.test_dataset_path, 'text/'+pair[i*args.batch_size+j]['caption'])
                    image_path = os.path.join(args.test_dataset_path, 'images/'+pair[i*args.batch_size+j]['image'])
                    f = open(caption_path, 'r')
                    caption = f.readline().replace('\n', '')
                    text_list.append(caption)
                    ori_image.append(image_path)

                input1 = {ModalityType.VISION: data.load_and_transform_vision_data(ori_image, args.device)}
                input2 = {ModalityType.TEXT: data.load_and_transform_text(text_list, args.device)}

                prob = T2IRetrieval(args, model, input1, input2)

        r1.append(getR1Accuary(prob))
        r5.append(getR5Accuary(prob))

    resr1 = sum(r1)/len(r1)
    resr5 = sum(r5)/len(r5)
    print(resr1)
    print(resr5)