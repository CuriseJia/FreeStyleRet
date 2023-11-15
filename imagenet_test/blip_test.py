import argparse
from tqdm import tqdm
import sys
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import S2ITestDataset, T2ITestDataset, M2ITestDataset
from comparison_test import Prompt_BLIP
from src.utils import setup_seed, getR1Accuary, getR5Accuary


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Prompt_BLIP test on ImageNet-X Dataset.')

    # project settings
    parser.add_argument('--resume', default='', type=str, help='load model checkpoint from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train test2image or style2image.')
    parser.add_argument("--root_json_path", type=str, default='imagenet/test.json')
    parser.add_argument("--other_json_path", type=str, default='imagenet/test.json')
    parser.add_argument("--root_file_path", type=str, default='imagenet/')
    parser.add_argument("--other_file_path", type=str, default='imagenet-s/')
    parser.add_argument("--batch_size", type=int, default=40)

    # model settings
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args


def S2IRetrieval(args, model, ori_images, pair_images):

    ori_feat = model(ori_images, mode='image')
    ske_feat =  model(pair_images, mode='image')  

    prob = torch.softmax(ske_feat.view(args.batch_size, -1) @ ori_feat.view(args.batch_size, -1).permute(1, 0), dim=-1)

    return prob



def T2IRetrieval(args, model, ori_images, text_caption):

    text_feat = model(text_caption, mode='text')
    img_feat = model(ori_images, mode='image')
    prob = torch.softmax(text_feat.view(args.batch_size, -1) @ img_feat.view(args.batch_size, -1).permute(1, 0), dim=-1)

    return prob


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    
    model = Prompt_BLIP(args)
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    model.to(args.device)

    if args.type == 'text2image':
        test_dataset = T2ITestDataset(args)
    elif args.type == 'style2image':
        test_dataset = S2ITestDataset(args)
    else:
        test_dataset = M2ITestDataset(args)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=16, shuffle=False, drop_last=True)

    r1 = []
    r5 = []
    
    if args.type == 'text2image':
        for data in enumerate(tqdm(test_loader)):
            caption = data[1][0]
            image = data[1][1].to(args.device, non_blocking=True)

            prob = T2IRetrieval(args, model, image, caption)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    else:
        for data in enumerate(tqdm(test_loader)):
            origin_image = data[1][0].to(args.device, non_blocking=True)
            retrival_image = data[1][1].to(args.device, non_blocking=True)

            prob = S2IRetrieval(args, model, origin_image, retrival_image)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    resr1 = sum(r1)/len(r1)
    resr5 = sum(r5)/len(r5)
    print(resr1)
    print(resr5)
