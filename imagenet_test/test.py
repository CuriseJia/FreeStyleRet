import argparse
from tqdm import tqdm
import sys
import torch
import open_clip
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import S2ITestDataset, T2ITestDataset, M2ITestDataset
from comparison_test import Prompt_BLIP, Prompt_CLIP, Prompt_ImageBind
from src.utils.utils import setup_seed, getR1Accuary, getR5Accuary


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Prompt_Model test on ImageNet-X Dataset.')

    # project settings
    parser.add_argument('--resume', default='', type=str, help='load model checkpoint from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train test2image or style2image.')
    parser.add_argument("--root_json_path", type=str, default='imagenet/test.json')
    parser.add_argument("--root_file_path", type=str, default='imagenet/')
    parser.add_argument("--other_file_path", type=str, default='imagenet-s/')
    parser.add_argument("--batch_size", type=int, default=16)

    # model settings
    parser.add_argument('--model', type=str, default='clip', help='Prompt_CLIP, Prompt_BLIP or Prompt_ImageBind.')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args


def eval(args, model, dataloader, **kwargs):

    r1 = []
    r5 = []

    if args.type == 'text2image':
        for data in enumerate(tqdm(dataloader)):

            caption = tokenizer(data[1][0]).to(device, non_blocking=True)
            image = data[1][1].to(device, non_blocking=True)

            image_feature = model.encode_image(image)
            text_feature = model.encode_text(caption)

            image_feature = F.normalize(image_feature, dim=-1)
            text_feature = F.normalize(text_feature, dim=-1)

            prob = torch.softmax((100.0 * text_feature @ image_feature.T), dim=-1)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    else:
        for data in enumerate(tqdm(dataloader)):
            
            origin_image = data[1][0].to(device, non_blocking=True)
            retrival_image = data[1][1].to(device, non_blocking=True)

            original_feature = model.encode_image(origin_image)
            retrival_feature = model.encode_image(retrival_image)

            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob = torch.softmax((100.0 * retrival_feature @ original_feature.T), dim=-1)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    resr1 = sum(r1)/len(r1)
    resr5 = sum(r5)/len(r5)
    print(resr1)
    print(resr5)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    
    if args.model == 'clip':
        model = Prompt_CLIP(args)
        tokenizer = model.tokenizer
    elif args.model == 'blip':
        model = Prompt_BLIP(args)
    else:
        model = Prompt_ImageBind(args)

    model.load_state_dict(torch.load(args.prompt_resume))
    model.eval()
    model.to(args.device)

    if args.type == 'text2image':
        test_dataset = T2ITestDataset(args)
    elif args.type == 'style2image':
        test_dataset = S2ITestDataset(args)
    else:
        test_dataset = M2ITestDataset(args)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            prefetch_factor=16,
                            shuffle=False,
                            drop_last=True
                            )

    eval(args, model, test_loader, tokenizer)