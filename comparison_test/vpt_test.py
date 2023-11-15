import argparse
from tqdm import tqdm
import torch
import sys
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader

from prompt_model import VPT_Deep
from src.dataset.data import T2ITestDataset, I2ITestDataset
from src.utils import setup_seed, getR1Accuary, getR5Accuary


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for VPT-Deep test on DSR dataset.')

    # project settings
    parser.add_argument("--resume", type=str, default='', help='load checkpoints from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image, style2image.')
    parser.add_argument("--style", type=str, default='sketch', help='choose sketch, art or mosaic.')
    parser.add_argument("--test_dataset_path", type=str, default='DSR/')
    parser.add_argument("--test_json_path", type=str, default='DSR/test.json')
    parser.add_argument("--batch_size", type=int, default=32)

    # model settings
    parser.add_argument('--n_prompts', type=int, default=4)
    parser.add_argument('--prompt_dim', type=int, default=1024)

    args = parser.parse_args()
    return args


def eval(args, model, dataloader):
    model.eval()

    r1 = []
    r5 = []

    if args.type == 'text2image':
        for data in enumerate(tqdm(dataloader)):
            caption = data[1][0]
            image = data[1][1].to(args.device, non_blocking=True)

            t1 = time.time()

            image_feature = model(image, mode='image')
            text_feature = model(caption, mode='text')

            image_feature = F.normalize(image_feature, dim=-1)
            text_feature = F.normalize(text_feature, dim=-1)

            prob = torch.softmax((100.0 * text_feature @ image_feature.T), dim=-1)

            t2 = time.time()
            print('inference a batch costs {}ms'.format((t2-t1)*1000))

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    else:
        for data in enumerate(tqdm(dataloader)):
            origin_image = data[1][0].to(args.device, non_blocking=True)
            retrival_image = data[1][1].to(args.device, non_blocking=True)

            t1 = time.time()

            original_feature = model(origin_image, mode='image')
            retrival_feature = model(retrival_image, mode='image')

            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob = torch.softmax((100.0 * retrival_feature @ original_feature.T), dim=-1)

            t2 = time.time()
            print('inference a batch costs {}ms'.format((t2-t1)*1000))

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    resr1 = sum(r1)/len(r1)
    resr5 = sum(r5)/len(r5)
    print(resr1)
    print(resr5)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    
    model = VPT_Deep(args)
    model.to(args.device)
    model.load_state_dict(torch.load(args.resume))
    pre_process_val = model.pre_process_val

    if args.type == 'text2image':
        test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)
    else:
        test_dataset = I2ITestDataset(args.style, args.test_dataset_path,  args.test_json_path, pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=16, shuffle=False, drop_last=True)

    eval(args, model, test_loader)