import argparse
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.style_retrieval import StyleRetrieval
from src.dataset.data import T2ITestDataset, I2ITestDataset
from src.utils.utils import setup_seed, getR1Accuary, getR5Accuary, getR10Accuary

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # project settings
    parser.add_argument('--resume', default='output/i2i_epoch8.pth', type=str, help='load checkpoints from given path')
    parser.add_argument('--style_encoder_path', default='fscoco/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument("--local_rank", type=int)

    # data settings
    parser.add_argument("--type", type=str, default='image2image', help='choose train image2text or image2image.')
    parser.add_argument("--test_dataset_path", type=str, default='fscoco/')
    parser.add_argument("--test_json_path", type=str, default='fscoco/test.json')
    parser.add_argument("--batch_size", type=int, default=24)

    # model settings
    parser.add_argument('--prompt', type=str, default='ShallowPrompt', help='ShallowPrompt or DeepPrompt')
    parser.add_argument('--n_prompts', type=int, default=4)
    parser.add_argument('--prompt_dim', type=int, default=1024)

    args = parser.parse_args()
    return args


def eval(args, model, dataloader):
    model.eval()

    acc = []

    if args.type == 'image2text':
        for data in enumerate(tqdm(dataloader)):
            caption = model.tokenizer(data[1][0]).to(device, non_blocking=True)
            image = data[1][1].to(device, non_blocking=True)

            image_feature = model(image, dtype='image')
            text_feature = model(caption, dtype='text')

            image_feature = F.normalize(image_feature, dim=-1)
            text_feature = F.normalize(text_feature, dim=-1)

            prob = torch.softmax((100.0 * image_feature @ text_feature.T), dim=-1)

            acc.append(getR1Accuary(prob))

    else:
        for data in enumerate(tqdm(dataloader)):
            origin_image = data[1][0].to(device, non_blocking=True)
            retrival_image = data[1][1].to(device, non_blocking=True)

            original_feature = model(origin_image, dtype='image')
            retrival_feature = model(retrival_image, dtype='image')

            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob = torch.softmax((100.0 * original_feature @ retrival_feature.T), dim=-1)

            acc.append(getR1Accuary(prob))

    res = sum(acc)/len(acc)
    print(res)


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    model = StyleRetrieval(args)
    model = model.to(device)
    model.load_state_dict(torch.load(args.resume))

    # test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, model.pre_process_val)
    test_dataset = I2ITestDataset(args.test_dataset_path,  args.test_json_path, model.pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            prefetch_factor=16,
                            shuffle=False,
                            drop_last=True
                            )

    eval(args, model, test_loader)