import argparse
from tqdm import tqdm
import time
import torch
import open_clip
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset.data import T2ITestDataset, I2ITestDataset, X2ITestDataset
from src.utils import setup_seed, getR1Accuary, getR5Accuary
from prompt_model import Prompt_CLIP


image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Prompt_CLIP or Origin_CLIP test.')

    # project settings
    parser.add_argument('--origin_resume', default='', type=str, help='load origin model checkpoint from given path')
    parser.add_argument('--prompt_resume', default='', type=str, help='load prompt model checkpoint from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image, style2image or x2image.')
    parser.add_argument("--style", type=str, default='sketch', help='choose sketch, art or mosaic.')
    parser.add_argument("--test_dataset_path", type=str, default='DSR/')
    parser.add_argument("--test_json_path", type=str, default='DSR/test.json')
    parser.add_argument("--batch_size", type=int, default=16)

    # model settings
    parser.add_argument('--model', type=str, default='prompt', help='prompt-clip or origin_clip.')
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args


def eval(args, model, tokenizer, dataloader):
    model.eval()

    r1 = []
    r5 = []

    if args.type == 'text2image':
        for data in enumerate(tqdm(dataloader)):
            caption = tokenizer(data[1][0]).to(device, non_blocking=True)
            image = data[1][1].to(device, non_blocking=True)

            t1 = time.time()

            image_feature = model.encode_image(image)
            text_feature = model.encode_text(caption)

            image_feature = F.normalize(image_feature, dim=-1)
            text_feature = F.normalize(text_feature, dim=-1)

            prob = torch.softmax((100.0 * text_feature @ image_feature.T), dim=-1)

            t2 = time.time()
            print('inference a batch costs {}ms'.format((t2-t1)*1000))

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    elif args.type == 'style2image':
        for data in enumerate(tqdm(dataloader)):
            origin_image = data[1][0].to(device, non_blocking=True)
            retrival_image = data[1][1].to(device, non_blocking=True)

            t1 = time.time()

            original_feature = model.encode_image(origin_image)
            retrival_feature = model.encode_image(retrival_image)

            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob = torch.softmax((100.0 * retrival_feature @ original_feature.T), dim=-1)

            t2 = time.time()
            print('inference a batch costs {}ms'.format((t2-t1)*1000))

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))
    
    else:
        for data in enumerate(tqdm(dataloader)):
            caption = model.tokenizer(data[1][0]).to(device, non_blocking=True)
            origin_image = data[1][1].to(device, non_blocking=True)
            retrival_image = data[1][2].to(device, non_blocking=True)

            text_feature = model(caption, dtype='text')
            original_feature = model(origin_image, dtype='image')
            retrival_feature = model(retrival_image, dtype='image')

            text_feature = F.normalize(text_feature, dim=-1)
            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob1 = torch.softmax((100.0 * text_feature @ original_feature.T), dim=-1)
            prob2 = prob = torch.softmax((100.0 * retrival_feature @ original_feature.T), dim=-1)
            prob = prob1.max(prob2)
            
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
    
    if args.model == 'prompt':
        model = Prompt_CLIP(args)
        model.load_state_dict(torch.load(args.prompt_resume))
        tokenizer = model.tokenizer
        model.to(device)
    else:
        model, _, pre_process_val = open_clip.create_model_and_transforms(model_name='ViT-L-14', pretrained=args.origin_resume, device=device)
        tokenizer = open_clip.get_tokenizer('ViT-L-14')

    if args.type == 'text2image':
        test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)
    elif args.type == 'style2image':
        test_dataset = I2ITestDataset(args.style, args.test_dataset_path,  args.test_json_path, pre_process_val)
    else:
        test_dataset = X2ITestDataset(args.style, args.test_dataset_path,  args.test_json_path, pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=16, shuffle=False, drop_last=True)

    eval(args, model, tokenizer, test_loader)