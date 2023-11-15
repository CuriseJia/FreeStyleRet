import argparse
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models import ShallowStyleRetrieval, DeepStyleRetrieval, BLIP_Retrieval
from src.dataset.data import T2ITestDataset, I2ITestDataset, X2ITestDataset
from src.utils.utils import setup_seed, getR1Accuary, getR5Accuary


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for FreeStyleRet Training.')

    # project settings
    parser.add_argument('--resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--origin_resume', default='model_large_retrieval_coco.pth', type=str, help='load checkpoints from given path')
    parser.add_argument('--gram_encoder_path', default='pretrained/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--style_cluster_path', default='pretrained/style_cluster.npy', type=str, help='load style prompt from given npy')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image or style2image.')
    parser.add_argument("--style", type=str, default='sketch', help='choose sketch, art or mosaic.')
    parser.add_argument("--test_dataset_path", type=str, default='DSR/')
    parser.add_argument("--test_json_path", type=str, default='DSR/test.json')
    parser.add_argument("--batch_size", type=int, default=24)

    # model settings
    parser.add_argument('--prompt', type=str, default='DeepPrompt', help='ShallowPrompt or DeepPrompt')
    parser.add_argument('--gram_prompts', type=int, default=4)
    parser.add_argument('--gram_prompt_dim', type=int, default=1024)
    parser.add_argument('--style_prompts', type=int, default=4)
    parser.add_argument('--style_prompt_dim', type=int, default=1024)

    args = parser.parse_args()
    return args


def eval(args, model, dataloader):
    model.eval()

    r1 = []
    r5 = []

    if args.type == 'text2image':
        for data in enumerate(tqdm(dataloader)):
            if args.prompt == 'BLIP_Retrieval':
                caption = data[1][0]
            else:
                caption = model.tokenizer(data[1][0]).to(args.device, non_blocking=True)
            image = data[1][1].to(args.device, non_blocking=True)

            image_feature = model(image, dtype='image')
            text_feature = model(caption, dtype='text')

            image_feature = F.normalize(image_feature, dim=-1)
            text_feature = F.normalize(text_feature, dim=-1)

            prob = torch.softmax((100.0 * text_feature @ image_feature.T), dim=-1)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))

    elif args.type == 'style2image':
        for data in enumerate(tqdm(dataloader)):
            origin_image = data[1][0].to(args.device, non_blocking=True)
            retrival_image = data[1][1].to(args.device, non_blocking=True)

            original_feature = model(origin_image, dtype='image')
            retrival_feature = model(retrival_image, dtype='image')

            original_feature = F.normalize(original_feature, dim=-1)
            retrival_feature = F.normalize(retrival_feature, dim=-1)

            prob = torch.softmax((100.0 * retrival_feature @ original_feature.T), dim=-1)

            r1.append(getR1Accuary(prob))
            r5.append(getR5Accuary(prob))
    
    else:
        for data in enumerate(tqdm(dataloader)):
            if args.prompt == 'BLIP_Retrieval':
                caption = data[1][0]
            else:
                caption = model.tokenizer(data[1][0]).to(args.device, non_blocking=True)
            origin_image = data[1][1].to(args.device, non_blocking=True)
            retrival_image = data[1][2].to(args.device, non_blocking=True)

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
    print('R@1 Acc is {}'.format(resr1))
    print('R@5 Acc is {}'.format(resr5))


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)

    if args.prompt == 'ShallowPrompt':
        model = ShallowStyleRetrieval(args)
    elif args.prompt == 'DeepPrompt':
        model = DeepStyleRetrieval(args)
    else:
        model = BLIP_Retrieval(args)
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.resume))
    
    if args.type == 'text2image':
        test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, model.pre_process_val)
    elif args.type == 'style2image':
        test_dataset = I2ITestDataset(args.style, args.test_dataset_path,  args.test_json_path, model.pre_process_val)
    else:
        test_dataset = X2ITestDataset(args.style, args.test_dataset_path,  args.test_json_path, model.pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=16, shuffle=False, drop_last=True)

    eval(args, model, test_loader)