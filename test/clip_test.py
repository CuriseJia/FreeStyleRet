import argparse
from tqdm import tqdm
import torch
import open_clip
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.dataset.data import T2ITestDataset, I2ITestDataset
from src.utils.utils import setup_seed, getR1Accuary, getR5Accuary, getR10Accuary

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # project settings
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train test2image or style2image.')
    parser.add_argument("--test_dataset_path", type=str, default='fscoco/')
    parser.add_argument("--test_json_path", type=str, default='fscoco/test.json')
    parser.add_argument("--batch_size", type=int, default=16)

    args = parser.parse_args()
    return args


def eval(args, model, tokenizer, dataloader):
    model.eval()

    r1 = []
    r5 = []

    if args.type == 'image2text':
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

    model, _, pre_process_val = open_clip.create_model_and_transforms(model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=device)
    tokenizer = open_clip.get_tokenizer('ViT-L-14')

    # test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)
    test_dataset = I2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, 
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            prefetch_factor=16,
                            shuffle=False,
                            drop_last=True
                            )

    eval(args, model, tokenizer, test_loader)