import argparse
import os
import time
from tqdm import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.style_retrieval import StyleRetrieval
from src.dataset.data import StyleI2IDataset, StyleT2IDataset
from src.utils.utils import setup_seed, save_loss, getI2TR1Accuary, getI2IR1Accuary

def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # project settings
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--out_path', default='origin-text-loss.jpg')
    parser.add_argument('--resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--style_encoder_path', default='fscoco/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument("--local_rank", type=int)

    # data settings
    parser.add_argument("--type", type=str, default='image2text', help='choose train image2text or image2image.')
    parser.add_argument("--train_ori_dataset_path", type=str, default='fscoco/')
    parser.add_argument("--train_json_path", type=str, default='fscoco/dataset.json')
    parser.add_argument("--train_batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5)

    # model settings
    parser.add_argument('--prompt', type=str, default='ShallowPrompt', help='ShallowPrompt or DeepPrompt')
    parser.add_argument('--n_prompts', type=int, default=4)
    parser.add_argument('--prompt_dim', type=int, default=1024)

    # optimizer settings
    parser.add_argument('--clip_ln_lr', type=float, default=1e-4)
    parser.add_argument('--prompt_lr', type=float, default=1e-4)

    args = parser.parse_args()
    return args



def train(args, model, device, dataloader, optimizer):
    model.train()

    best_loss = 10000000

    losses = []
    epoches = []
    count = 0

    if args.type == 'image2text':
        for epoch in range(args.epochs):
            temp_loss = []

            for data in enumerate(tqdm(dataloader)): 

                caption = model.tokenizer(data[1][0]).to(device, non_blocking=True)
                image = data[1][1].to(device, non_blocking=True)
                negative_image = data[1][2].to(device, non_blocking=True)

                text_feature = model(caption, dtype='text')
                image_feature = model(image, dtype='image')
                negative_feature = model(negative_image, dtype='image')

                loss = model.get_loss(original_feature, retrival_feature, negative_feature, optimizer)

                temp_loss.append(loss)

                print("loss: {:.6f}".format(loss))

            if len(temp_loss)!=0:
                res = round(sum(temp_loss)/len(temp_loss), 6)
                print("epoch_{} loss is {}.".format(epoch, res))
            losses.append(res)
            epoches.append(epoch)

            if res<best_loss:
                best_loss = res
                save_obj = model.state_dict()
                torch.save(save_obj, os.path.join(args.output_dir, 'i2t_epoch{}.pth'.format(epoch)))
                count = 0
            else:
                count +=1
            
            if best_loss < 0.0001 or count >= 5:
                break
    
    else:   # image2image retrival
        for epoch in range(args.epochs):
            temp_loss = []
            
            for data in enumerate(tqdm(dataloader)):

                original_image = data[1][0].to(device, non_blocking=True)
                retrival_image = data[1][1].to(device, non_blocking=True)
                negative_image = data[1][2].to(device, non_blocking=True)

                original_feature = model(original_image, dtype='image')
                retrival_feature = model(retrival_image, dtype='image')
                negative_feature = model(negative_image, dtype='image')

                loss = model.get_loss(original_feature, retrival_feature, negative_feature, optimizer)

                temp_loss.append(loss)

                print("loss: {:.6f}".format(loss))
            
            if len(temp_loss)!=0:
                res = round(sum(temp_loss)/len(temp_loss), 6)
                print("epoch_{} loss is {}.".format(epoch, res))
            losses.append(res)
            epoches.append(epoch)

            if res<best_loss:
                best_loss = res
                save_obj = model.state_dict()
                torch.save(save_obj, os.path.join(args.output_dir, 'i2i_epoch{}.pth'.format(epoch)))
                count = 0
            else:
                count +=1
            
            if best_loss < 0.0001 or count >= 5:
                break
    
    return losses, epoches


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    model = StyleRetrieval(args)
    model = model.to(device)
    # model.load_state_dict(torch.load(args.resume))

    train_dataset = StyleT2IDataset(args.train_ori_dataset_path,  args.train_json_path, model.pre_process_train)
    train_dataset = StyleI2IDataset(args.train_ori_dataset_path,  args.train_json_path, model.pre_process_train)

    optimizer = torch.optim.Adam([
            {'params': model.openclip.parameters(), 'lr': args.clip_ln_lr},
            {'params': [model.prompt], 'lr': args.prompt_lr}])

    train_loader = DataLoader(dataset=train_dataset, 
                            batch_size=args.train_batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            prefetch_factor=16,
                            shuffle=False,
                            drop_last=True
                            )

    loss, epochs = train(args, model, device, train_loader, optimizer)

    save_loss(loss, epochs, args.out_path)