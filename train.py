import argparse
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.models import ShallowStyleRetrieval, DeepStyleRetrieval
from src.dataset import StyleI2IDataset, StyleT2IDataset
from src.utils import setup_seed, save_loss


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Multi-Style-Retrieval.')

    # project settings
    parser.add_argument('--output_dir', default='output/')
    parser.add_argument('--out_path', default='origin-sketch-loss.jpg')
    parser.add_argument('--resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--gram_encoder_path', default='pretrained/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--style_cluster_path', default='pretrained/style_cluster.npy', type=str, help='load style prompt from given npy')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image or style2image.')
    parser.add_argument("--train_dataset_path", type=str, default='fscoco/')
    parser.add_argument("--train_json_path", type=str, default='fscoco/train.json')
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=10)

    # model settings
    parser.add_argument('--prompt_type', type=str, default='ShallowPrompt', help='ShallowPrompt or DeepPrompt')
    parser.add_argument('--prompt_location', type=str, default='Shallow', help='Shallow, Bottom or Middle')
    parser.add_argument('--gram_prompts', type=int, default=4)
    parser.add_argument('--gram_prompt_dim', type=int, default=1024)
    parser.add_argument('--style_prompts', type=int, default=4)
    parser.add_argument('--style_prompt_dim', type=int, default=1024)

    # optimizer settings
    parser.add_argument('--clip_ln_lr', type=float, default=1e-5)
    parser.add_argument('--prompt_lr', type=float, default=1e-5)

    args = parser.parse_args()
    return args



def train(args, model, device, dataloader, optimizer):
    model.train()

    best_loss = 10000000

    losses = []
    epoches = []
    count = 0

    if args.type == 'text2image':
        for epoch in range(args.epochs):
            temp_loss = []

            for data in enumerate(tqdm(dataloader)): 

                caption = model.tokenizer(data[1][0]).to(device, non_blocking=True)
                image = data[1][1].to(device, non_blocking=True)
                negative_image = data[1][2].to(device, non_blocking=True)

                text_feature = model(caption, dtype='text')
                image_feature = model(image, dtype='image')
                negative_feature = model(negative_image, dtype='image')

                loss = model.get_loss(image_feature, text_feature, negative_feature, optimizer)

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
                torch.save(save_obj, os.path.join(args.output_dir, 't2i_epoch{}.pth'.format(epoch)))
                count = 0
            else:
                count +=1
            
            if best_loss < 0.0001 or count >= 5:
                break
    
    else:   # style2image retrival
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
                torch.save(save_obj, os.path.join(args.output_dir, 's2i_epoch{}.pth'.format(epoch)))
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

    if args.prompt == 'ShallowPrompt':
        model = ShallowStyleRetrieval(args)
    else:
        model = DeepStyleRetrieval(args)
    model = model.to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))

    if args.type == 'text2image':
        train_dataset = StyleT2IDataset(args.train_dataset_path,  args.train_json_path, model.pre_process_train)
    else:
        train_dataset = StyleI2IDataset(args.train_dataset_path,  args.train_json_path, model.pre_process_train)

    optimizer = torch.optim.Adam([
            {'params': model.openclip.parameters(), 'lr': args.clip_ln_lr},
            {'params': [model.prompt], 'lr': args.prompt_lr}])

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=args.batch_size,
                            num_workers=args.num_workers,
                            pin_memory=True,
                            prefetch_factor=16,
                            shuffle=False,
                            drop_last=True
                            )

    loss, epochs = train(args, model, device, train_loader, optimizer)

    save_loss(loss, epochs, args.out_path)