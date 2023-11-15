import argparse
import os
import json
from tqdm import tqdm
import numpy as np
import torch
import sys
from torch.utils.data import DataLoader
from open_clip.factory import image_transform
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from comparison_test.prompt_model import Prompt_CLIP
from src.models import DeepStyleRetrieval
from src.dataset.data import VisualizationDataset, T2ITestDataset
from src.utils.utils import setup_seed


image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)

convert = {
    'bear' : '#000000',
    'bird' : '#860A35',
    'board' : '#0766AD',
    'boy' : '#83A2FF',
    'build' : '#EC8F5E',
    'cow' : '#ED5AB3',
    'dog' : '#363062',
    'elephant' : '#557C55',
    'giraffe' : '#FF6C22',
    'mountain' : '#9BBEC8',
    'people' : '#706233',
    'plane' : '#F4BF96',
    'sheep' : '#00A9FF',
    'tree' : '#B0A695',
    'truck' : '#C5E898',
    'zebra' : '#8E8FFA'}


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for Visualization.')

    # project settings
    parser.add_argument('--resume', default='pretrained/FreeCLIP.pth', type=str, help='load checkpoints from given path')
    parser.add_argument('--origin_resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--gram_encoder_path', default='pretrained/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--style_cluster_path', default='pretrained/style_cluster.npy', type=str, help='load style cluster from given npy')
    parser.add_argument('--device', default='cuda:1')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=6, type=int)

    # visualization settings
    parser.add_argument('--classname', default='dog', type=str)
    parser.add_argument('--list_json_path', default='DSR/json/', type=str, help='save class json path.')
    parser.add_argument('--out_json_path', default='DSR/json/dog.json', type=str, help='save class json path.')
    parser.add_argument('--out_tensor_path', default='DSR/feature/', type=str, help='save class tensor path.')
    parser.add_argument('--visualization_save_path', default='DSR/FreeCLIP.svg', type=str, help='save class tensor path.')

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image or style2image.')
    parser.add_argument("--style", type=str, default='sketch', help='choose sketch, art or mosaic.')
    parser.add_argument("--test_dataset_path", type=str, default='DSR/')
    parser.add_argument("--test_json_path", type=str, default='DSR/test.json')
    parser.add_argument("--batch_size", type=int, default=1)

    # model settings
    parser.add_argument('--prompt', type=str, default='DeepPrompt', help='Prompt_CLIP or FreeCLIP')
    parser.add_argument('--gram_prompts', type=int, default=4)
    parser.add_argument('--gram_prompt_dim', type=int, default=1024)
    parser.add_argument('--style_prompts', type=int, default=4)
    parser.add_argument('--style_prompt_dim', type=int, default=1024)
    parser.add_argument('--n_prompts', type=int, default=3)
    parser.add_argument('--prompt_dim', type=int, default=50176)

    args = parser.parse_args()
    return args


def select_class(args):
    pre_process_val = image_transform(224, True, image_mean, image_std)
    test_dataset = T2ITestDataset(args.test_dataset_path,  args.test_json_path, pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=4, shuffle=False, drop_last=True)
    
    temp = []
    
    for data in enumerate(tqdm(test_loader)):
        text = data[1][0]
        index = data[1][2]

        if text[0].find(args.classname) != -1:
            pair = {
                'image' : index[0],
                'class' : args.classname,
            }
            print(text[0])
            temp.append(pair)

    with open(args.out_json_path, "w") as file:
        json.dump(temp, file)


def get_tensor(args, model, classname, json_path):
    test_dataset = VisualizationDataset(args.test_dataset_path,  json_path, model.pre_process_val)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                            pin_memory=True, prefetch_factor=4, shuffle=False, drop_last=True)
    
    for data in enumerate(test_loader):

        ori_image = data[1][0].to(args.device, non_blocking=True)
        sketch_image = data[1][1].to(args.device, non_blocking=True)
        art_image = data[1][2].to(args.device, non_blocking=True)
        mosaic_image = data[1][3].to(args.device, non_blocking=True)
        index = data[1][4]

        ori_feature = model(ori_image, dtype='image').squeeze(0).detach().cpu().numpy()
        sketch_feature = model(sketch_image, dtype='image').squeeze(0).detach().cpu().numpy()
        art_feature = model(art_image, dtype='image').squeeze(0).detach().cpu().numpy()
        mosaic_feature = model(mosaic_image, dtype='image').squeeze(0).detach().cpu().numpy()

        np.save('{}/{}/{}_ori.npy'.format(args.out_tensor_path, classname, index[0]), ori_feature)
        np.save('{}/{}/{}_sketch.npy'.format(args.out_tensor_path, classname, index[0]), sketch_feature)
        np.save('{}/{}/{}_art.npy'.format(args.out_tensor_path, classname, index[0]), art_feature)
        np.save('{}/{}/{}_mosaic.npy'.format(args.out_tensor_path, classname, index[0]), mosaic_feature)
        print('successfully save feature of {}'.format(index[0]))


def visulization_result(data):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    return tsne.fit_transform(data)


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    for i in range(data.shape[0]):
        plt.scatter(data[i, 0], data[i, 1], s=1, c=convert[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


if __name__ == "__main__":
    args = parse_args()
    setup_seed(args.seed)

    # select_class(args)

    if args.prompt == 'FreeCLIP':
        model = DeepStyleRetrieval(args)
    else:
        model = Prompt_CLIP(args)
    model = model.to(args.device)
    model.load_state_dict(torch.load(args.resume))
    model.eval()
    print('success load ckpt from {}'.format(args.resume))

    jsonlist = os.listdir(args.list_json_path)
    for json in jsonlist:
        json_path = args.list_json_path + json
        classname = json.split('.')[0]
        get_tensor(args, model, classname, json_path)
    print('finish generate tensor.')

    tensor = []
    label = []
    classlist = os.listdir(args.out_tensor_path)
    for classes in classlist:
        feature_path = args.out_tensor_path + '/{}'.format(classes)
        feature_list = os.listdir(feature_path)
        print('class {} has {} obj'.format(classes, len(feature_list)))
        for feat in feature_list:
            feat_path = feature_path + '/{}'.format(feat)
            x = np.load(feat_path)
            tensor.append(x)
            label.append(classes)
    
    feature = np.vstack(tensor)
    ll = np.array(label)

    np.save('{}/{}_feature.npy'.format(args.out_tensor_path, args.prompt), feature)
    np.save('{}/{}_label.npy'.format(args.out_tensor_path, args.prompt), ll)

    print('feature.shape', feature.shape)
    print('label中包括', len(set(label)), '个不同的类别')
    print('Computing t-SNE embedding')
    result = visulization_result(feature)
    print('result.shape',result.shape)
    fig = plot_embedding(result, label,
                         't-SNE {}\'s result on DSR test dataset'.format(args.prompt))

    fig.savefig(args.visualization_save_path, format='svg')
    print('finish.', result.shape)