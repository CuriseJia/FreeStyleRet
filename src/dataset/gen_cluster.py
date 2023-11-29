import argparse
import torch
import torch.nn as nn
import json
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from open_clip.factory import image_transform

from src.models import DeepStyleRetrieval


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for generate cluster.')

    # project settings
    parser.add_argument('--resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--origin_resume', default='', type=str, help='load checkpoints from given path')
    parser.add_argument('--gram_encoder_path', default='pretrained/vgg_normalised.pth', type=str, help='load vgg from given path')
    parser.add_argument('--style_cluster_path', default='pretrained/style_cluster.npy', type=str, help='load vgg from given path')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_workers', default=6, type=int)

    # data settings
    parser.add_argument("--type", type=str, default='style2image', help='choose train text2image or style2image.')
    parser.add_argument("--style", type=str, default='sketch', help='choose sketch, art or mosaic.')
    parser.add_argument("--style2", type=str, default='art', help='choose sketch, art or mosaic.')
    parser.add_argument("--test_dataset_path", type=str, default='DSR/')
    parser.add_argument("--test_json_path", type=str, default='DSR/test.json')
    parser.add_argument("--batch_size", type=int, default=16)

    # model settings
    parser.add_argument('--prompt', type=str, default='DeepPrompt', help='ShallowPrompt or DeepPrompt')
    parser.add_argument('--location', type=str, default='top', help='top or bottom')
    parser.add_argument('--init', type=str, default='random', help='random, gram, or style')
    parser.add_argument('--gram_prompts', type=int, default=4)
    parser.add_argument('--gram_prompt_dim', type=int, default=1024)
    parser.add_argument('--style_prompts', type=int, default=4)
    parser.add_argument('--style_prompt_dim', type=int, default=1024)

    args = parser.parse_args()
    return args


root_path = ''

image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)

args = parse_args()

filelist = json.load(open('train.json', 'r'))

process = image_transform(224, False, image_mean, image_std)

embed = []

model = DeepStyleRetrieval(args)
model.load_state_dict(torch.load(args.resume))
model.to('cuda:0')

for file in tqdm(filelist):
    path = root_path + 'sketch/' + file['image']
    img = process(Image.open(path)).unsqueeze(0).to('cuda:0')
    feat = model._get_gram_prompt(img)  # (1, 4, 1024)
    embed.append(feat.detach().cpu())

feature = torch.stack(embed, dim=0).squeeze(1).detach().numpy()

np.save('gram_cluster/sketch.npy', feature)

kmeans = KMeans(n_clusters=1)

kmeans.fit(feature)

cluster_labels = kmeans.labels_

cluster_centers = kmeans.cluster_centers_

cluster_tensor = np.mean(cluster_centers, axis=0)
cluster_tensor = cluster_tensor.reshape((1, 4096))

np.save('sketch_cluster.npy', cluster_tensor)