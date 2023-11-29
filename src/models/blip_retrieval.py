import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from open_clip.factory import image_transform
import sys

from BLIP.models.blip_retrieval import blip_retrieval
from models import VGG


image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


def select_style_prompt(input, cluster):
    input = input.view(input.shape[0], -1)
    input_temp = input / torch.norm(input, dim=-1, keepdim=True)
    cluster_temp = cluster / torch.norm(cluster, dim=-1, keepdim=True)
    sim = torch.mm(input_temp, cluster_temp.T)
    sim_prob = F.softmax(sim, dim=1)
    feature = torch.mm(sim_prob, cluster)

    return feature


class BLIP_Retrieval(nn.Module):
    def __init__(self, model_args):
        super(BLIP_Retrieval, self).__init__()
        self.args = model_args
        self.blip = blip_retrieval(pretrained=self.args.origin_resume, image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
        self.blip.apply(freeze_all_but_bn)
        self.visual = self.blip.visual_encoder.blocks
        # Prompt Token
        self.gram_prompt = nn.Parameter(torch.randn(
            self.args.gram_prompts, self.args.gram_prompt_dim))
        self.gram_encoder = VGG
        self.gram_encoder.load_state_dict(torch.load(self.args.gram_encoder_path))
        self.gram_encoder.apply(freeze_model)
        self.gram_patch = nn.Conv2d(128, 256, 16, 16)
        self.gram_pool = nn.Linear(256, 4)
        self.gram_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, self.args.gram_prompt_dim))
        self.style_prompt = nn.Parameter(torch.randn(
            self.args.style_prompts, self.args.style_prompt_dim))
        self.style_patch = nn.Conv2d(256, 256, 16, 16)
        self.style_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, self.args.gram_prompt_dim))
        # loss and process
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        self.pre_process_train = image_transform(224, True, image_mean, image_std)
        self.pre_process_val = image_transform(224, False, image_mean, image_std)
    

    def _get_features(self, image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',  
                    '5': 'conv2_1',  
                    '10': 'conv3_1', 
                    '19': 'conv4_1', 
                    '21': 'conv4_2', 
                    '28': 'conv5_1',
                    '31': 'conv5_2'}  
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)   
            if name in layers:
                features[layers[name]] = x
    
        return features


    def _get_gram_prompt(self, input):
        latent_feature = self._get_features(input, self.gram_encoder)
        embed = self.gram_patch(latent_feature['conv3_1'])
        n, c, h, w = embed.shape    # (b, 256, 7, 7)

        features = embed.view(n, c, -1)  # (b*256, 49)
        features = torch.bmm(features, features.transpose(1, 2))
        features = self.gram_pool(features)
        prompt_feature = self.gram_linear(features.permute(0, 2, 1))

        return prompt_feature
    

    def _get_style_prompt(self, input):
        # style_feature = torch.tensor(torch.randn(4, 4096))
        feature = torch.from_numpy(np.load(self.args.style_cluster_path)).view(4, 4096).float().to(self.args.device)
        
        gram = self._get_features(input, self.gram_encoder)
        embed = self.gram_patch(gram['conv3_1'])
        n, c, h, w = embed.shape
        gram = embed.view(n, c, -1)  # (b*256, 49)
        gram = torch.bmm(gram, gram.transpose(1, 2))

        gram = self.gram_pool(gram)
        gram = self.gram_linear(gram.permute(0, 2, 1))

        feature = select_style_prompt(gram, feature)

        return feature


    def forward(self, data, dtype='image'):
        if dtype == 'image':
            gram_prompt = self._get_gram_prompt(data)
            style_prompt = self._get_style_prompt(data)

            feat = self.blip.visual_encoder.patch_embed(data)
            cls_tokens = self.blip.visual_encoder.cls_token.expand(data.shape[0], -1, -1)
            feat = torch.cat((cls_tokens, feat), dim=1)
            feat = feat + self.blip.visual_encoder.pos_embed[:,:feat.size(1),:]
            feat = self.blip.visual_encoder.pos_drop(feat)

            feat = torch.cat([feat[:, 0, :].unsqueeze(1), style_prompt, feat[:, 1:, :]], dim=1)
            for r in range(len(self.blip.visual_encoder.blocks)):
                if r == len(self.blip.visual_encoder.blocks)-1:
                    feat = torch.cat([feat[:, 0, :].unsqueeze(1), 
                                gram_prompt,
                                feat[:, 1:, :]], dim=1)
                feat = self.blip.visual_encoder.blocks[r](feat)
            
            feat = self.blip.visual_encoder.norm(feat)
            
            ori_embed = F.normalize(self.blip.vision_proj(feat[:,0,:]),dim=-1)    

            return ori_embed
        
        else:
            text = self.blip.tokenizer(data, padding='max_length', truncation=True, max_length=35, 
                              return_tensors="pt").to(self.args.device)
            text_output = self.blip.text_encoder(text.input_ids, attention_mask = text.attention_mask,                      
                                            return_dict = True, mode = 'text')
            text_feat = F.normalize(self.blip.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1)

            return text_feat
    

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss = self.triplet_loss(image_feature, pair_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()