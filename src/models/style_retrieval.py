import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

from vgg import VGG


def freeze_model(m):
    m.requires_grad_(False)


def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class StyleRetrieval(nn.Module):
    def __init__(self, model_args, tgt_device='cpu'):
        super(StyleRetrieval, self).__init__()
        self.model_args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=tgt_device,
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.openclip.apply(freeze_all_but_bn)
        # Prompt Token
        self.embedding_layer = nn.Sequential(*list(self.openclip.visual.children())[:3])
        self.transformer_layer = nn.Sequential(*list(self.openclip.visual.children())[4:])
        self.prompt = nn.Parameter(torch.randn(
            self.model_args.n_prompts, self.model_args.prompt_dim))
        self.style_encoder = VGG
        self.style_encoder.load_state_dict(torch.load(self.model_args.style_encoder_path))
        self.style_encoder.apply(freeze_model)
        self.style_patch = nn.Conv2d(128, 256, 8, 8)
        self.style_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, model_args.prompt_dim))
        # loss
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        

    def get_features(image, model, layers=None):
        if layers is None:
            layers = {'0': 'conv1_1',  
                    '5': 'conv2_1',  
                    '10': 'conv3_1', 
                    '19': 'conv4_1', 
                    '21': 'conv4_2', 
                    '28': 'conv5_1',
                    '31': 'conv5_2'
                    }  
        features = {}
        x = image
        for name, layer in model._modules.items():
            x = layer(x)   
            if name in layers:
                features[layers[name]] = x
    
        return features
    

    def get_prompt(self, input):
        latent_feature = self.get_features(input, self.style_encoder)
        embed = self.style_patch(latent_feature['conv3_1'])
        # print(embed.shape)
        prompt_feature = self.style_linear(embed.view(256, -1).permute(1, 0))

        return prompt_feature
    

    def forward(self, data, dtype='image'):
        if dtype == 'image': 
            embed = self.embedding_layer(data)
            embed = embed.flatten(2).transpose(-1, -2)
            self.prompt = self.get_prompt(data)
            embed = torch.cat((
                self.prompt.expand(data.shape[0],-1,-1),
                embed,
            ), dim=1)
            feat = self.transformer_layer(embed)

        elif dtype == 'text':
            feat = self.openclip.encode_text(data)

        return feat