import open_clip

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import VGG


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
        self.visual = self.openclip.visual
        # Prompt Token
        self.prompt = nn.Parameter(torch.randn(
            self.model_args.n_prompts, self.model_args.prompt_dim))
        self.style_encoder = VGG
        self.style_encoder.load_state_dict(torch.load(self.model_args.style_encoder_path))
        self.style_encoder.apply(freeze_model)
        self.style_patch = nn.Conv2d(128, 256, 16, 16)
        self.style_pool = nn.Linear(256, 4)
        self.style_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, model_args.prompt_dim))
        # loss
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss = self.triplet_loss(image_feature, pair_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()
        

    def _get_features(self, image, model, layers=None):
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
    

    def _get_prompt(self, input):
        latent_feature = self._get_features(input, self.style_encoder)
        embed = self.style_patch(latent_feature['conv3_1'])
        n, c, h, w = embed.shape    # (b, 256, 7, 7)

        features = embed.view(n, c, -1)  # (b*256, 49)
        features = torch.bmm(features, features.transpose(1, 2))
        features = self.style_pool(features)
        prompt_feature = self.style_linear(features.permute(0, 2, 1))

        return prompt_feature
    

    def _visual_forward(self, x):
        input = x
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat(
            [self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)

        # a patch_dropout of 0. would mean it is disabled and this function would do nothing but return what was passed in
        x = self.visual.patch_dropout(x)
        x = self.visual.ln_pre(x)

        self.prompt.parameter = self._get_prompt(input)
        x = torch.cat([x[:, 0, :].unsqueeze(1), self.prompt.expand(x.shape[0],-1,-1), x[:, 1:, :]], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # if self.visual.attn_pool is not None:
        #     x = self.visual.attn_pool(x)
        #     x = self.visual.ln_post(x)
        #     pooled, tokens = self.visual._global_pool(x)
        # else:
        pooled, tokens = self.visual._global_pool(x)
        pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj

        # if self.visual.output_tokens:
        #     return pooled, tokens
        
        return pooled
        

    def forward(self, data, dtype='image'):
        if dtype == 'image': 
            feat = self._visual_forward(data)

        elif dtype == 'text':
            feat = self.openclip.encode_text(data)

        return feat