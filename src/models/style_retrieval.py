import open_clip
import numpy as np
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


class ShallowStyleRetrieval(nn.Module):
    def __init__(self, model_args, tgt_device='cpu'):
        super(ShallowStyleRetrieval, self).__init__()
        self.args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=tgt_device)
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.openclip.apply(freeze_all_but_bn)
        self.visual = self.openclip.visual
        self.transformer = self.visual.transformer
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
        # loss
        self.i2t_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        self.t2i_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss_1 = self.i2t_loss(image_feature, pair_feature, negative_feature)
        loss_2 = self.t2i_loss(pair_feature, image_feature, negative_feature)
        loss = (loss_1 + loss_2) / 2
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
    

    def _get_gram_prompt(self, input):
        latent_feature = self._get_features(input, self.gram_encoder)
        embed = self.gram_patch(latent_feature['conv3_1'])
        n, c, h, w = embed.shape    # (b, 256, 7, 7)

        features = embed.view(n, c, -1)  # (b*256, 49)
        features = torch.bmm(features, features.transpose(1, 2))
        features = self.gram_pool(features)
        prompt_feature = self.gram_linear(features.permute(0, 2, 1))

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

        self.gram_prompt.parameter = self._get_gram_prompt(input)

        if self.args.prompt_location == 'Shallow':

            x = torch.cat([x[:, 0, :].unsqueeze(1), self.gram_prompt.expand(x.shape[0],-1,-1), x[:, 1:, :]], dim=1)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.visual.transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
        
        elif self.args.prompt_location == 'Bottom':

            x = x.permute(1, 0, 2)  # NLD -> LND
            for r in range(len(self.transformer.resblocks)):
                if r == len(self.transformer.resblocks)-1:
                    x = torch.cat([x[0, :, :].unsqueeze(0), 
                                self.gram_prompt.expand(self.args.batch_size,-1,-1).permute(1, 0, 2), 
                                x[1:, :, :]], dim=0)
                x = self.transformer.resblocks[r](x)
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


class DeepStyleRetrieval(nn.Module):
    def __init__(self, model_args, tgt_device='cpu'):
        super(DeepStyleRetrieval, self).__init__()
        self.args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(
            model_name='ViT-L-14', pretrained='laion2b_s32b_b82k', device=tgt_device)
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.openclip.apply(freeze_all_but_bn)
        self.visual = self.openclip.visual
        self.transformer = self.visual.transformer
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
        self.style_patch = nn.Sequential(
                            nn.Conv2d(128, 256, 16, 16),
                            nn.Conv2d(256, 256, 7, 7))
        self.style_linear = nn.Sequential(
                                nn.Linear(256, 512),
                                nn.Linear(512, 1024),
                                nn.Linear(1024, self.args.gram_prompt_dim))
        # loss
        self.i2t_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        self.t2i_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss_1 = self.i2t_loss(image_feature, pair_feature, negative_feature)
        loss_2 = self.t2i_loss(pair_feature, image_feature, negative_feature)
        loss = (loss_1 + loss_2) / 2
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
    

    def _get_gram_prompt(self, input):
        latent_feature = self._get_features(input, self.gram_encoder)
        embed = self.gram_patch(latent_feature['conv3_1'])
        n, c, h, w = embed.shape    # (b, 256, 7, 7)

        features = embed.view(n, c, -1)  # (b*256, 49)
        features = torch.bmm(features, features.transpose(1, 2))
        features = self.gram_pool(features)
        prompt_feature = self.gram_linear(features.permute(0, 2, 1))

        return prompt_feature
    

    def _get_style_prompt(self):
        feature = torch.from_numpy(np.load(self.args.style_prompt_path)).view(4, 128, 112, 112).float().to(self.args.device)    # (4, 1605632)
        feature = self.style_patch(feature).view(4, 256)
        feature = self.style_linear(feature)

        return feature


    def _visual_forward(self, x):
        input = x
        self.gram_prompt.parameter = self._get_gram_prompt(input)
        self.style_prompt.parameter = self._get_style_prompt()

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

        # add style_prompt
        x = torch.cat([x[:, 0, :].unsqueeze(1), self.style_prompt.expand(x.shape[0],-1,-1), x[:, 1:, :]], dim=1)

        # add gram_prompt before the last block of transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        for r in range(len(self.transformer.resblocks)):
            if r == len(self.transformer.resblocks)-1:
                x = torch.cat([x[0, :, :].unsqueeze(0), 
                            self.gram_prompt.expand(self.args.batch_size,-1,-1).permute(1, 0, 2), 
                            x[1:, :, :]], dim=0)
            x = self.transformer.resblocks[r](x)
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