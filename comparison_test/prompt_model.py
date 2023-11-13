import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
from open_clip.factory import image_transform
import sys

from ImageBind.imagebind import imagebind_model, data, ModalityType
from BLIP.models.blip_retrieval import blip_retrieval


image_mean = (0.48145466, 0.4578275, 0.40821073)
image_std = (0.26861954, 0.26130258, 0.27577711)

def freeze_all_but_bn(m):
    if not isinstance(m, torch.nn.LayerNorm):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.requires_grad_(False)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.requires_grad_(False)


class Prompt_ImageBind(nn.Module):
    def __init__(self, model_args):
        super(Prompt_ImageBind, self).__init__()
        self.args = model_args
        self.imagebind = imagebind_model.imagebind_huge(pretrained=self.args.origin_resume)
        self.imagebind.apply(freeze_all_but_bn)
        self.prompt = nn.Parameter(torch.randn(
            self.args.n_prompts, self.args.prompt_dim))
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        self.pre_process_train = image_transform(224, True, image_mean, image_std)
        self.pre_process_val = image_transform(224, False, image_mean, image_std)

    
    def forward(self, x, dtype='image'):
        if dtype == 'image':
            x = x + self.prompt.expand(x.shape[0], -1, -1).view(
                    x.shape[0],x.shape[1],x.shape[2],x.shape[3])
            input = {ModalityType.VISION: x}
            feat = self.imagebind(input)

            return feat[ModalityType.VISION]

        else:
            input = {ModalityType.TEXT: data.load_and_transform_text(x, self.args.device)}
            feat = self.imagebind(input)

            return feat[ModalityType.TEXT]
        
    
    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss = self.triplet_loss(image_feature, pair_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()


class Prompt_BLIP(nn.Module):
    def __init__(self, model_args):
        super(Prompt_BLIP, self).__init__()
        self.args = model_args
        self.blip = blip_retrieval(pretrained=self.args.origin_resume, image_size=224, vit='large', vit_grad_ckpt=True, vit_ckpt_layer=10)
        self.blip.apply(freeze_all_but_bn)
        self.prompt = nn.Parameter(torch.randn(
            self.args.n_prompts, self.args.prompt_dim))
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
        self.pre_process_train = image_transform(224, True, image_mean, image_std)
        self.pre_process_val = image_transform(224, False, image_mean, image_std)
    

    def forward(self, data, dtype='image'):
        if dtype == 'image':
            image = data + self.prompt.expand(data.shape[0], -1, -1).view(
                    data.shape[0],data.shape[1],data.shape[2],data.shape[3])
            ori_feat = self.blip.visual_encoder(image)
            ori_embed = F.normalize(self.blip.vision_proj(ori_feat[:,0,:]),dim=-1)    

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
    

class Prompt_CLIP(nn.Module):
    def __init__(self, model_args):
        super(Prompt_CLIP, self).__init__()
        self.args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(model_name='ViT-L-14')
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.openclip.apply(freeze_all_but_bn)
        # Prompt Token
        self.prompt = nn.Parameter(torch.randn(
            self.args.n_prompts, self.args.prompt_dim))
        # loss
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
    

    def forward(self, data, dtype='image'):
        if dtype == 'image': 
            feat = self.openclip.encode_image(
                data + self.prompt.expand(data.shape[0], -1, -1).view(
                    data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
            
        else:
            text = self.tokenizer(data).to(self.args.device)
            feat = self.openclip.encode_text(text)
        return feat
    

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss = self.triplet_loss(image_feature, pair_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()
    

class VPT_Deep(nn.Module):
    def __init__(self, model_args):
        super(VPT_Deep, self).__init__()
        self.args = model_args
        self.openclip, self.pre_process_train, self.pre_process_val = open_clip.create_model_and_transforms(model_name='ViT-L-14')
        self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
        self.openclip.apply(freeze_all_but_bn)
        self.visual = self.openclip.visual
        self.transformer = self.visual.transformer
        # Prompt Token
        self.prompt = nn.Parameter(torch.randn(
            self.args.n_prompts, self.args.prompt_dim))
        # loss
        self.triplet_loss = nn.TripletMarginWithDistanceLoss(
            distance_function=lambda x, y: 1.0-F.cosine_similarity(x, y), 
            margin=1)
    

    def _visual_forward(self, x):

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

        x = x.permute(1, 0, 2)  # NLD -> LND
        for r in range(len(self.transformer.resblocks)):
            if r == 0:
                x = torch.cat([x[0, :, :].unsqueeze(0), 
                            self.prompt.expand(self.args.batch_size, -1, -1).permute(1, 0, 2), 
                            x[1:, :, :]], dim=0)
            else:
                x[1:5, :, :] = self.prompt.expand(self.args.batch_size, -1, -1).permute(1, 0, 2)
            x = self.transformer.resblocks[r](x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        pooled, tokens = self.visual._global_pool(x)
        pooled = self.visual.ln_post(pooled)

        if self.visual.proj is not None:
            pooled = pooled @ self.visual.proj
        
        return pooled


    def forward(self, data, dtype='image'):
        if dtype == 'image': 
            feat = self._visual_forward(data)
            
        else:
            text = self.tokenizer(data).to(self.args.device)
            feat = self.openclip.encode_text(text)
        return feat
    

    def get_loss(self, image_feature, pair_feature, negative_feature, optimizer):
        loss = self.triplet_loss(image_feature, pair_feature, negative_feature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.detach().cpu().numpy()