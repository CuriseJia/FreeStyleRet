import os
import argparse
import torch
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from safetensors import safe_open

import sys
sys.path.append('/home/slimp/AnimateDiff')
from animatediff.utils.convert_from_ckpt import convert_ldm_unet_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description='Parse args for SixModal Prompt Tuning.')

    # path settings
    parser.add_argument('--model_path', default='AnimateDiff/models/StableDiffusion/stable-diffusion-v1-5')
    parser.add_argument('--lora_path', default='AnimateDiff/download_bashscripts/models/DreamBooth_LoRA/toonyou_beta3.safetensors')
    parser.add_argument('--root_path', default='fscoco/')



def Load_SD_with_LoRA(model_path, lora_path):
    state_dict = {}
    pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

    with safe_open(lora_path, framework='pt', device='cpu') as f:
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    convert_unet_checkpoint = convert_ldm_unet_checkpoint(state_dict, pipe.unet.config)
    pipe.unet.load_state_dict(convert_unet_checkpoint, strict=False)
    pipe.to('cuda')
    torch.manual_seed(1)

    return pipe

def gen_art_image(pipe, root_path):
    folderlist = os.listdir(root_path + '/images/')

    if os.path.exists(root_path + '/art/'):
        pass
    else:
        os.mkdir(root_path + '/art/')

    for folder in tqdm(folderlist):
        # os.mkdir(root_path + '/art/{}'.format(folder))
        filelist = os.listdir(root_path + '/images/{}'.format(folder))

        for file in tqdm(filelist):
            text_path = root_path + '/text/{0}/{1}'.format(folder, file.replace('.jpg', '.txt'))
            art_path = root_path + '/art/{0}/{1}'.format(folder, file)

            with open(text_path, 'r') as f:
                prompt = f.readline().replace('\n', '')
            
            image = pipe(prompt).images[0]
            image.save(art_path)


if __name__ == "__main__":
    args = parse_args()
    pipe = Load_SD_with_LoRA(args.model_path, args.lora_path)
    gen_art_image(pipe, args.root_path)