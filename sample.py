import warnings
warnings.filterwarnings("ignore")
import os, sys, pdb
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import copy
import argparse
from PIL import Image
from tqdm import tqdm

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from src.template import template_dict
from src.utils import *


def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, desc=None, **kwargs,):

    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        ) # epsilon(t, x_t, c)
        
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return latents


@torch.no_grad()
def main():

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--seed', type=int, default=0)
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, edit')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=20, help='The total timesteps of the sampling process')
    parser.add_argument('--num_samples', type=int, default=10, help='The number of samples per prompt to generate' )
    parser.add_argument('--batch_size', type=int, default=10, help='The batch size of the sampling process')
    parser.add_argument('--prompts', type=str, default=None)
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--edit_ckpt', type=str, default=None)
    args = parser.parse_args()
    assert args.num_samples >= args.batch_size

    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')

    # region [If certain concept is already sampled, then skip it.]
    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',')]
    if 'edit' in mode_list:
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'edit')
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != len(template_dict[args.erase_type]) * 10:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0: sys.exit()
    # endregion

    # region [Prepare Models]
    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    if 'edit' in mode_list:
        unet_edit = copy.deepcopy(unet)
        edit_path = args.edit_ckpt or os.path.join("logs/checkpoints", sorted(os.listdir("logs/checkpoints"))[-1])
        unet_edit.load_state_dict(torch.load(edit_path, map_location='cpu'), strict=False)
    # endregion

    uncond_embedding = get_textencoding(get_token('', tokenizer), text_encoder)

    # Sampling process
    seed_everything(args.seed, True)
    if args.prompts is None:
        prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    else:
        prompt_list = [[x.format(concept) for x in args.prompts.split(';')] for concept in concept_list]
    for i in range(int(args.num_samples // bs)):
        latent = torch.randn(bs, 4, 64, 64).to(pipe.device, dtype=pipe.dtype)
        for concept, prompts in zip(concept_list, prompt_list):
            for count, prompt in enumerate(prompts):

                save_images = {}
                embedding = get_textencoding(get_token(prompt, tokenizer), text_encoder)

                if 'original' in mode_list:
                    save_images['original'] = diffusion(unet=unet, scheduler=pipe.scheduler, 
                                                   latents=latent, start_timesteps=0, 
                                                   text_embeddings=torch.cat([uncond_embedding] * bs + [embedding] * bs, dim=0), 
                                                   total_timesteps=args.total_timesteps, 
                                                   guidance_scale=args.guidance_scale, 
                                                   desc=f"{count} x {prompt} | original")
                if 'edit' in mode_list:
                    save_images['edit'] = diffusion(unet=unet_edit, scheduler=pipe.scheduler,
                                               latents=latent, start_timesteps=0, 
                                               text_embeddings=torch.cat([uncond_embedding] * bs + [embedding] * bs, dim=0), 
                                               total_timesteps=args.total_timesteps, 
                                               guidance_scale=args.guidance_scale, 
                                               desc=f"{count} x {prompt} | edit")
                                        
                save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)
                for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1: os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                # Decode and process images
                decoded_imgs = {
                    name: [process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]) for img in img_list]
                    for name, img_list in save_images.items()
                }

                # Save images
                def combine_images_horizontally(Images):
                    widths, heights = zip(*(img.size for img in Images))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for i, img in enumerate(Images): new_img.paste(img, (sum(widths[:i]), 0))
                    return new_img
                for idx in range(len(decoded_imgs[mode_list[0]])):
                    save_filename = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_') + f"_{int(idx + bs * i)}.png"
                    images_to_combine = []
                    for mode in mode_list: 
                        decoded_imgs[mode][idx].save(os.path.join(save_path, mode, save_filename))
                        images_to_combine.append(decoded_imgs[mode][idx])
                    if len(mode_list) > 1:
                        img_combined = combine_images_horizontally(images_to_combine)
                        img_combined.save(os.path.join(save_path, 'combine', save_filename.replace('.png', '.jpg')))


if __name__ == '__main__':
    main()