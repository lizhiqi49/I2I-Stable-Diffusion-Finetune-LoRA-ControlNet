import os
import inspect
import logging
import math
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import imageio.v3 as imageio
from tqdm.auto import tqdm
from einops import rearrange
from torchvision.utils import make_grid
from typing import Optional, Literal, Union
from omegaconf import OmegaConf

from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

import wandb
import diffusers
import transformers
from diffusers import (
    ModelMixin,
    AutoencoderKL,
    ControlNetModel,
    DDIMScheduler,
    UNet2DConditionModel,
    DiffusionPipeline,
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline
)
from diffusers.models.attention_processor import (
    LoRAAttnProcessor, 
    LoRAXFormersAttnProcessor,
    XFormersAttnProcessor, 
    AttnProcessor
)

from diffusers.utils import is_xformers_available
from diffusers.loaders import AttnProcsLayers
from diffusers.optimization import get_scheduler

from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger

from dataset import LoRADataset
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector


logger = get_logger(__name__, log_level="INFO")

apply_canny = CannyDetector()
default_neg_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


def parse_args():
    parser = argparse.ArgumentParser(desciption="Sampling")
    parser.add_argument("--pretrained_unet_lora_path", type=str, default=None,
                        help="The path of pretrained LoRA's state dict.")
    parser.add_argument("--pretrained_controlnet_path", type=str, default=None,
                        help="The path or version of pretrained ControlNet.")
    parser.add_argument("--pretrained_model_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="The path or version of pretrained Stable Diffusion.")
    parser.add_argument("--hint_image", type=str, default=None,
                        help="The path of source image. If None, the script will perform random generation.")
    parser.add_argument("--prompt", type=str, default="a painting",
                        help="The textual prompt for sampling.")
    parser.add_argument("--num_images_per_promptp", type=int, default=1,
                        help="The number of samples.")
    parser.add_argument("--num_inference_steps", type=int, default=30,
                        help="The inference steps for diffusion model.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed.")
    parser.add_argument("--fp16", action="store_true", default=False,
                        help="Whether use fp16 in inference.")
    parser.add_argument("--guidance_scale", type=float, default=9.0,
                        help="The guidance scale for classifier-free-guidance.")
    parser.add_argument("--negative_prompt", type=str, default=default_neg_prompt,
                        help="The negative prompt for classifier-free-guidance.")
    parser.add_argument("--output_path", type=str, default='./sample.png',
                        help="The output jpg/png path to save sampling results.")
    args = parser.parse_args()
    
    return args




def init_lora_attn(model, lora_rank=4):
    lora_attn_procs = {}
    for name in model.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else model.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = model.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(model.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = model.config.block_out_channels[block_id]
        procs = LoRAAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=lora_rank)
        lora_attn_procs[name] = procs
    return lora_attn_procs



def main(
    pretrained_unet_lora_path: str = None,
    pretrained_controlnet_path: Optional[str] = "lllyasviel/sd-controlnet-canny",
    pretrained_model_path: str = "runwayml/stable-diffusion-v1-5",
    hint_image: np.ndarray = None,
    prompt: Union[str, list[str]] = None,
    num_images_per_prompt: int = 1,
    num_inference_steps: int = 30,
    canny_low_threshold: int = 100,
    canny_high_threshold: int = 200,
    image_reso: int = 512,
    seed: int = 0,
    fp16: bool = True,
    enable_xformers_memory_efficient_attention: bool = True,
    guidance_scale: float = 9.0,
    negative_prompt: str = default_neg_prompt,
    output_path: str = './figs/sample.png'
):
    set_seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load scheduler, tokenizer and models
    # noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    # text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    # Initialize UNet and maybe load lora attention
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    if pretrained_unet_lora_path is not None:
        lora_attn_procs = init_lora_attn(unet)
        unet.set_attn_processor(lora_attn_procs)
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    if pretrained_unet_lora_path is not None:        
        lora_layers = AttnProcsLayers(unet.attn_processors)
        lora_layers.load_state_dict(torch.load(pretrained_unet_lora_path))

    weight_dtype = torch.float16 if fp16 else torch.float32
    generator = torch.manual_seed(seed)
    # Do not use controlnet, unconditional sampling
    if pretrained_controlnet_path is None:
        pipeline = StableDiffusionPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet.to(dtype=weight_dtype),
            torch_dtype=weight_dtype
        ).to(device)

        if isinstance(prompt, str):
            prompt = [prompt]

        # Sampling
        sample = pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=[negative_prompt]*len(prompt),
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
            output_type='decode'
        ).images
        sample = torch.from_numpy(sample).permute(0, 3, 1, 2).to(device)
        grid = rearrange(sample, 'b c h w -> c (b h) w')
        grid = rearrange(grid, 'c (n h) w -> c h (n w)', n=int(np.sqrt(num_images_per_prompt)))

    else:   # Use controlnet, conditional sampling

        # Initialize ControlNetModel
        controlnet = ControlNetModel.from_pretrained(pretrained_controlnet_path)
        if enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                controlnet.enable_xformers_memory_efficient_attention()
        print(f"Load pretrained ControlNetModel from {pretrained_controlnet_path}.")

        # Create pipeline
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            pretrained_model_path,
            unet=unet.to(dtype=weight_dtype),
            controlnet=controlnet.to(dtype=weight_dtype),
            torch_dtype=weight_dtype
        ).to(device)

        # Prepare input
        hint_image_ = []
        control_image = []
        if hint_image.ndim == 3:
            hint_image = [hint_image]
        for img in hint_image:
            img = resize_image(HWC3(img), image_reso)
            canny_map = apply_canny(img, canny_low_threshold, canny_high_threshold)
            canny_map = HWC3(canny_map)
            control = torch.from_numpy(canny_map.copy()).float() / 255.0
            control_image.append(control)
            image = torch.from_numpy(img.copy()).float() / 255.0
            hint_image_.append(image)
        hint_image = torch.stack(hint_image_).permute(0, 3, 1, 2).to(device, dtype=weight_dtype)
        control_image = torch.stack(control_image).permute(0, 3, 1, 2).to(device, dtype=weight_dtype)

        if isinstance(prompt, str):
            prompt = [prompt] * control_image.shape[0]

        assert len(prompt) == control_image.shape[0]

        # Sampling
        generator = torch.manual_seed(seed)
        sample = pipeline(
            prompt=prompt,
            image=control_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=[negative_prompt]*len(prompt),
            generator=generator,
            num_images_per_prompt=num_images_per_prompt,
            output_type='decode'
        ).images
        sample = torch.from_numpy(sample).permute(0, 3, 1, 2).to(device)

        # grid = torch.cat([
        #     hint_image[:, None, ...].repeat_interleave(num_images_per_prompt, dim=0),
        #     control_image[:, None, ...].repeat_interleave(num_images_per_prompt, dim=0),
        #     sample[:, None, ...]
        # ], dim=1)
        # grid = rearrange(grid, 'b n c h w -> (b n) c h w')
        # grid = make_grid(grid, nrow=3)
        grid = rearrange(sample, 'b c h w -> c (b h) w')
        grid = rearrange(grid, 'c (n h) w -> c h (n w)', n=int(np.sqrt(num_images_per_prompt)))

    grid = (grid * 255.).permute(1,2,0).cpu().numpy().clip(0, 255).astype(np.uint8)
    imageio.imwrite(output_path, grid)


if __name__ == '__main__':
    
    args = parse_args() 
    main(**args)








    
