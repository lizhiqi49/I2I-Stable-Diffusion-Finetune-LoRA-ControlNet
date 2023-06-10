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
from typing import Optional, Literal
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
    StableDiffusionPipeline
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


logger = get_logger(__name__, log_level="INFO")

# modify here
data_root = "./data/chinesepainting"


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
    *,
    exp_name: str,
    mode: Literal['lora', 'textual_inversion'] = 'lora',
    num_images_to_train: int = 20,
    pretrained_model_path: str = "runwayml/stable-diffusion-v1-5",
    # controlnet_model_path: str = "lllyasviel/sd-controlnet-depth",
    lora_rank: int = 4,
    output_dir: str = "./output",
    seed: Optional[int] = 0,
    learning_rate: float = 1e-5,
    train_batch_size: int = 6,
    max_train_steps: int = 500,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-3,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    mixed_precision: Optional[str] = "fp16",
    scale_lr: bool = False,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = False,
    checkpointing_step_interv: int = 1000,
    validation_step_interv: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    enable_xformers_memory_efficient_attention: bool = True,
    num_inference_steps: int = 20,
    num_validation_images: int = 8,
    validation_prompt: str = "a painting in the style of Chinese painting",
    guidance_scale: float = 9.0,
    negative_prompt: str = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
):
    *_, config = inspect.getargvalues(inspect.currentframe())
    output_dir = os.path.join(output_dir, exp_name)
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with="tensorboard",
        project_dir=os.path.join(output_dir, 'logs')
    )
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Handle the output folder creation
    if accelerator.is_main_process:
        # now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        # output_dir = os.path.join(output_dir, now)
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/validation", exist_ok=True)
        os.makedirs(f"{output_dir}/pretrained", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    # Freeze vae, unet and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # mixed precision
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)


    lora_attn_procs = init_lora_attn(unet, lora_rank)
    unet.set_attn_processor(lora_attn_procs)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")
        
    lora_layers = AttnProcsLayers(unet.attn_processors)
    
    if gradient_checkpointing:
        lora_layers.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = (
            learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        lora_layers.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon
    )

    # Dataset creation
    train_dataset = LoRADataset(
        root=data_root,
        split="train",
        length=num_images_to_train,
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
    )
    
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    lora_layers, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        lora_layers, optimizer, train_dataloader, lr_scheduler
    )



    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(exp_name)

    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        
        if path is None:
                accelerator.print(
                    f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    @torch.no_grad()
    def get_input(batch, encode_image=True, encode_prompt=True):
        """
        the batch should contain:
        1. hint: a batch of condition images (depth maps) with 3 channels and values normalized to 0~1.
        2. image: a batch of ground truth images with values normalized to -1~1.
        3. camera_params: a batch of camera_parameters.
        4. prompts: a string of prompt or a list of prompts whose length should equal to batch size.
        """
        x = batch['image']
        B = x.shape[0]
        device = accelerator.device
        
        if x.shape[-1] == 3:
            x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(device=device, memory_format=torch.contiguous_format)

        if encode_image:
            # Convert images to latent space
            x = vae.encode(batch["image"].to(dtype=weight_dtype)).latent_dist.sample()
            x = x * vae.config.scaling_factor

        prompt = batch['prompt']
        if isinstance(prompt, str):
            prompt = [prompt] * B
        elif isinstance(prompt, list):
            assert len(prompt) == B, "The length of prompt list should equal to batch size!"

        if encode_prompt:
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = tokenizer.batch_decode(
                    untruncated_ids[:, tokenizer.model_max_length - 1 : -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(text_encoder.config, "use_attention_mask") and text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt = text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt = prompt[0]

            prompt = prompt.to(dtype=text_encoder.dtype, device=device)

        return x, prompt
    


    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Get input
                x, prompt_embed = get_input(batch)
                t = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                    (x.shape[0],), device=accelerator.device
                ).long()
                # Add noise
                noise = torch.randn_like(x)
                x_noisy = noise_scheduler.add_noise(x, noise, t)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(x, noise, t)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(x_noisy, t, prompt_embed).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_step_interv == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        if accelerator.is_main_process:
            if validation_prompt is not None and global_step % validation_step_interv == 0:
                logger.info(
                    f"Running validation... \n Generating {num_validation_images} images with prompt:"
                    f" {validation_prompt}."
                )
                # create pipeline
                pipeline = StableDiffusionPipeline.from_pretrained(
                    pretrained_model_path,
                    unet=accelerator.unwrap_model(unet),
                    torch_dtype=weight_dtype,
                )
                pipeline = pipeline.to(accelerator.device)
                pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                images = []
                for _ in range(num_validation_images):
                    images.append(
                        pipeline(
                            validation_prompt, 
                            num_inference_steps=num_inference_steps, 
                            generator=generator,
                            guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt,
                        ).images[0]
                    )

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = unet.to(torch.float32)
        unet.save_attn_procs(output_dir)
            
    accelerator.end_training()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/lora_chinesepainting.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))









