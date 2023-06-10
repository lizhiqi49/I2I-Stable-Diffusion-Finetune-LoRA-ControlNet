# Stable Diffusion fine-tuning with LoRA plus ControlNet attachment for image transfer

This is a project of course "Deep Learning" of Westlake University, and this repository seeks to perform few-shot image transfer by fine-tuning Stable Diffusion on a small dataset contains several images of target domain, and then sampling attached with ControlNet.


## Quickstart

### Setup environment

1. Install Pytorch

This project is experimented on Pytorch-1.2, please refer to [Pytorch's official webpage](https://pytorch.org/) for installation.

2. Install dependency packages

```bash
git clone https://github.com/lizhiqi49/Stable-Diffusion-Finetune
cd Stable-Diffusion-Finetune
pip install -r requirements.txt
```

### For pretrained Stable Diffusion and ControlNet

The different versions of pretrained Stable Diffusion and ControlNet are all hosted on [huggingface-hub](https://huggingface.co/).
The version of Stable Diffusion we used is [*runwayml/stable-diffusion-v1-5*](https://huggingface.co/runwayml/stable-diffusion-v1-5), and the version of ControlNet we used is [*lllyasviel/sd-controlnet-canny*](https://huggingface.co/lllyasviel/sd-controlnet-canny).
For more details about usage of huggingface's model please refer to [this page](https://huggingface.co/docs/diffusers).


### Dataset

Put your image collection in a folder and set `data_root` in the script `train.py` to the path of the folder. 



### Start training

1. Configure hyper-parameters

Configure your own training hyper-parameters under `configs/{exp_name}.yaml`.

2. Configure Accelerate

This project uses library [Accelerate](https://github.com/huggingface/accelerate) for mixed-precision and distributed training, before training start, you need configure your accelerate using `accelerate config` on your shell. 

3. Train!

```
accelerate launch train.py --config configs/{exp_name}.yaml
```

### Sampling

Use `sample.py` for target domain image sampling or image transfer:

```
python sample.py \
    --pretrained_unet_lora_path ... \   # Defaultly None, sampling using the pretrained Stable Diffusion
    --pretrained_controlnet_path ... \  # Defaultly None, sampling without ControlNet, can only perform text-to-image generation
    --pretrained_model_path ... \   # Defaultly 'runwayml/stable-diffusion-v1-5', can't be None
    --hint_image xxx.png \   # The path to your source image
    --prompt "a painting" \
    --num_images_per_prompt 4 \
    --num_inference_steps 30 \
    --guidance_scale 9.0 \
    --output_path xxx.png \
    --fp16
```


