import torch
from diffusers import StableDiffusionPipeline
from networks.lora import LoRAModule, create_network_from_weights
from safetensors.torch import load_file
import library.model_util as model_util

# if the ckpt is CompVis based, convert it to Diffusers beforehand with tools/convert_diffusers20_original_sd.py. See --help for more details.

# model_id_or_dir = "weights/model_768.safetensors"
model_id_or_dir = 'stabilityai/stable-diffusion-2'
device = "cuda"

# create pipe
print(f"creating pipe from {model_id_or_dir}...")
# pipe = StableDiffusionPipeline.from_pretrained(
#     "weights/model_768.safetensors",
#     use_safetensors=True
# )
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_dir, torch_dtype=torch.float16)
# text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(True, model_id_or_dir)

pipe = pipe.to(device)
vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet

# Create pipeline
# pipe = StableDiffusionPipeline(
#     vae=vae,
#     text_encoder=text_encoder,
#     unet=unet,
#     tokenizer=None,  # Will be loaded separately
#     scheduler=None,  # Will be loaded separately
#     safety_checker=None,
#     feature_extractor=None,
# )
pipe = pipe.to(device)

pipe.scheduler.set_timesteps(30)

# # Load tokenizer for SD 2.0
# from transformers import CLIPTokenizer
# tokenizer = CLIPTokenizer.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="tokenizer")
# pipe.tokenizer = tokenizer

# # Load scheduler for SD 2.0
# from diffusers import DDIMScheduler
# scheduler = DDIMScheduler.from_pretrained("stabilityai/stable-diffusion-2-base", subfolder="scheduler")
# pipe.scheduler = scheduler

# load lora networks
print(f"loading lora networks...")

lora_path1 = "output/lora/last_converted.safetensors"
sd = load_file(lora_path1)   # If the file is .ckpt, use torch.load instead.
network1, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd, is_sdxl=True)
network1.apply_to(text_encoder, unet)
network1.load_state_dict(sd)
network1.to(device, dtype=torch.float16)


# prompts
prompt = "mewos sitting, beautiful lighting, detailed fur"
negative_prompt = "low quality, worst quality, blurry, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry"

# execute pipeline
print("generating image...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]
# if not merged, you can use set_multiplier
# network1.set_multiplier(0.8)
# and generate image again...

# save image
image.save("generated_images/temple.png")