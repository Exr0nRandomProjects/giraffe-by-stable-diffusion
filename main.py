import torch
from torch import autocast
from diffusers import StableDiffusionPipeline
from PIL import Image

from tqdm import trange


model_id = "CompVis/stable-diffusion-v-1-3-original"
device = "cuda"

pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
pipe = pipe.to(device)

def merge_images(images):
    widths = [img.size[0] for img in images]
    heights = [img.size[1] for img in images]
    new_image = Image.new('RGB',(sum(widths), max(heights)), (250,250,250))


    left_px = 0
    for img in images:
        new_image.paste(img, (left_px, 0))
        left_px += img.size[0]

    return new_image

while True:
# for i in trange(1000):
    prompt = [input("prompt? ")]*4
    # prompt = "abstract giraffe logo"

    images = pipe(prompt, num_inference_steps=64)["sample"]

    # merge_images(images).save(f"out-huxley-{i}.png")
    merge_images(images).save("out.png")

