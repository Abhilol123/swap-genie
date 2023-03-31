from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import DiffusionPipeline
from lora_diffusion import patch_pipe, tune_lora_scale
from PIL import Image
from clip_interrogator import Config, Interrogator
from matplotlib import cm
import torch
import numpy as np


class InferencePipeline:
    def __init__(self) -> None:
        self.processor = CLIPSegProcessor.from_pretrained(
            "CIDAS/clipseg-rd64-refined")
        self.model = CLIPSegForImageSegmentation.from_pretrained(
            "CIDAS/clipseg-rd64-refined")
        self.diffusion_pipe = DiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            custom_pipeline="text_inpainting",
            segmentation_model=self.model,
            segmentation_processor=self.processor
        )
        self.ci = Interrogator(Config(clip_model_name="ViT-L-14/openai"))
        return None



pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
patch_pipe(pipe, f"./Cathy.safetensors")
tune_lora_scale(pipe.unet, 0.6)

image = Image.open("input.png").convert('RGB').resize((512, 512))
inferred_prompt = ci.interrogate(image)
print(inferred_prompt)

text = "face with hair"
prompt = f"<s1><s2> {inferred_prompt}"

with torch.no_grad():
    inputs = processor(text=text, images=image,
                       padding="max_length", return_tensors="pt").to("cuda")
    outputs = model(**inputs)
    mask = torch.sigmoid(outputs.logits).cpu().detach().unsqueeze(-1).numpy()
    mask_pil = Image.fromarray(np.uint8(cm.gist_earth(mask)*255), mode='RGBA')
    mask_pil.save("mask.png")


def dummy(images, **kwargs):
    return images, False


pipe.safety_checker = dummy

image = pipe(
    image=image,
    text=text,
    prompt=prompt,
    negative_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
).images[0]
image.save("output.png")
