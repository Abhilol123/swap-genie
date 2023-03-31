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
        self.diffusion_pipe = self.diffusion_pipe.to("cuda")
        self.diffusion_pipe.enable_xformers_memory_efficient_attention()
        self.diffusion_pipe.safety_checker = lambda images, **kwargs: images, False
        self.clip_interrogator = Interrogator(
            Config(clip_model_name="ViT-L-14/openai"))
        self.target_text_prompt = "face with hair"
        self.negative_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
        return None

    def get_face_mask(self, image) -> Image:
        with torch.no_grad():
            inputs = self.processor(text=self.target_text_prompt, images=image,
                                    padding="max_length", return_tensors="pt").to("cuda")
            outputs = self.model(**inputs)
            mask = torch.sigmoid(outputs.logits).cpu(
            ).detach().unsqueeze(-1).numpy()
            mask_pil = Image.fromarray(
                np.uint8(cm.gist_earth(mask)*255), mode='RGBA')
            return mask_pil

    def inference(self, model_name, image, pronoun="man"):
        patch_pipe(self.diffusion_pipe, f"./lora/{model_name}.safetensors")
        tune_lora_scale(self.diffusion_pipe.unet, 0.6)
        inferred_prompt = self.clip_interrogator.interrogate(image)
        prompt = f"<s1><s2> {inferred_prompt}"
        inference_image = self.diffusion_pipe(
            image=image,
            text=self.target_text_prompt,
            prompt=prompt.replace("man", pronoun),
            negative_prompt=self.negative_prompt
        ).images[0]
        return inference_image
