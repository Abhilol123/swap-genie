from fastapi import Request
from dao.repository.inference import inferencePipeline
from PIL import Image
from io import BytesIO
import base64


class InferenceController:
    async def inference(self, request: Request):
        body = await request.json()
        model_name = body["model_name"]
        pronoun = body["pronoun"]
        image = Image.open(BytesIO(base64.b64decode(body["image"]))).convert(
            "RGB").resize((512, 512))
        image_result = inferencePipeline.inference(
            model_name, image, pronoun)
        return image_result


inferenceController = InferenceController()
