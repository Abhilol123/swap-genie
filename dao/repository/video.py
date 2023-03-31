import cv2
from PIL import Image
import shutil
import logging


class VideoPipeline:
    @staticmethod
    def convert_video_to_data(video_path, no_of_images, name) -> bool:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            logging.error("Error opening video file")
            return False

        no_of_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        for frame in range(0, no_of_frames, int(no_of_frames / no_of_images)):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame)
            success, image = video.read()
            if not success:
                return False
            pil_image = Image.fromarray(image)
            pil_image.save(f"./images/{name}/{str(frame)}.png")

        shutil.make_archive(f"./data/{name}.zip",
                            'zip', f"./images/{name}/{str(frame)}.png")
        logging.info(f"data saved in the folder: ./data/{name}.zip")
        return True
