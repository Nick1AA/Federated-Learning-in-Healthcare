from model import *
import cv2
from PIL import Image
from pathlib import Path

if __name__ == "__main__":
    image = "/pfs/data5/home/kit/aifb/sq8430/Federated-Learning-in-Healthcare/data/CheXpert-v1.0-small/train/patient00001/study1/view1_frontal.jpg"
    print(image)
    path = Path(image)
    if path.is_file():
        image = Image.open(image)
        print(image.mode)