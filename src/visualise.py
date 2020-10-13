import torch 
import numpy as np
import matplotlib.pyplot as plt 
from PIL import Image, ImageDraw 
import pandas as pd #testing
import json

def imshow(img):
    img = img.detach().numpy().transpose((1, 2, 0))
    plt.figure(figsize = (15, 7))
    plt.imshow(img)
    plt.show()
    
def showrect(image_fn, bbox):
    img = Image.open(image_fn).convert('RGBA')

    bbox_canvas = Image.new('RGBA', img.size)
    bbox_draw = ImageDraw.Draw(bbox_canvas)
    for box in bbox:
        x, y, w, h = box[0], box[1], box[2], box[3]
        bbox_draw.rectangle((x, y, x+w, y+h), fill=(255, 255, 255, 0), outline=(255, 0, 0, 255), width = 5)
    img = Image.alpha_composite(img, bbox_canvas)
    img = img.convert("RGB") # Remove alpha for saving in jpg format.
    viz_img = np.asarray(img)
    plt.figure(figsize=(15, 15))
    plt.title(img)
    plt.imshow(viz_img)
    plt.show()

if __name__ == '__main__':
    img_link = "/home/son/yolo-pytorch/data/global-wheat-detection/train/0a3cb453f.jpg"
    df = pd.read_csv("/home/son/yolo-pytorch/data/global-wheat-detection/train.csv")
    df["bbox"] = df["bbox"].apply(json.loads)
    bbox = df[df["image_id"] == "0a3cb453f"]["bbox"].values
    showrect(img_link, bbox)