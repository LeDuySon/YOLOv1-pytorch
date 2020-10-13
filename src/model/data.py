import torch
from torch.utils.data import Dataset, DataLoader, BatchSampler, SequentialSampler
from torchvision import transforms, datasets
from collections import defaultdict
import csv 
from ast import literal_eval
from collections import defaultdict
import os
import sys
import copy 
from PIL import Image, ImageDraw
import time 
import pickle
import pandas as pd

sys.path.append(".")

from utils.utils import BoundingBox, normalize_coordinate

def read_boxes_from_csv(
    csv_path, 
    image_dir="../input/global-wheat-detection/train", 
    preprocess = transforms.Compose([
            transforms.Resize((448, 448)),
            # transforms.CenterCrop(227),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    ):
    """Read data from global-wheat-detection dataset csv file

    Args:
        csv_path (str): path to csv file
    Return:
        map: image_id => image_tensor, bounding_boxes
    """

    df = pd.read_csv(csv_path)
    rows = df[["image_id","bbox"]].groupby("image_id")["bbox"].apply(list).reset_index()
    
    return rows

def collate_fn(batch, device=torch.device("cpu")):
    image_tensors = batch[0][0]
    bounding_boxes = batch[0][1]
    return torch.stack(image_tensors).to(device), bounding_boxes

class GlobalWheatDataset(Dataset):
    """Custom dataset for Global-Wheat
    Load the preprocessed tensor
    """
    def __init__(self, mapping_csv_path=None, image_dir=None, resize_img_width=448, resize_img_height=448):
        self.mapping_csv_path = mapping_csv_path
        self.image_dir = image_dir
        self.resize_img_width = resize_img_width
        self.resize_img_height = resize_img_height

        self.preprocess = transforms.Compose([
            transforms.Resize((resize_img_width, resize_img_height)),
            transforms.ToTensor(),
        ])
        
        self.init_data()

        
    def init_data(self):
        df = pd.read_csv(self.mapping_csv_path)
        self.length = len(df)
        self.data = df.iterrows()

    def __len__(self):
        return self.length

    def __getitem__(self, ids):
        image_tensors, bounding_boxes = [], []
        for idx in ids:
            _, item = next(self.data)

            image_path = f'{item["image_id"]}.jpg'
            image = Image.open(os.path.join(self.image_dir, image_path))
            
            image_width, image_height = image.size 

            image_tensor = self.preprocess(image)
            sample_bounding_boxes = []

            for box in literal_eval(item["bbox"]):
                xmin, ymin, w, h = literal_eval(box)
                x = xmin + w / 2
                y = ymin + h / 2

                x, y = normalize_coordinate(image_width, image_height, x, y)
                w, h = normalize_coordinate(image_width, image_height, w, h)


                sample_bounding_boxes.append(BoundingBox(x, y, w, h, 1, 0))

            image_tensors.append(image_tensor)
            bounding_boxes.append(sample_bounding_boxes)
    
            # Reload data when reaching bottom
            if idx + 1 == self.length:
                self.init_data()
                break

        return image_tensors, bounding_boxes

class DataLoader(DataLoader):
    def __init__(self, dataset, **kargs):
        device = kargs.pop("device", torch.device("cpu"))
        if not "collate_fn" in kargs:
            kargs["collate_fn"] = collate_fn
        kargs["collate_fn"] = lambda x: collate_fn(x, device)
        super(DataLoader, self).__init__(dataset, **kargs)

if __name__ == "__main__":
    # images = read_boxes_from_csv("/home/proxyht/Learning/YOLOv1-pytorch/input/global-wheat-detection/train_full.csv").sample(frac=1)
    # train_set = images[:int(len(images)*0.8)]
    # dev_set = images[int(len(images)*0.8):int(len(images)*0.9)]
    # test_set = images[int(len(images)*0.9):]
    # train_set.to_csv("/home/proxyht/Learning/YOLOv1-pytorch/input/global-wheat-detection/train.csv")
    # dev_set.to_csv("/home/proxyht/Learning/YOLOv1-pytorch/input/global-wheat-detection/dev.csv")
    # test_set.to_csv("/home/proxyht/Learning/YOLOv1-pytorch/input/global-wheat-detection/test.csv")
    
    train_set = GlobalWheatDataset(
        mapping_csv_path="/home/proxyht/Learning/YOLOv1-pytorch/input/global-wheat-detection/train.csv",
        image_dir="/home/proxyht/Learning/YOLOv1-pytorch/input/global-wheat-detection/images"
    )
    train_loader = DataLoader(
        dataset=train_set,
        collate_fn=collate_fn,
        sampler=BatchSampler(
            SequentialSampler(train_set), batch_size=3, drop_last=False
        )
    )
    for item in train_loader:
        print(len(item[0][0]), len(item[1][0]))
    # train_set, dev_set, eval_set = [], [], []
    # dividen = [0.8,0.9,1]

    # prev_img_id = None
    # count_img = 0
    # # total_img = int(os.popen(f"ls ../input/global-wheat-detection/train | wc -l").read().split()[0])
    # total_img = len(images)
    # for idx, key in enumerate(images):
    #     print("Image #",idx,len(images), total_img)
    #     # print(images[key])
    #     if key != prev_img_id:
    #         count_img += 1
        
    #     segment = count_img / total_img
    #     if segment < dividen[0]:
    #         train_set.append(images[key])
    #     elif dividen[0] <= segment < dividen[1]:
    #         dev_set.append(images[key])
    #     else:
    #         eval_set.append(images[key]) 

    # f_train = open("../input/global-wheat-detection/train.txt","w")
    # f_dev = open("../input/global-wheat-detection/dev.txt","w")
    # f_eval = open("../input/global-wheat-detection/eval.txt","w")
    # for item in train_set:
    #     f_train.write(str(item)+"\n")
    # for item in dev_set:
    #     f_dev.write(str(item)+"\n")
    # for item in eval_set:
    #     f_eval.write(str(item)+"\n")