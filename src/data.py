from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch
import pandas as pd
import json
from PIL import Image
from config import C, S, B # C: num classes, S: Grid_size
import numpy as np
from visualise import imshow



preprocess = {"train": transforms.Compose([
    transforms.Resize(448),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),]),
               "test": transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
])
}

link = "../../yolo-pytorch/data/global-wheat-detection/train.csv"

image_link = "../../yolo-pytorch/data/global-wheat-detection/train"

class GlobalWheatData(Dataset):
    def __init__(self, csv_file, image_link, preprocess, img_size = 448, mode = "train"):
        """Handle GlobalWheat Dataset for yolo training and testing

        Args:
            csv_file ([file]): bounding box coord file
            image_link ([type]): image file
            preprocess ([type]): augment data
            img_size (int, optional): img size input in yolo model. Defaults to 448.
            mode (str, optional): process data for training or testing. Defaults to "train".
        """
        super(GlobalWheatData, self).__init__()
        self.file = csv_file
        self.img_size = img_size
        self.wheat_size = 1024
        self.image_link = image_link
        self.mode = mode
        self.preprocess = preprocess
        self.data_x = []
        self.data_y = []
        self.load_data()
    def load_data(self):
        df = pd.read_csv(self.file)
        box_coord = df[["image_id", "bbox"]].groupby("image_id")["bbox"].apply(list).reset_index()
        mapDict = {k:v for k, v in zip(box_coord["image_id"], box_coord["bbox"])}
        N = len(mapDict.keys())
        X = np.zeros((N, self.img_size, self.img_size, 3), dtype='uint8')
        for idx, (id, boxes) in enumerate(mapDict.items()):
            image_name = self.image_link + "/" + id + ".jpg"
            X = Image.open(image_name)
            img_tensor = self.preprocess_img(X)            
            y = np.zeros((S, S, 5*B+ C))
            for i, box in enumerate(boxes):
                box = json.loads(box)
                xmin, ymin, w, h = box[0], box[1], box[2], box[3]
                # convert coord from 1024 image size to 448 image size
                # xmin, ymin, w, h = xmin/self.wheat_size * 448, ymin/1024 * 448, w/1024 * 448, h/1024 * 448
                x_center, y_center = (xmin+w)/2, (ymin+h)/2
                x_center, y_center, w, h, x_idx, y_idx = self.convert_coord(x_center, y_center, w, h) 
                # x_idx, y_idx = int(x_center/self.img_size * S), int(y_center/self.img_size * S)
                y[x_idx, y_idx] = 1, x_center, y_center, w, h, 1, x_center, y_center, w, h, 1
            print(idx)
            if(idx == 100):
                break
            y_tensor = torch.from_numpy(y)
            self.data_x.append(img_tensor)
            self.data_y.append(y_tensor)
    def convert_coord(self, *box):
        x, y, w, h = box
        x_norm, y_norm = x/self.wheat_size, y/self.wheat_size
        loc = [S*x_norm, S*y_norm]
        loc_x = int(loc[0])
        loc_y = int(loc[1])
        x = loc[0] - loc_x
        y = loc[1] - loc_y
        w, h = w/self.img_size, h/self.img_size
        return x, y, w, h, loc_x, loc_y
        
    def preprocess_img(self, img):
        if self.mode == "train":
            return_img = self.preprocess[self.mode](img)
        elif self.mode == "test":
            return_img = self.preprocess[self.mode](img)
        else:
            raise Exception("Wrong mode")
        return return_img
    def __getitem__(self, idx):
        """Generate data

        Args:
            idx ([int]): index data

        Returns:
            X: (tensor) ->  3*448*448
            y: (tensor) -> (S*S*(5+C)): 5+C: confidence, x, y, w, h, class_prob
        """
        X = self.data_x[idx]
        y = self.data_y[idx]
        return X, y
    
    def __len__(self):
        return len(self.data_x)

if __name__ == '__main__':
    dt = GlobalWheatData(link, image_link, preprocess)
    # print(dt[0])
    train_data = torch.utils.data.DataLoader(dt, batch_size = 8, shuffle = True)
    testing_x, testing_y = next(iter(train_data))
    test_grid = torchvision.utils.make_grid(testing_x)
    imshow(test_grid)