from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch
import pandas as pd
import json
from PIL import Image
from config import C, S, B # C: num classes, S: Grid_size
import numpy as np
from visualise import imshow, showrect 
from torch.utils.tensorboard import SummaryWriter

from infer import decode_output
from utils import convertxcyc_xmym



preprocess = {"train": transforms.Compose([
    transforms.Resize(448),
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
        for idx, (id, boxes) in enumerate(mapDict.items()):
            image_name = self.image_link + "/" + id + ".jpg"
            self.data_x.append(image_name)
            self.data_y.append(boxes)
            
    def convert_coord(self, *box):
        x, y, w, h = box
        # print("x_b, y_b: ", x, y)
        x_norm, y_norm = x/self.wheat_size, y/self.wheat_size
        loc = [S*x_norm, S*y_norm]
        loc_x = int(loc[0])
        loc_y = int(loc[1])
        x = loc[0] - loc_x
        y = loc[1] - loc_y
        w, h = w/self.wheat_size, h/self.wheat_size
        # print("x, y: ", loc_x, loc_y)
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
        image_name = self.data_x[idx]
        boxes = self.data_y[idx]
        X = Image.open(image_name)
        img_tensor = self.preprocess_img(X)
        # class_prob = torch.zeros(C)            
        y = np.zeros((S, S, 5*B + C))
        for i, box in enumerate(boxes):
            box = json.loads(box)
            xmin, ymin, w, h = box[0], box[1], box[2], box[3]
            x_center, y_center = xmin+w/2, ymin+h/2
            x_center, y_center, w, h, x_idx, y_idx = self.convert_coord(x_center, y_center, w, h) 
            y[y_idx, x_idx] = 1, x_center, y_center, w, h, 1, x_center, y_center, w, h, 1
        y_tensor = torch.from_numpy(y)

        # X = self.data_x[idx]
        # y = self.data_y[idx]
        return img_tensor, y_tensor
    
    def __len__(self):
        return len(self.data_x)

if __name__ == '__main__':
    dt = GlobalWheatData(link, image_link, preprocess)
    
    # print(dt[0])
    # train_data = torch.utils.data.DataLoader(dt, batch_size = 8, shuffle = True)
    # testing_x, testing_y = next(iter(train_data))
    # test_grid = torchvision.utils.make_grid(testing_x)
    
    # print(testing_y)
    
    # writer = SummaryWriter()
    # # writer.add_image('my_image_HWC', img_HWC, 0, dataformats='HWC')

    # # writer.add_image('my_image', img, 0)
    # # imshow(test_grid)
    # writer.add_image('my_image', test_grid)
    
    # writer.close()
    img_tensor, out_tensor, bo = dt[1]
    boxes, prob = decode_output(out_tensor)
    box_in = [c for i in boxes for c in i]
    for i in range(len(box_in)):
        box_in[i] =  convertxcyc_xmym(box_in[i])
    print(box_in)
    showrect(img_tensor, box_in)
    print(bo)
    showrect(img_tensor, list(map(json.loads, bo)))