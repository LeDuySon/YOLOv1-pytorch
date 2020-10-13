import torch
from torchvision import transforms
import sys
from PIL import Image, ImageDraw
from model.model import YOLOv1
import pandas as pd 
from ast import literal_eval
from utils.utils import BoundingBox

class Infer():
    def __init__(self, model=None, img_size=448, grid_size=7, threshold=0.5):
        self.img_size = img_size
        self.grid_size = grid_size
        self.threshold = threshold
        self.model = model
        if model:
            self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    def infer(self, img_path, ground_truth_boxes=None):
        im = Image.open(img_path)
        origin_w, origin_h = im.size

        # im = im.resize((self.img_size, self.img_size))
        draw = ImageDraw.Draw(im)

        if ground_truth_boxes:
            for box in ground_truth_boxes:
                box = literal_eval(box)
                print(box)
                x0, y0, w, h = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                print("TRUE CORD:", x0+w/2, y0+h/2, w, h)
                x1, y1 = x0 + w, y0 + h
                # draw.rectangle(([(x0,y0),(x1,y1)]),outline=(0,255,0), width=3)

        if self.model:
            image_tensor = self.preprocess(im)
            print(image_tensor, image_tensor.shape)
            output = model(torch.stack([image_tensor]))[0]

            pred_boxes = []
            print(output,output.shape)
            for grid_y in range(output.shape[0]):
                for grid_x in range(output.shape[1]):
                    pred_box_1 = output[grid_y][grid_x][:5]
                    pred_box_2 = output[grid_y][grid_x][5:10]

                    x1, y1, w1, h1, conf1 = pred_box_1
                    x2, y2, w2, h2, conf2 = pred_box_2

                    w1 *= origin_w 
                    h1 *= origin_h 
                    x1 = (x1+grid_x) / self.grid_size * origin_w 
                    y1 = (y1+grid_y) / self.grid_size * origin_h
                    
                    w2 *= origin_w
                    h2 *= origin_h
                    x2 = (x2+grid_x) / self.grid_size * origin_w 
                    y2 = (y2+grid_y) / self.grid_size * origin_h

                    pred_boxes.append(BoundingBox(x1, y1, w1, h1, conf1))
                    pred_boxes.append(BoundingBox(x2, y2, w2, h2, conf2))

            for idx, box in enumerate(pred_boxes):
                print("BOX",idx,box)
                x, y, w, h, confidence = box.x, box.y, box.w, box.h, box.confidence
                x0, y0 = x - w/2, y-h/2
                x1, y1 = x + w/2, y + h/2
                if confidence > self.threshold:
                    draw.rectangle(([(x0,y0),(x1,y1)]),outline=(255,0,0), width=3)

        im.show()


if __name__ == "__main__":
    df = pd.read_csv("../input/global-wheat-detection/train_small.csv")
    sample = df.head(10)
    img_path, true_boxes = f'../input/global-wheat-detection/images/{sample["image_id"][0]}.jpg', literal_eval(sample["bbox"][0])

    model = None
    ckpt_path = sys.argv[1]

    ckpt = torch.load(ckpt_path)
    model = YOLOv1(grid_size=7, num_bounding_boxes=2, num_labels=1, last_layer_hidden_size=256)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

    
    
    inferer = Infer(model, threshold=0.2)
    inferer.infer(img_path, ground_truth_boxes=true_boxes)