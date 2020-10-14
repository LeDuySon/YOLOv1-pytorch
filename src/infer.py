import torch
from model import YOLOD
from config import S, B
import numpy as np
from PIL import Image
import time
import torchvision.transforms as transforms

from visualise import showrect
import glob
from utils import convertxcyc_xmym

preprocess = {"train": transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),]),
               "test": transforms.Compose([
    transforms.Resize(448),
    transforms.ToTensor(),
])
}


def predict(model, image, use_cuda = False):
    if not isinstance(image, torch.Tensor):    
        img = Image.open(image)
        img_tensor = preprocess["test"](img)
    else:
        img_tensor = image
    img_tensor = img_tensor.unsqueeze(0)
    if(use_cuda):
        img_tensor = img_tensor.cuda()
    output_tensor = model(img_tensor)
    boxes, prob = decode_output(output_tensor)
    box_in = [c for i in boxes for c in i]
    for i in range(len(box_in)):
        box_in[i] =  convertxcyc_xmym(box_in[i])
    showrect(image, box_in)

    
    

def decode_output(pred_tensor, threshold = 0.1, init_size = 1024):
    """Decode output model

    Args:
        pred_tensor ([tensor]): S*S*(B*5+C) 
        threshold (float, optional): Threshold confidence for having object or not. Defaults to 0.5.
    """
    
    box_tensor = pred_tensor[..., :B*5]
    class_tensor = pred_tensor[..., B*5:]
    boxes = []
    probC = []
    for i in range(box_tensor.shape[0]):
        for j in range(box_tensor.shape[1]):
            cbox = box_tensor[i][j]
            cbox = cbox.contiguous().view(-1, 5)
            cbox_mask = cbox[:, 0] > threshold
            cbox = cbox[cbox_mask]
            # cbox[cbox < 0] = 0
            if(cbox.shape[0] == 0):
                continue
            cbox[:, 1] = (cbox[:, 1] + i)/S * init_size
            cbox[:, 2] = (cbox[:, 2] + j)/S * init_size
            cbox[:, 3:] = cbox[:, 3:] * init_size
            boxes.append(cbox[:, 1:].detach().cpu().numpy().tolist()) 
            probC.append(torch.argmax(class_tensor[i][j]).cpu().numpy())
    return boxes, probC
            
    
if __name__ == '__main__':
    start = time.time()
    checkpoint = torch.load("best_model_2.pth")
    model = YOLOD()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(time.time() - start)
    image_link = "../../yolo-pytorch/data/global-wheat-detection/test/0a3cb453f.jpg"
    pred_test = np.random.normal(3, 2.5, size=(7, 7, 11))
    pred_tensor = torch.from_numpy(pred_test)
    predict(model, image_link)
    # print(decode_output(pred_tensor))