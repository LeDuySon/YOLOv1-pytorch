import torchvision
import torchvision.transforms as transforms
import torch

def convertxyxy_xywh(box):
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    x_c, y_c = (xmin+xmax)/2, (ymin+ymax)/2
    w, h = xmax-xmin, ymax-ymin
    return x_c, y_c, w, h        

def convertxywh_xyxy(box):
    x_c, y_c, w, h = box[0], box[1], box[2], box[3] 
    xmin, ymin = x_c-w/2, y_c-h/2 
    xmax, ymax = x_c+w/2, y_c+h/2
    return xmin, ymin, xmax, ymax

def convertxcyc_xmym(box):
    x_c, y_c, w, h = box[0], box[1], box[2], box[3]
    x_m = x_c - w/2
    y_m = y_c - h/2
    return x_m, y_m, w, h
    
    
    
    
    
    
def bbox_iou(box1, box2):
    """Compute IOU between two boxes

    Args:
        box: [x_center, y_center, w, h]
    """
    pass


