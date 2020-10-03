import torchvision
import torchvision.transforms as transforms

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

def bbox_iou(box1, box2):
    """Compute IOU between two boxes

    Args:
        box: [x_center, y_center, w, h]
    """


