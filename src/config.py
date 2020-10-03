# Hyper params
"""YOLO MODEL PARAMS"""
B = 2 # Num bounding box in one grid cell
S = 7 # Num gridcell
C = 20 # Num classes
IMG_SIZE = 448 # img size
noobject_scale= .5 # params for no object when calculate loss
coord_scale= 5 # params for coord loss

"""Model training hyperparameters"""
epochs = 1
batch_size = 32
use_gpu = True
lr = 1e-3
