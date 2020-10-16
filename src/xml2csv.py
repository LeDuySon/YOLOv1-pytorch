import glob 
import pandas as pd 
from xml.etree import ElementTree as ET
import csv

link = "/home/son/yolo-pytorch/data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations"

xml_files = glob.glob(link + "/*.xml")

print(xml_files[:10])

dict_csv = {}

for xml in xml_files:
    tree = ET.parse(xml)
    root = tree.getroot()
    name = root.find('filename').text
    print(name + '')
    dict_csv[name] = {"boxes" : [],
                    "labels" : [],}
    labels = []
    for i in root.findall('object'):
        boxes = []
        print(i.find('name').text)
        labels.append(i.find('name').text)
        for box in i.find('bndbox'):
            boxes.append(box.text)
            boxes = list(map(int, boxes))
        dict_csv[name]["boxes"].append(boxes)
    dict_csv[name]["labels"] = labels
    
print()     
    # for
fieldnames = ["image", "boxes", "labels"] 
with open("Vocdataset.csv", "w") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames = fieldnames) 
    writer.writeheader()
    for name in dict_csv.keys():
        rows = {}
        rows["image"] = name 
        rows["boxes"] = dict_csv[name]["boxes"]
        rows["labels"] = dict_csv[name]["labels"]
        writer.writerow(rows)
        
    
    