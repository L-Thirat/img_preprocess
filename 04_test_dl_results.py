import cv2
import json
import numpy as np
from utils import load_config, extract_filename, load_base_xml, write_base_cml
# from shapely.geometry import Polygon

# project setting
# todo on colab
project = "car/"
project_path = "../projects/" + project
# project_path = "../../drive/MyDrive/projects/" + project

# load config
cfg = load_config("./")
cfg["train_data_dir"] = project_path + "datasets/" + 'train'
cfg["val_data_dir"] = project_path + "datasets/" + 'val'
cfg["test_data_dir"] = project_path + "datasets/" + 'test'


sum_path = "../projects/%s/results/summarize.json" % project
with open(sum_path, 'r') as outfile:
    data = json.load(outfile)

for f, val in data.items():
    detection_boxes = val["detection_boxes"]
    detection_classes = val["detection_classes"]
    detection_scores = val["detection_scores"]
    print(len(detection_boxes))
    print(len(detection_classes))
    print(len(detection_scores))

    img = cv2.imread(cfg["test_data_dir"]+"/"+f)
    h, w, _ = img.shape
    print(img.shape)
    if cfg["fill_square"]:
        ratio_w = max(w,h)/cfg["img_resize"]
        ratio_h = ratio_w
    else:
        ratio_w = w/cfg["img_resize"]
        ratio_h = h/cfg["img_resize"]

    # idx_scores = set()
    for score, box, clf in zip(detection_scores, detection_boxes, detection_classes):
        if clf == 1 and score > 0.95:
            start_point = (int(box[1]*cfg["img_resize"]*ratio_w), int(box[0]*cfg["img_resize"]*ratio_h))
            end_point = (int(box[3]*cfg["img_resize"]*ratio_w), int(box[2]*cfg["img_resize"]*ratio_h))
            img = cv2.rectangle(img, start_point, end_point, (0, 255, 0), 2)
        elif clf == 2 and score > 0.3:
            start_point = (int(box[1]*cfg["img_resize"]*ratio_w), int(box[0]*cfg["img_resize"]*ratio_h))
            end_point = (int(box[3]*cfg["img_resize"]*ratio_w), int(box[2]*cfg["img_resize"]*ratio_h))
            img = cv2.rectangle(img, start_point, end_point, (0,0,255), 2)
        else:
            break
    # Polygon(0.5, 0.5).within(Polygon(coords))

    # print(idx_scores)
    print(detection_boxes[0])
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
