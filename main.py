import os
from glob import glob
import argparse
import cv2
import torch

from yolo.detect import RoadDetector
from hough.detect import HoughLaneDetector
from unetpp.land_detect import LaneDetector


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("img_path", type=str, help="img file or folder path")
    parser.add_argument("--lane_model", choices=['hough', 'unetpp'], default="unetpp", help="choose lane detector")

    return parser.parse_args()


def main(args):
    
    if os.path.isfile(args.img_path):
        img_list = [args.img_path]
    elif os.path.isdir(args.img_path):
        img_list = glob(os.path.join(args.img_path, '**', '*'), recursive=True)
    
    save_dir = f"result/{args.lane_model}"
    
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo_detector = RoadDetector(device)
    
    if args.lane_model == "unetpp":
        lane_detector = LaneDetector(device)
    elif args.lane_model == "hough":
        lane_detector = HoughLaneDetector()
    
    for file in img_list:
        print(file.split('/')[-1])
        img = cv2.imread(file)
        lane_img = lane_detector.detection(img)
        result = yolo_detector.detect(img, lane_img)
        
        cv2.imwrite(os.path.join(save_dir,f"result_{file.split('/')[-1]}"), result)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)