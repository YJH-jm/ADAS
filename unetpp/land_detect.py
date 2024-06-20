import os
import yaml
import torch
import numpy as np
import cv2

from . import archs


class LaneDetector():
    def __init__(self, device, cfg = "unetpp/config/lane_segmentation.yml", model_path="unetpp/saved/lane_segmentation.pth"):
        
        self.device = device
        with open(cfg) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        model = archs.__dict__[data['arch']](data['num_classes'],data['input_channels'],data['deep_supervision'])
        self.model = model.to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.img_w, self.img_h = data['input_w'],data['input_h']

    def detection(self, ori_img):
        img = cv2.resize(ori_img, (self.img_w,self.img_h))
        img = img.astype('float32') / 255
        img = np.expand_dims(img, axis=0)
        
        img = torch.from_numpy(img).to(self.device)
        img = img.permute(0, 3, 1, 2) 

        output = self.model(img)
        output = output.permute(0, 2, 3, 1).cpu().detach().numpy()

        pred = np.array(output[0])*255
        pred_final = pred[:,:,0] + pred[:,:,1]
        pred_final = cv2.resize(pred_final, (ori_img.shape[1], ori_img.shape[0]))
        _, pred_final = cv2.threshold(pred_final, 250, 255, cv2.THRESH_BINARY)


        lane = np.zeros_like(ori_img)
        lane[:,:,1] = pred_final
        result = cv2.addWeighted(lane, 0.9, ori_img, 1.0, 0.0)
        
        return result


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = "./sample_images/20201125_0_0_00_0_0_1_front_0027287.png"
    img = cv2.imread(src)
    detector = LaneDetector(device)
    detector.detection(img)
    