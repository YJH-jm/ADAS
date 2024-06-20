import numpy as np
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

class RoadDetector:
    def __init__(self, device, model_path="yolo/saved/best.pt"):
        self.device = device
        self.model = YOLO(model_path)
        self.color_dict = {'car':(255,0,0), 'pedestrian':(255, 0, 255)}

    def detect(self, img, lane_img=None):
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.model(img)

        warning_status = False
        black_img = np.zeros_like(lane_img)

        for result in results:
            annotator = Annotator(img)
            boxes = result.boxes
            for box in boxes:
                xyxy = box.xyxy[0]
                cls = int(box.cls)
                p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                if int(xyxy[3]) > black_img.shape[0]*0.95:
                    warning_status = True
                cv2.rectangle(black_img,p1,p2, color = self.color_dict[self.model.names[cls]], thickness=-1)


        final_result = cv2.addWeighted(black_img, 0.6, lane_img, 1.0, 0.0)
        if warning_status:
            # cv2.rectangle(final_result, (0,0), (600, 140), (255,0,0), -1, cv2.LINE_AA)  # filled
            cv2.putText(final_result, 'Warning!', (0,100),cv2.FONT_HERSHEY_DUPLEX, 4,(255,255,255), thickness=3, lineType=cv2.LINE_AA)

        return final_result
   