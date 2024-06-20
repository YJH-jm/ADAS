import numpy as np
import cv2

from .utils import region_of_interest, hough_lines

import matplotlib.pyplot as plt


class HoughLaneDetector:
    def __init__(self):
        self.rho = 2
        self.theta = 3*np.pi/180
        self.ksize = 5
        self.canny_th = (50, 150)
            

    def detection(self, img):

        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        gray_blur = cv2.GaussianBlur(gray_img, (self.ksize, self.ksize), 0)
        
        edge_img = cv2.Canny(gray_blur, self.canny_th[0], self.canny_th[1])
        
        height, width = gray_img.shape[0], gray_img.shape[1]
        verticies = [(0,height),(2.4*width/5, 1.22*height/2), (2.6*width/5, 1.22*height/2), (width,height)]
        # verticies = [(0,height), (0,0), (width, 0), (width, height)]
        verts = np.array([verticies], dtype=np.int32)
        masked_image = region_of_interest(edge_img, verts)
        
        
        min_line_length = width//16
        max_line_gap = min_line_length//2
        threshold = min_line_length//4
        lines, VERTS = hough_lines(masked_image, self.rho, self.theta, threshold, min_line_length, max_line_gap)
        # plt.title('roi mask')
        # plt.imshow(masked_image)
        # plt.show()
        
        # plt.title('line detection result')
        # plt.imshow(lines)
        # plt.show()
        
        
        # 6. Line 시각화
        result = cv2.addWeighted(lines, 0.8, img, 1.0, 0.0)
        
        return result


if __name__ == "__main__":
    src = "./sample_images/20201125_0_0_00_0_0_1_front_0027287.png"
    img = cv2.imread(src)
    # result = hough_lane_detector(img)
    
    # cv2.imshow("result", result)
    # if cv2.waitKey() == 27:
    #     cv2.destroyAllWindows()