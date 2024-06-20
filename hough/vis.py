import cv2

def weighted_img(initial_img, img, a=0.8, b=1., l=0.):
    return cv2.addWeighted(initial_img, a, img, b, l)

def draw_lines(img, lines, color=[255, 0, 0], thickness=8):
    if lines is None: return lines
    
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)
