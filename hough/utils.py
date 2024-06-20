import collections
import numpy as np
import cv2
import matplotlib.pyplot as plt
from .vis import draw_lines, weighted_img

def region_of_interest(img, verts):  # 차가 지나가는 방향만 보겠다는 의미 
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    
    
    cv2.fillPoly(mask, verts, ignore_mask_color)
    # plt.title('roi mask')
    # plt.imshow(mask, cmap='gray')
    # plt.show()
    
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    cv2.polylines(line_img, verts, isClosed=True, color=[0, 255, 0], thickness=4)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


prev_left = []
prev_right = []

right_mem = collections.deque(maxlen=10)
left_mem  = collections.deque(maxlen=10)


# def average_lines(lines, img):
    
#     if lines is None: return lines
    
#     global prev_left, prev_right, right_mem, left_mem

#     positive_slopes = []
#     positive_xs = []
#     positive_ys = []
    
#     negative_slopes = []
#     negative_xs = []
#     negative_ys = []

#     min_slope = .3
#     max_slope = 1000

#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             if x1 == x2: continue
#             slope = (y2-y1)/(x2-x1)

#             if abs(slope) < min_slope or abs(slope) > max_slope: continue

#             if slope > 0:
#                 positive_slopes.append(slope)
#                 positive_xs.append(x1)
#                 positive_ys.append(y1)
#             else:
#                 negative_slopes.append(slope)
#                 negative_xs.append(x1)
#                 negative_ys.append(y1)

#     ysize, xsize = img.shape[0], img.shape[1]
#     XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
#     white = img == 255
#     YY[~white] = ysize*2  
    
#     region_top_y = np.amin(YY)
    
#     new_lines = []
#     if len(positive_slopes) > 0:
#         m = np.mean(positive_slopes)
#         avg_x = np.mean(positive_xs)
#         avg_y = np.mean(positive_ys)
        
#         b = avg_y - m*avg_x
        
#         x1 = int((region_top_y - b)/m)
#         x2 = int((ysize - b)/m)
#         prev_left = [(x1, region_top_y, x2, ysize)]

#         left_mem.append([(x1, region_top_y, x2, ysize)])
#         new_lines.append(np.mean(left_mem,axis=0).astype(int))

#     else:
#         if(len(left_mem)>0):
#             new_lines.append(np.mean(left_mem,axis=0).astype(int))
        
    
#     if len(negative_slopes) > 0:
#         m = np.mean(negative_slopes)
#         avg_x = np.mean(negative_xs)
#         avg_y = np.mean(negative_ys)
        
#         b = avg_y - m*avg_x
        
#         x1 = int((region_top_y - b)/m)
#         x2 = int((ysize - b)/m)
        
#         prev_right = [(x1, region_top_y, x2, ysize)]
#         right_mem.append([(x1, region_top_y, x2, ysize)])
#         new_lines.append(np.mean(right_mem, axis=0).astype(int))
#     else:
#         if(len(prev_right)>0):
#             new_lines.append(np.mean(right_mem,axis=0).astype(int))
    
#     return np.array(new_lines)



def average_lines(lines, img):
    '''
    img should be a regioned canny output
    '''
    if lines is None: return lines
    global prev_left, prev_right, right_mem, left_mem

    positive_slopes = []
    positive_xs = []
    positive_ys = []
    negative_slopes = []
    negative_xs = []
    negative_ys = []
    
    min_slope = .3
    max_slope = 1000
    
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2: continue
            slope = (y2-y1)/(x2-x1)
            
            if abs(slope) < min_slope or abs(slope) > max_slope: continue 
                
            positive_slopes.append(slope) if slope > 0 else negative_slopes.append(slope)
            positive_xs.append(x1) if slope > 0 else negative_xs.append(x1)
            positive_ys.append(y1) if slope > 0 else negative_ys.append(y1)
    
    ysize, xsize = img.shape[0], img.shape[1]
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    white = img == 255
    YY[~white] = ysize*2  
    
    region_top_y = np.amin(YY)
    
    new_lines = []
    if len(positive_slopes) > 0:
        m = np.mean(positive_slopes)
        avg_x = np.mean(positive_xs)
        avg_y = np.mean(positive_ys)
        
        b = avg_y - m*avg_x
        
        x1 = int((region_top_y - b)/m)
        x2 = int((ysize - b)/m)
        prev_left = [(x1, region_top_y, x2, ysize)]

        left_mem.append([(x1, region_top_y, x2, ysize)])
        new_lines.append(np.mean(left_mem,axis=0).astype(int))

    else:
        if(len(left_mem)>0):
            new_lines.append(np.mean(left_mem,axis=0).astype(int))
        
    
    if len(negative_slopes) > 0:
        m = np.mean(negative_slopes)
        avg_x = np.mean(negative_xs)
        avg_y = np.mean(negative_ys)
        
        b = avg_y - m*avg_x
        
        x1 = int((region_top_y - b)/m)
        x2 = int((ysize - b)/m)
        
        prev_right = [(x1, region_top_y, x2, ysize)]
        right_mem.append([(x1, region_top_y, x2, ysize)])
        new_lines.append(np.mean(right_mem, axis=0).astype(int))
    else:
        if(len(prev_right)>0):
            new_lines.append(np.mean(right_mem,axis=0).astype(int))
    
    return np.array(new_lines)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),  minLineLength=min_line_len, maxLineGap=max_line_gap)
    avg_lines = average_lines(lines, img)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    verts_ = np.array([[ 0,0],[ 0,0],[ 0, 0], [0,0]])
    print(avg_lines)
    
    if avg_lines is None:
        return line_img, verts_
    if len(avg_lines)<2:
        return line_img, verts_

    if avg_lines[0][0][0] < avg_lines[1][0][0]:
        return line_img, verts_
    
    draw_lines(line_img, avg_lines, color=[0,255,0])

    if(avg_lines is not None):
        if(len(avg_lines)>1):
            avg_lines = np.concatenate(avg_lines).ravel().tolist()


    if(avg_lines is not None):
        if(len(avg_lines)>6):
            verts_ = np.array([[avg_lines[0],avg_lines[1]],
                       [avg_lines[4],avg_lines[5]],
                       [avg_lines[6],avg_lines[7]],
                       [avg_lines[2],avg_lines[3]]])
        
            poly_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
            cv2.fillPoly(poly_img, pts = [verts_], color = (0,255,0))
            line_img = weighted_img(line_img,poly_img,a=1.0, b=.2, l=0.)
            
    return line_img, verts_