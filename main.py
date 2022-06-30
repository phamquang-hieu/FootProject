from skimage.io import imread
from utils import *
import argparse
import numpy as np

def calc(img_path, n_run=3, paper_width=210.0, paper_height=297.0, color_sys='RGB'):
    img = imread(img_path)
    kernel = np.ones((21, 21), np.uint8) # for erosion vss dilation

    preprocessedImg = preprocess(img, color_sys)
    foot_size_final = np.array([0.0, 0.0])
    stages = []
    for i in range(n_run):
        clusteredImg = GMM_cluster(preprocessedImg)
        clusteredImg = cv2.medianBlur(clusteredImg, 15)

        clusteredImg = cv2.erode(clusteredImg, kernel)
        clusteredImg = cv2.dilate(clusteredImg, kernel)

        edgedImg = edgeDetection(clusteredImg)

#         paper = contours(edgedImg)[0]
#         with_paper = edgedImg[paper[:, 1].min():paper[:, 1].max(),  paper[:, 0].min() : paper[:, 0].max()]
#         x, y = with_paper.shape[1]//10, with_paper.shape[0]//10


#         cropped = with_paper[x:-x, y:-y]
        
        paper, paper_with_edge_length = contours(edgedImg)
        
        paper_w = int(paper_with_edge_length[1][0])
        paper_h = int(paper_with_edge_length[1][1])
        
        
        # with_paper = edgedImg[paper[:, 1].min():paper[:, 1].max(),  paper[:, 0].min() : paper[:, 0].max()]
        M = cv2.getPerspectiveTransform(paper.astype('float32'), np.float32([
                                                [0, 0],
                                                [0, paper_h - 1],
                                                [paper_w - 1, paper_h - 1],
                                                [paper_w - 1, 0]]))
        
        cropped = cv2.warpPerspective(edgedImg, M, (paper_w, paper_h), flags=cv2.INTER_LINEAR)
        if paper_w > paper_h:
            paper_w, paper_h = paper_h, paper_w
            
        crop_rate = 0.05
        cropped = cropped[int(paper_h*crop_rate):-int(paper_h*crop_rate), int(paper_w*crop_rate):-int(paper_w*crop_rate)]
        
        cropped = cv2.dilate(cropped, kernel)
        cropped = cv2.erode(cropped, kernel)
        _, foot_w_h = contours(cropped)
        
        foot_w, foot_h = foot_w_h[1]
        if foot_w > foot_h:
            foot_w, foot_h = foot_h, foot_w
        
        foot_size = np.array([foot_w, foot_h])/np.array([paper_w, paper_h])
        
        # print(h_w(foot))
        # print(h_w(paper))
        foot_size[0] *= paper_width
        foot_size[1] *= paper_height
        foot_size_final += foot_size
        if i == n_run -1:
            edgedImg = cv2.dilate(edgedImg, kernel)
            cropped = cv2.dilate(cropped, kernel)
            stages = [clusteredImg, cropped, edgedImg]
    
    foot_size_final/=n_run
    foot_size_final/=10.0
    foot_size_final = np.round(foot_size_final, 1)

    print("foot size (w, h) cm:", foot_size_final)
    
    
    return {"width": foot_size_final[0], "length":foot_size_final[1]}, stages

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-path', '--path', help='path to the foot image')
    parser.add_argument('-width', '--width', type=float, default=210.0, help='width of the paper')
    parser.add_argument('-height', '--height', type=float, default=297.0, help='height of the paper')
    parser.add_argument('-n_r', '--n_run', type=int, default=3, help='numbers of run')
    parser.add_argument('-c_s', '--color_sys', type=str, default="RGB", help='image color system to convert to from RGB')
    args = parser.parse_args()
    
    calc(args.path, args.n_run, args.width, args.height, args.color_sys)
    
    