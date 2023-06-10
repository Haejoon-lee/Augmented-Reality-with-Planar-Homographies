import numpy as np
import cv2
import sys
from PIL import Image
from opts import get_opts
from tqdm import tqdm
import time

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, computeH, computeH_norm, compositeH # Import necessary functions
from helper import loadVid

"""
you are going to track the computer vision text book in each frame of book.mov, 
and overlay each frame of ar_source.mov onto the book in book.mov.
"""
    
start = time.time()
#Write script for Q3.1
opts = get_opts()

img_cv_covoer_path = '../data/cv_cover.jpg'
img_cv_cover = Image.open(img_cv_covoer_path).convert("RGB")  
img_cv_cover = np.array(img_cv_cover).astype(np.float32) / 255 #normalize

vd_book_path = "../data/book.mov"
vd_source_path = "../data/ar_source.mov"

vd_book = loadVid(vd_book_path)
vd_source = loadVid(vd_source_path)

ar_vd_list = []

# for i in tqdm(range(110, 130)):
for i in tqdm(range(len(vd_source))):

    frame_book = np.array(vd_book[i]).astype(np.float32) / 255
    frame_source = np.array(vd_source[i]).astype(np.float32) / 255 #normalize

    # cv_cover: 440x350x3, frame_source: 360x640x3 -> crop frame to get rid of black background and have the same ratio as cv_cover  
    # then resize the frame
    frame_source_crop = frame_source[44:314, 212:427, :]
    h_cv_cover, w_cv_cover = img_cv_cover.shape[0], img_cv_cover.shape[1]
    frame_source_resized = cv2.resize(frame_source_crop, (w_cv_cover, h_cv_cover))
    # rotated_image = ndimage.rotate(cv_desk, 40, reshape=False)

    matches, locs1, locs2 = matchPics(frame_book, img_cv_cover, opts)

    locs1_matched = locs1[matches[:, 0]]
    locs2_matched = locs2[matches[:, 1]]

    bestH2to1, inliers = computeH_ransac(locs1_matched, locs2_matched, opts)

    img_composited = compositeH(bestH2to1, frame_book, frame_source_resized)
    
    ar_vd_list.append(img_composited)

print("time :", time.time() - start, "s")

#Saving video
ar_vd = np.stack(ar_vd_list, axis = 0)

width = ar_vd.shape[2]
hieght = ar_vd.shape[1]
channel = ar_vd.shape[3]
 
fps = 25.5 #similar fps to source video

fourcc = cv2.VideoWriter_fourcc(*'MJPG') # FourCC is a 4-byte code used to specify the video codec.

video = cv2.VideoWriter('ar_result_modi.avi', fourcc, float(fps), (width, hieght))
 
for frame_count in range(len(ar_vd)):
    img = (ar_vd[frame_count]*255).astype(np.uint8)
    video.write(img)

video.release()