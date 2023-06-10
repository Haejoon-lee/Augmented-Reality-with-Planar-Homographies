import numpy as np
import cv2
import time
from PIL import Image
from tqdm import tqdm
import tqdm
# Import necessary functions

from loadVid import loadVid

#Import necessary functions
from matchPics import matchPics
from planarH import computeH_ransac, computeH, computeH_norm, compositeH # Import necessary functions
from helper import loadVid


start = time.time()
# for i in tqdm(range(110, 130)):
img_cv_covoer_path = '../data/cv_cover.jpg'
img_cv_cover = Image.open(img_cv_covoer_path).convert("RGB")  
img_cv_cover = np.array(img_cv_cover).astype(np.float32) / 255 #normalize

vd_book_path = "../data/book.mov"
vd_source_path = "../data/ar_source.mov"

vd_book = loadVid(vd_book_path)
vd_source = loadVid(vd_source_path)

H_list = []
ar_vd_even_list = []
length_vid = len(vd_source) - len(vd_source)%2 #Simply cut the last frame for even number of frames

# for i in tqdm(range(len(vd_source))):
for i in tqdm(range(int(length_vid/2))):
    frame_book = np.array(vd_book[2*i]).astype(np.float32) / 255
    frame_source = np.array(vd_source[2*i]).astype(np.float32) / 255 #normalize

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
    H_list.append(bestH2to1)
    
    img_composited = compositeH(bestH2to1, frame_book, frame_source_resized)
    
    ar_vd_even_list.append(img_composited)
    
    ar_vd_odd_list = []

for i in range(len(H_list)-1):
    frame_book = np.array(vd_book[2*i+1]).astype(np.float32) / 255
    frame_source = np.array(vd_source[2*i+1]).astype(np.float32) / 255 #normalize

    # cv_cover: 440x350x3, frame_source: 360x640x3 -> crop frame to get rid of black background and have the same ratio as cv_cover  
    # then resize the frame
    frame_source_crop = frame_source[44:314, 212:427, :]
    h_cv_cover, w_cv_cover = img_cv_cover.shape[0], img_cv_cover.shape[1]
    frame_source_resized = cv2.resize(frame_source_crop, (w_cv_cover, h_cv_cover))

    bestH2to1 = (H_list[i] + H_list[i+1])/2

    img_composited = compositeH(bestH2to1, frame_book, frame_source_resized)

    ar_vd_odd_list.append(img_composited)

ar_vd_list = []
for i in range(length_vid-1):
    if i%2 == 0:
        ar_vd_list.append(ar_vd_even_list[int(i/2)])
    else:
        ar_vd_list.append(ar_vd_odd_list[int((i-1)/2)])



# Q3.2
