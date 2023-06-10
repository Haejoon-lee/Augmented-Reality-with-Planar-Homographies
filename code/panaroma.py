import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from opts import get_opts
from planarH import computeH_ransac, computeH, computeH_norm, compositeH # Import necessary functions
from matchPics import matchPics


# Q4, construct a panorama image
opts = get_opts()

img_pano_left = Image.open('../data/my_pano_left.jpeg').convert("RGB")  
img_pano_left = np.array(img_pano_left).astype(np.float32) / 255 #normalize

img_pano_right = Image.open('../data/my_pano_right.jpeg').convert("RGB")  
img_pano_right = np.array(img_pano_right).astype(np.float32) / 255 #normalize

matches, locs1, locs2 = matchPics(img_pano_left, img_pano_right, opts)

locs1_matched = locs1[matches[:, 0]]
locs2_matched = locs2[matches[:, 1]]

bestH2to1, inliers = computeH_ransac(locs1_matched, locs2_matched, opts)

width = img_pano_left.shape[1] + img_pano_right.shape[1]
height = img_pano_left.shape[0] + img_pano_right.shape[0]

img_warped = cv2.warpPerspective(img_pano_right.swapaxes(0, 1), bestH2to1, (height, width)).swapaxes(0, 1)
img_warped[0:img_pano_left.shape[0], 0:img_pano_left.shape[1]] = img_pano_left

plt.imshow(img_warped[:1250,:2000])
plt.show()

