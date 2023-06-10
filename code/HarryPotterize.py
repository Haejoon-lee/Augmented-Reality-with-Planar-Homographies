import numpy as np
import cv2
import skimage.io 
import skimage.color
from PIL import Image
import matplotlib.pyplot as plt

from opts import get_opts
from matchPics import matchPics
from planarH import computeH_ransac, computeH, computeH_norm, compositeH

# Import necessary functions

# Q2.2.4

def warpImage(opts):
    """
    1. Reads cv_cover.jpg, cv_desk.png, and hp_cover.jpg
    2. Computes a homography automatically using matchPics and computeH_ransac
    3. Uses the computed homography to warp hp_cover.jpg to the dimensions of the cv_desk.png image 
    using the OpenCV function cv2.warpPerspective function
    4. Now compose this warped image with the desk image
    """

    img_cv_covoer_path = '../data/cv_cover.jpg'
    img_cv_cover = Image.open(img_cv_covoer_path).convert("RGB")  
    img_cv_cover = np.array(img_cv_cover).astype(np.float32) / 255 #normalize

    img_cv_desk_path = '../data/cv_desk.png'
    img_cv_desk = Image.open(img_cv_desk_path).convert("RGB")
    img_cv_desk = np.array(img_cv_desk).astype(np.float32)/ 255 #normalize

    img_hp_cover_path = '../data/hp_cover.jpg'
    img_hp_cover = Image.open(img_hp_cover_path).convert("RGB")
    img_hp_cover = np.array(img_hp_cover).astype(np.float32)/ 255 #normalize

    # cv_cover: 440x350x3, hp_cover: 295x200x3
    # resize hp_cover to be matched with cv_cover
    h_cv_cover, w_cv_cover = img_cv_cover.shape[0], img_cv_cover.shape[1]
    img_hp_cover_resized = cv2.resize(img_hp_cover, (w_cv_cover, h_cv_cover))
    # rotated_image = ndimage.rotate(cv_desk, 40, reshape=False)

    matches, locs1, locs2 = matchPics(img_cv_desk, img_cv_cover, opts)

    locs1_matched = locs1[matches[:, 0]]
    locs2_matched = locs2[matches[:, 1]]

    bestH2to1, inliers = computeH_ransac(locs1_matched, locs2_matched, opts)

    img_hp_warped = cv2.warpPerspective(img_hp_cover_resized.swapaxes(0, 1), bestH2to1, (img_cv_desk.shape[0], img_cv_desk.shape[1])).swapaxes(0, 1)
    plt.imshow(img_hp_warped)
    plt.show()

    img_composited = compositeH(bestH2to1, img_cv_desk, img_hp_cover_resized)
    plt.imshow(img_composited)
    plt.show()


if __name__ == "__main__":

    opts = get_opts()
    warpImage(opts)


