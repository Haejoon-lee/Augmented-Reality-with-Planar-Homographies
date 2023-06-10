import numpy as np
import cv2
from matchPics import matchPics
from opts import get_opts

from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm

#Q2.1.6 

def rotTest(opts):    
    #Read the image and convert to grayscale, if necessary

    img_path = '../data/cv_cover.jpg'

    img = Image.open(img_path).convert("RGB")  
    img = np.array(img).astype(np.float32) / 255 #normalize

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    counts_matches = []

    for i in tqdm(range(36)):
        # Rotate Image
        #Compute features, descriptors and Match features
        rot_ang = i * 10
        img_rot = ndimage.rotate(img, rot_ang, reshape=False)
        matches, locs1, locs2 = matchPics(img, img_rot, opts) #Should put opt!
        counts_matches.append(matches.shape[0])

    plt.bar(range(0, 360, 10), counts_matches, width=10)
    plt.show()



if __name__ == "__main__":

    opts = get_opts()
    rotTest(opts)
