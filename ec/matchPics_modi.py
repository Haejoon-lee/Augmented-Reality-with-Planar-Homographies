import numpy as np
import cv2
import skimage.color
from scipy import ndimage

from helper import briefMatch
from helper import computeBrief
from helper import corner_detection

# Q2.1.4
def matchPics_modi(I1, I2, angle_rot, opts):
        """
        Match features across images

        Input
        -----
        I1, I2: Source images
        opts: Command line args

        Returns
        -------
        matches: List of indices of matched features across I1, I2 [p x 2]
        locs1, locs2: Pixel coordinates of matches [N x 2]
        """
        
        ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
        # ratio = 0.7  #'ratio for BRIEF feature descriptor'
        sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        
        # TODO: Convert Images to GrayScale
        I1_gray = cv2.cvtColor(I1, cv2.COLOR_BGR2GRAY)
        I2_gray = cv2.cvtColor(I2, cv2.COLOR_BGR2GRAY)
        I2_gray = ndimage.rotate(I2_gray, -angle_rot, reshape = False) # rotate I2 to be aligned with I1.

        # TODO: Detect Features in Both Images
        locs1 = corner_detection(I1_gray, sigma)
        locs2 = corner_detection(I2_gray, sigma)

        # TODO: Obtain descriptors for the computed feature locations
        desc1, locs1 = computeBrief(I1_gray, locs1)
        desc2, locs2 = computeBrief(I2_gray, locs2)

        # TODO: Match features using the descriptors
        matches = briefMatch(desc1, desc2, ratio)
        
        R = np.array([[np.cos(angle_rot*(np.pi/180)), -np.sin(angle_rot*(np.pi/180))], 
                  [np.sin(angle_rot*(np.pi/180)), np.cos(angle_rot*(np.pi/180))]])
        
        #move the center coordinate of locs2 to (0, 0)
        locs2[:,0] = locs2[:,0] - I2_gray.shape[0]/2
        locs2[:,1] = locs2[:,1] - I2_gray.shape[1]/2
        
        #rotate locs according to the original angle
        locs2 = (R @ locs2.T).T
        
        #relocate the center coordinate of locs2
        locs2[:,0] = locs2[:,0] + I2_gray.shape[0]/2
        locs2[:,1] = locs2[:,1] + I2_gray.shape[1]/2
#         locs2 = R @ locs2.T
        
        return matches, locs1, locs2

# def matchPics(I1, I2, opts):
#         """
#         Match features across images

#         Input
#         -----
#         I1, I2: Source images
#         opts: Command line args

#         Returns
#         -------
#         matches: List of indices of matched features across I1, I2 [p x 2]
#         locs1, locs2: Pixel coordinates of matches [N x 2]
#         """
        
#         ratio = opts.ratio  #'ratio for BRIEF feature descriptor'
#         sigma = opts.sigma  #'threshold for corner detection using FAST feature detector'
        

#         # TODO: Convert Images to GrayScale
        
        
#         # TODO: Detect Features in Both Images
        
        
#         # TODO: Obtain descriptors for the computed feature locations
        

#         # TODO: Match features using the descriptors
        

#         return matches, locs1, locs2

