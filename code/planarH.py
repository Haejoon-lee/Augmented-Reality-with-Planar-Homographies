import numpy as np
import cv2
from PIL import Image

def computeH(x1, x2):
    #Q2.2.1
    #Compute the homography between two sets of points
    #Rewrite x1[i] ≡ H@x2[i] to Ai@h = 0 where h is a column vector reshaped from H and solve it 

    indx_rand = np.random.randint(len(x1), size=4) #Need 4 pair of points to solve h with 8 DoF 
    A = []
    for i in indx_rand:
        xcor2, ycor2 = x2[i]
        xcor1, ycor1 = x1[i]

        Ai = np.array([[xcor2, ycor2, 1, 0, 0, 0, -xcor1 * xcor2, -xcor1 * ycor2, -xcor1],
                      [0, 0, 0, xcor2, ycor2, 1, -ycor1 * xcor2, -ycor1 * ycor2, -ycor1]])
        A.append(Ai)
        
    A = np.vstack(A)

    U, S, V = np.linalg.svd(A)
    H2to1 = V.T[:, -1].reshape([3, 3])

    return H2to1

def computeH_norm(x1, x2):
    #Q2.2.2
    #Compute the centroid of the points
    mean1 = np.mean(x1, axis=0)
    mean2 = np.mean(x2, axis=0) 
    #Shift the origin of the points to the centroid
    x1 = x1 - mean1
    x2 = x2 - mean2

    #Normalize the points so that the largest distance from the origin is equal to sqrt(2)
    max1 = np.max(np.linalg.norm(x1, axis=1))
    max2 = np.max(np.linalg.norm(x2, axis=1))

    if max1 == 0:   #if max = 0 -> # of point in x is single and its value is 0
        c1 = 1
    else:
        c1 = 1 / (max1 / 2 ** 0.5)
    if max2 == 0:
        c2 = 1
    else:
        c2 = 1 / (max2 / 2 ** 0.5)

    x1 *= c1
    x2 *= c2

    #Similarity transform 1
    T1 = np.array([[c1, 0, -c1 * mean1[0]], [0, c1, -c1 * mean1[1]], [0, 0, 1]])

    #Similarity transform 2
    T2 = np.array([[c2, 0, -c2 * mean2[0]], [0, c2, -c2 * mean2[1]], [0, 0, 1]])

    #Compute homography
    h = computeH(x1, x2)

    #Denormalization
    H2to1 = np.linalg.inv(T1).dot(h).dot(T2)

    return H2to1

def computeH_ransac(locs1, locs2, opts):
    #Q2.2.3
    #Compute the best fitting homography given a list of matching points
    
    max_iters = opts.max_iters  # the number of iterations to run RANSAC for
    inlier_tol = opts.inlier_tol # the tolerance value for considering a point to be an inlier

    locs1_hc = np.hstack((locs1, np.ones((len(locs1), 1))))
    locs2_hc = np.hstack((locs2, np.ones((len(locs2), 1))))

    for i in range(0, max_iters):
        H = computeH_norm(locs1, locs2)

        locs2_warped = H @ locs2_hc.T
        locs2_warped /= locs2_warped[2, :]
        locs2_warped[2, :] = np.ones(locs2_warped.shape[1])

        error_sub = locs1_hc.T - locs2_warped
        error_squar = np.sum(error_sub ** 2, axis=0)
        
        inlier_idx = (error_squar <= inlier_tol**2).astype(int)

        if i == 0:
           bestH2to1 = H
           inliers_best = inlier_idx

        if np.sum(inlier_idx) > np.sum(inliers_best):
            bestH2to1 = H
            inliers_best = inlier_idx
            # print(inlier_idx)

    return bestH2to1, inliers_best

def compositeH(H2to1, template, img):
    
    #Create a composite image after warping the template image on top
    #of the image using the homography

    #Note that the homography we compute is from the image to the template;
    #x_template = H2to1*x_photo
    #For warping the template to the image, we need to invert it.

    img_cv_covoer_path = '../data/cv_cover.jpg'  
    img_cv_cover = Image.open(img_cv_covoer_path).convert("RGB")  
    img_cv_cover = np.array(img_cv_cover).astype(np.float32) / 255 #normalize


    # if img is blank -> put blank image to template with the position of warped cv_cover position
    if img.max() != 0: # if background
        img_warped = cv2.warpPerspective(img.swapaxes(0, 1), H2to1, (template.shape[0], template.shape[1])).swapaxes(0, 1)

        mask = np.zeros(img_warped.shape)
        mask = (img_warped != 0).astype(int)
        mask = np.logical_not(mask).astype(int)

        composite_img = img_warped + template * mask

    else:
        img_warped = cv2.warpPerspective(img_cv_cover.swapaxes(0, 1), H2to1, (template.shape[0], template.shape[1])).swapaxes(0, 1)
        mask = np.zeros(img_warped.shape)
        mask = (img_warped != 0).astype(int)
        mask = np.logical_not(mask).astype(int)

        composite_img = template * mask


    return composite_img