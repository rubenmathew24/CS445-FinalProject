import numpy as np
import cv2
from numpy.linalg import svd


def score_projection(pt1, pt2):
    '''
    Score corresponding to the number of inliers for RANSAC
    Input: pt1 and pt2 are 2xN arrays of N points such that pt1[:, i] and pt2[:,i] should be close in Euclidean distance if they are inliers
    Outputs: score (scalar count of inliers) and inliers (1xN logical array)
    '''

    # TO DO

    dist = np.linalg.norm(pt1 - pt2, axis=0)
    perm = np.argsort(dist)

    inliers = np.zeros(len(perm), dtype=bool)

    # for i in range(len(perm) - 1):
    #   if dist[perm[i]] / dist[perm[i+1]] < 1:
    #     inliers[i] = True

    inliers = dist < 1

    score = np.sum(inliers)

    return score, inliers


def auto_homography(Ia, Ib, homography_func=None, normalization_func=None):
    '''
    Computes a homography that maps points from Ia to Ib

    Input: Ia and Ib are images
    Output: H is the homography

    '''
    if Ia.dtype == 'float32' and Ib.dtype == 'float32':
        Ia = (Ia*255).astype(np.uint8)
        Ib = (Ib*255).astype(np.uint8)

    Ia_gray = cv2.cvtColor(Ia, cv2.COLOR_BGR2GRAY)
    Ib_gray = cv2.cvtColor(Ib, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()  # updated for OpenCV 4.0

    # find the keypoints and descriptors with SIFT
    kp_a, des_a = sift.detectAndCompute(Ia_gray, None)
    kp_b, des_b = sift.detectAndCompute(Ib_gray, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_a, des_b, k=2)

    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    numMatches = int(len(good))

    matches = good

    # Xa and Xb are 3xN matrices that contain homogeneous coordinates for the N
    # matching points for each image
    Xa = np.ones((3, numMatches))
    Xb = np.ones((3, numMatches))

    for idx, match_i in enumerate(matches):
        Xa[:, idx][0:2] = kp_a[match_i.queryIdx].pt
        Xb[:, idx][0:2] = kp_b[match_i.trainIdx].pt

    # RANSAC
    niter = 1000
    best_score = 0
    n_to_sample = 4  # Put the correct number of points here ##JPCHECK

    for t in range(niter):
        # estimate homography
        subset = np.random.choice(numMatches, n_to_sample, replace=False)
        pts1 = Xa[:, subset]
        pts2 = Xb[:, subset]

        # edit helper code below (computeHomography)
        H_t = homography_func(pts1, pts2, normalization_func)

        # score homography
        # project points from first image to second using H
        Xb_ = np.dot(H_t, Xa)

        score_t, inliers_t = score_projection(
            Xb[:2, :]/Xb[2, :], Xb_[:2, :]/Xb_[2, :])

        if score_t > best_score:
            best_score = score_t
            H = H_t
            in_idx = inliers_t

    # Optionally, you may want to re-estimate H based on inliers

    return H


def computeHomography(pts1, pts2, normalization_func=None):
    '''
    Compute homography that maps from pts1 to pts2 using SVD. Normalization is optional.

    Input: pts1 and pts2 are 3xN matrices for N points in homogeneous
    coordinates. 

    Output: H is a 3x3 matrix, such that pts2~=H*pts1
    '''
    # TO DO

    A = np.zeros((2*pts1.shape[1], 9))

    u = pts1[0, :]
    v = pts1[1, :]
    u_ = pts2[0, :]
    v_ = pts2[1, :]

    for i in range(len(u)):
        A[2*i] = [-u[i], -v[i], -1,     0,     0,
                  0, u[i]*u_[i], v[i]*u_[i], u_[i]]
        A[2*i + 1] = [0,     0,      0, -u[i], -
                      v[i], -1, u[i]*v_[i], v[i]*v_[i], v_[i]]

    U, S, V = svd(A)

    H = V[-1].reshape(3, 3)

    return H
