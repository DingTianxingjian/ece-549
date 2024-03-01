import numpy as np
import cv2
from scipy import signal


def blend(im1, im2, mask, levels = 6):
    h, w, dept = im1.shape
    # Resize the input images and mask to the desired format
    im1 = cv2.resize(im1, (512, 512), interpolation=cv2.INTER_AREA)
    im2 = cv2.resize(im2, (512, 512), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_AREA)

    # Function to generate Gaussian pyramid
    def generate_gaussian_pyramid(img, levels):
        G = img.copy()
        gp = [G]
        for _ in range(levels):
            G = cv2.pyrDown(G)
            gp.append(G)
        return gp

    # Function to build a Laplacian pyramid
    def generate_laplacian_pyramid(gp):
        lp = [gp[-1]]
        for i in range(len(gp) - 1, 0, -1):
            GE = cv2.pyrUp(gp[i])
            L = cv2.subtract(gp[i - 1], GE[:gp[i - 1].shape[0], :gp[i - 1].shape[1]])  # Match size
            lp.append(L)
        return lp

    # Function to reconstruct image from Laplacian pyramid
    def reconstruct_from_laplacian_pyramid(lp):
        img = lp[0]
        for i in range(1, len(lp)):
            img = cv2.pyrUp(img)
            img = cv2.add(img, lp[i][:img.shape[0], :img.shape[1]])  # Match size
        return img

    # Generate Gaussian pyramids
    gpA = generate_gaussian_pyramid(im1, levels)
    gpB = generate_gaussian_pyramid(im2, levels)

    # Generate Laplacian pyramids
    lpA = generate_laplacian_pyramid(gpA)
    lpB = generate_laplacian_pyramid(gpB)

    # Blend the Laplacian pyramids
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols = la.shape[:2]
        ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
        LS.append(ls)

    # Reconstruct the blended image from the Laplacian pyramid
    output = reconstruct_from_laplacian_pyramid(LS)
    output = cv2.resize(output, (h, w), interpolation=cv2.INTER_AREA)

    return output
