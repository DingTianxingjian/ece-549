import numpy as np
import scipy
import cv2
from scipy.ndimage.filters import maximum_filter
def compute_corners(I, alpha=0.042, sigma=0.8, threshold=0.01, nms_size=3):
    # Input:
    #   I: input image, H x W x 3 BGR image
    # Output:
    #   response: H x W response map in uint8 format
    #   corners: H x W map in uint8 format _after_ non-max suppression. Each
    #   pixel stores the score for being a corner. Non-max suppressed pixels
    #   should have a low / zero-score.

    # Convert to grayscale
    gray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Compute gradients
    Ix, Iy = compute_simple_gradients(gray)

    # Compute products of derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Compute the sums of the products of derivatives at each pixel
    h, w = gray.shape
    response = np.zeros((h, w), dtype=np.float32)
    corners = np.zeros((h, w), dtype=np.uint8)

    # Apply Gaussian filter to sum of products
    Ixx = scipy.ndimage.gaussian_filter(Ixx, sigma=sigma)
    Iyy = scipy.ndimage.gaussian_filter(Iyy, sigma=sigma)
    Ixy = scipy.ndimage.gaussian_filter(Ixy, sigma=sigma)

    # Compute the response of the detector at each pixel
    k = alpha
    detM = Ixx * Iyy - Ixy ** 2
    traceM = Ixx + Iyy
    response = detM - k * traceM ** 2

    # Perform non-maximum suppression
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    corners = cv2.dilate(response, kernel=kernel)
    corners = np.where((response == corners) & (response > 0.01 * response.max()), 255, 0)
    corners = corners.astype(np.uint8)

    neighborhood_size = 7
    data_max = maximum_filter(response, neighborhood_size)
    max_mask = (response == data_max)
    response *= max_mask
    corners = np.where((response > 0) & (response > 0.01 * response.max()), 255, 0)
    corners = corners.astype(np.uint8)

    # Normalize the response map for visualization
    response_vis = cv2.normalize(response, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    response_vis = response_vis.astype(np.uint8)

    return response_vis, corners



def compute_simple_gradients(image):
    gray = image
    # 初始化梯度图
    h, w = gray.shape
    Ix = np.zeros((h, w), dtype=np.float32)
    Iy = np.zeros((h, w), dtype=np.float32)

    # 计算水平梯度（Ix）
    Ix[:, 1:-1] = gray[:, 2:] - gray[:, :-2]

    # 计算垂直梯度（Iy）
    Iy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    return Ix, Iy