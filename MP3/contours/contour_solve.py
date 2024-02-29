import numpy as np
from scipy import signal
import cv2

def compute_edges_dxdy(I, sigma = 4.3):
  """Returns the norm of dx and dy as the edge response function."""
  # I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  # I = I.astype(np.float32)/255.
  # radius = int(3 * sigma)
  # Gx, Gy = gaussian_derivative_kernels(sigma)
  # I_padded_x = np.pad(I, ((0, 0), (radius, radius)), mode='edge')
  # I_padded_y = np.pad(I, ((radius, radius), (0, 0)), mode='edge')
  # # Padding to minmize the artifacts
  # dx = signal.convolve2d(I_padded_x, Gx, mode='valid')
  # dy = signal.convolve2d(I_padded_y, Gy, mode='valid')
  # mag = np.sqrt(dx**2 + dy**2)
  # mag = mag / 1.5
  # mag = mag * 255.
  # mag = np.clip(mag, 0, 255)
  # mag = mag.astype(np.uint8)
  I = I.astype(np.float32) / 255.  # Normalize the image

  radius = int(3 * sigma)
  Gx, Gy = gaussian_derivative_kernels(sigma)

  # 初始化边缘强度矩阵
  mag_combined = np.zeros(I.shape[:2], dtype=np.float32)
  dx_combined = np.zeros_like(mag_combined)
  dy_combined = np.zeros_like(mag_combined)
  for channel in range(3):  # 分别处理R, G, B通道
    I_channel = I[:, :, channel]

    # 应用padding
    I_padded_x = np.pad(I_channel, ((0, 0), (radius, radius)), mode='edge')
    I_padded_y = np.pad(I_channel, ((radius, radius), (0, 0)), mode='edge')

    # 计算每个通道的梯度
    dx = signal.convolve2d(I_padded_x, Gx, mode='valid')
    dy = signal.convolve2d(I_padded_y, Gy, mode='valid')

    # 计算梯度幅度
    mag = np.sqrt(dx ** 2 + dy ** 2)

    update_mask = mag > mag_combined
    mag_combined[update_mask] = mag[update_mask]
    dx_combined[update_mask] = dx[update_mask]
    dy_combined[update_mask] = dy[update_mask]

  # 标准化最终的梯度幅度
  mag_combined = mag_combined / 1.5
  mag_combined = mag_combined * 255.
  mag_combined = np.clip(mag_combined, 0, 255)
  mag_combined = mag_combined.astype(np.uint8)
  mag = mag_combined
  # Implement NMS
  M, N = mag.shape
  result = np.zeros((M, N), dtype=np.float32)
  angle = np.arctan2(dy, dx)

  for i in range(1, M - 1):
    for j in range(1, N - 1):
      ang = angle[i, j]
      # Calculate the direction vector (dx, dy) for the gradient direction
      dx_dir = np.cos(ang)
      dy_dir = np.sin(ang)

      # Calculate gradient magnitude at the two neighboring pixels in the gradient direction
      mag1 = bilinear_interpolation(mag, j + dx_dir, i + dy_dir)
      mag2 = bilinear_interpolation(mag, j - dx_dir, i - dy_dir)

      # Suppress the pixel if it's not greater than both neighbors
      if mag[i, j] >= mag1 and mag[i, j] >= mag2:
        result[i, j] = mag[i, j]
      else:
        result[i, j] = 0
  return result


def gaussian_derivative_kernels(sigma):
  """Generate Gaussian derivative kernels in x and y directions based on sigma."""
  # Define the range for the kernel
  radius = int(3 * sigma)
  x = np.arange(-radius, radius + 1)

  # Compute the 1D Gaussian derivative kernel
  gx = -x / (sigma ** 2) * np.exp(-x ** 2 / (2 * sigma ** 2))
  gx = gx.reshape(1, -1)  # Reshape gx to a 2D array (row vector)
  gy = gx.T  # Transpose to get the y derivative

  return gx, gy


def bilinear_interpolation(image, x, y):
  h, w = image.shape

  # 确保坐标在图像范围内
  x = np.clip(x, 0, w - 1)
  y = np.clip(y, 0, h - 1)

  x1, y1 = int(x), int(y)
  x2, y2 = min(x1 + 1, w - 1), min(y1 + 1, h - 1)

  # 计算插值权重
  wa = (x2 - x) * (y2 - y)
  wb = (x - x1) * (y2 - y)
  wc = (x2 - x) * (y - y1)
  wd = (x - x1) * (y - y1)

  # 直接从图像中获取四个邻近点的值
  A, B, C, D = image[y1, x1], image[y1, x2], image[y2, x1], image[y2, x2]

  # 计算插值结果
  return wa * A + wb * B + wc * C + wd * D
