# -*- coding: utf-8 -*-
"""EEP 596 HW2
"""

import numpy as np
import cv2
import scipy
import matplotlib.pyplot as plt

class ComputerVisionAssignment():
  def __init__(self) -> None:
    self.ant_img = cv2.imread('ant_outline.png')
    self.cat_eye = cv2.imread('cat_eye.jpg', cv2.IMREAD_GRAYSCALE)

  def floodfill(self, seed = (0, 0)):

    # Define the fill color (e.g., bright green)
    fill_color = (0,0,255)
    # Create a copy of the input image to keep the original image unchanged
    output_image = self.ant_img.copy()
    h,w = output_image.shape[:2]
    filled = np.zeros((h,w), dtype = bool)
    seed_x, seed_y = seed
    if not (0 <= seed_x < w and 0 <= seed_y < h): return ValueError('not valid')
    target_color = self.ant_img[seed_y,seed_x].copy()
    # Define a stack for floodfill
    stack = [seed]
    while stack:
      x,y = stack.pop()
      if x < 0 or x >= w or y < 0 or y >= h: 
        continue
      if filled[y,x]:
        continue
      filled[y,x] = True
      if not np.array_equal(self.ant_img[y,x],target_color): continue
      output_image[y,x] = fill_color
      stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1),(x - 1, y - 1), (x + 1, y + 1), (x + 1, y - 1), (x - 1, y + 1)])
    
    cv2.imwrite('floodfille.jpg', output_image)
    return output_image

  def gaussian_blur(self):
    """
    Apply Gaussian blur to the image iteratively.
    """
    kernel = 0.25 * np.array([1,2,1])
    image = self.cat_eye.astype(np.float64)
    h, w = image.shape
    self.blurred_images = []
    for i in range(5):
        # Apply convolution
        pad = np.pad(image, ((1,1), (0,0)), mode = "constant")
        vertical = np.zeros((h,w), dtype = np.float64)
        for y in range(h):
          for x in range(w):
            vertical[y,x] = (kernel[0] * pad[y,x] + kernel[1] * pad[y+1,x] + kernel[2] * pad[y+2,x])

        pad = np.pad(vertical, ((0,0), (1,1)), mode = "constant")
        output = np.zeros((h,w), dtype = np.float64)

        for y in range(h):
          for x in range(w):
            output[y,x] = (kernel[0] * pad[y,x] + kernel[1] * pad[y,x+1] + kernel[2] * pad[y,x+2])

        
        image = output
        # Store the blurred image
        self.blurred_images.append(image)
        
        #cv2.imwrite(f'gaussain blur {i}.jpg', image)
    return self.blurred_images

  def gaussian_derivative_vertical(self):
    # Define kernels
    kernel = 0.25 * np.array([1,2,1])
    sobel_kernel = 0.5 * np.array([1,0,-1])
    # Store images
    self.vDerive_images = []
    for i in range(5):
      # Apply horizontal and vertical convolution
      src = self.blurred_images[i].astype(np.float64)
      h,w = src.shape
      pad = np.zeros((h+2,w+2), dtype = np.float64)
      pad[1:-1,1:-1] = src

      image = np.zeros((h,w), dtype = np.uint8)

      for y in range(h):
        for x in range(w):
          top = (kernel[0] * pad[y,x] + kernel[1] * pad[y,x+1] + kernel[2] * pad[y,x+2])
          bottom = (kernel[0] * pad[y+2,x] + kernel[1] * pad[y+2,x+1] + kernel[2] * pad[y+2,x+2])

          temp = sobel_kernel[0] * top + sobel_kernel[2] * bottom
          output = 2 * temp + 127
          if output < 0: output = 0
          elif output > 255:
            output = 255
    
          image[y,x] = output
      
      self.vDerive_images.append(image)
      #cv2.imwrite(f'vertical {i}.jpg', image)
    return self.vDerive_images

  def gaussian_derivative_horizontal(self):
    #Define kernels
    kernel = 0.25 * np.array([1,2,1])
    sobel_kernel = 0.5 * np.array([1,0,-1])
    # Store images after computing horizontal derivative
    self.hDerive_images = []

    for i in range(5):

      # Apply horizontal and vertical convolution
      src = self.blurred_images[i].astype(np.float64)
      h,w = src.shape
      pad = np.zeros((h+2,w+2), dtype = np.float64)
      pad[1:-1,1:-1] = src
      image = np.zeros((h,w), dtype = np.uint8)

      for y in range(h):
        for x in range(w):
          left = (kernel[0] * pad[y,x] + kernel[1] * pad[y+1,x] + kernel[2] * pad[y+2,x])
          right = (kernel[0] * pad[y,x+2] + kernel[1] * pad[y+1,x+2] + kernel[2] * pad[y+2,x+2])

          temp = sobel_kernel[0] * left + sobel_kernel[2] * right
          output = temp * 2  + 127
          if output < 0: output = 0
          elif output > 255: output = 255

          image[y,x] = output

      self.hDerive_images.append(image)
      #cv2.imwrite(f'horizontal {i}.jpg', image)
    return self.hDerive_images

  def gradient_magnitute(self):
    # Store the computed gradient magnitute
    self.gdMagnitute_images =[]
    for i, (vimg, himg) in enumerate(zip(self.vDerive_images, self.hDerive_images)):
      gx = (himg.astype(np.float32) - 127) / 2
      gy = (vimg.astype(np.float32) - 127) / 2
      magnitude = 4 * (np.abs(gx) + np.abs(gy))
      image = np.clip(magnitude,0,255).astype(np.uint8)
      self.gdMagnitute_images.append(image)
      #cv2.imwrite(f'gradient {i}.jpg', image)
    return self.gdMagnitute_images
    
  def scipy_convolve(self):
    # Define the 2D smoothing kernel
    kernel = np.outer(0.5 * np.array([1, 0, -1]),0.25 * np.array([1, 2, 1]))
    # Store outputs
    self.scipy_smooth = []

    for i in range(5):
      # Perform convolution
      derivative = scipy.signal.convolve2d(self.blurred_images[i].astype(np.float64), kernel, mode='same', boundary = 'fill', fillvalue = 0)
      image= np.clip(2 * derivative + 127, 0, 255).astype(np.uint8)
      self.scipy_smooth.append(image)
      #cv2.imwrite(f'scipy smooth {i}.jpg', image)
    return self.scipy_smooth

  def box_filter(self, num_repetitions):
    # Define box filter
    box_filter = [1, 1, 1]
    out = [1, 1, 1]

    for _ in range(num_repetitions):
      # Perform 1D conlve
      out = np.convolve(out,box_filter,mode = 'full')

    return out

if __name__ == "__main__":
    ass = ComputerVisionAssignment()
    # # Task 1 floodfill
    #floodfill_img = ass.floodfill(seed=(100, 100))

    # Task 2 Convolution for Gaussian smoothing.
    blurred_imgs = ass.gaussian_blur()

    # Task 3 Convolution for differentiation along the vertical direction
    vertical_derivative = ass.gaussian_derivative_vertical()

    # Task 4 Differentiation along another direction along the horizontal direction
    horizontal_derivative = ass.gaussian_derivative_horizontal()

    # Task 5 Gradient magnitude.
    Gradient_magnitude = ass.gradient_magnitute()

    # Task 6 Built-in convolution
    scipy_convolve = ass.scipy_convolve()

    # Task 7 Repeated box filtering
    box_filter = ass.box_filter(5)
    
