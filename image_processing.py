import cv2 as cv
import numpy as np
from math import floor
import scipy

# Algorithm procedure definitions

## Function for automatic clipping of background

def safe_substract(arr: np.array, scalar):

  # upcast the array
  arr = arr.astype(np.int32)

  arr = np.clip(arr - scalar, 0, 65535).astype(np.uint16)

  return arr

def clip_bg_global(roi, grad_thresh=0.5, VERBOSE=True):
  """
  :param grad_thresh: this determines the aggressiveness of thresholding. the smaller the more background will be cut out
  """

  # calculate histogram
  bins = 256 if roi.dtype == np.uint8 else 65536
  if bins == 256:
    print('WARNING: 8-bit type input image!')

  hist_roi, _ = np.histogram(roi, bins=bins, range=(0, bins))

  # smooth out histogram
  # 101 in 16 bit is still only half a color step in 8 bit images
  hist_roi_filtered = scipy.signal.medfilt(hist_roi, 255)

  # find indices of local maxima in the histogram

  #try to find peaks with the highes prominence possible
  x_local_maxima_hist_roi = []
  prom = 30
  while len(x_local_maxima_hist_roi) == 0 and prom >= 0:
    x_local_maxima_hist_roi, _ = scipy.signal.find_peaks(hist_roi_filtered[:65000], prominence=prom)
    prom -= 1

  if len(x_local_maxima_hist_roi) == 0:
    if VERBOSE:
      print('WARNING: No maximum in histogram!')
    return 0

  # print found local maxima
  if VERBOSE:
    print(
        f'(Found {len(x_local_maxima_hist_roi)} local maxima '
        f'at {x_local_maxima_hist_roi} with values {hist_roi_filtered[x_local_maxima_hist_roi]}'
    )

  # choose the rightmost peak in the histogram, i.e. with highest index
  x_background_peak = np.max(x_local_maxima_hist_roi)




  # calculate gradient of histogram
  grad_roi = np.gradient(hist_roi_filtered)

  # filter -> this is only to be used for thresholding
  grad_roi_filtered = scipy.signal.medfilt(grad_roi, 101)

  # find the negative peak in the gradient
  # that is the the point of interest closest to the threshold
  # this peak is going to be more to the right than the histogram peak
  x_grad_minimum = np.argmin(grad_roi[x_background_peak:]) + x_background_peak
  y_grad_minimum = grad_roi[x_grad_minimum]

  # calculate threshold vor the gradient based on the negative peak
  # rather include more background that is later filtered out at gaussian adaptive thresholding
  y_thresh = y_grad_minimum * grad_thresh
  x_thresh = np.argwhere(grad_roi_filtered[x_grad_minimum:] > y_thresh)

  if len(x_thresh) == 0:
    x_thresh = 500
  else:
    x_thresh = floor(x_thresh[0] + x_grad_minimum)

  return x_thresh

def clip_convert_8b(channel, grad_thresh=0.5, VERBOSE=False):
  """
  Applies automatic thresholding as defined in clip_bg_global and
  converts to 8 bit
  :param channel: Single channel 16-bit image. I.e. 2D array
  :param grad_thresh: aggressiveness of thresholding. (0; 1]
  """

  if not 0 < grad_thresh <= 1:
    raise ValueError('grad_thresh must be in (0; 1]')

  # Filter out background through thresholding
  x_thresh = clip_bg_global(channel, grad_thresh, VERBOSE)

  # apply threshold to the image
  channel_thresh = channel.copy()
  channel_thresh[channel_thresh < x_thresh] = 0

  # scale to 8 bit
  channel_thresh = (safe_substract(channel_thresh, x_thresh) / (channel_thresh.max() - x_thresh) * 255)
  channel_thresh = channel_thresh.astype(np.uint8)

  return channel_thresh

## Hole finding

def find_holes(img, HOLE_AGGRESSIVENESS = 2, VERBOSE=False):
  """
  Finds holes in the image by analyzing histogram peaks.
  :param img: opencv image channel (i.e. 2 D). Preferably 16 bit
  :param HOLE_AGGRESSIVENESS: 1 - 10: a high value means more low background tissue might be cut
  :return: A binary image (always 8 bit) with the holes as 256 and the rest as 0.
  """

  # calculate histogram
  bins = 256 if img.dtype == np.uint8 else 65536
  if bins == 256:
    print('WARNING: 8-bit type input image!')

  hist_roi, _ = np.histogram(img, bins=bins, range=(0, bins))

  # smooth out histogram
  # 101 in 16 bit is still only half a color step in 8 bit images
  if bins > 256:
    hist_roi_filtered = scipy.signal.medfilt(hist_roi, 255)

  # find indices of local maxima in the histogram

  #try to find peaks with the highest prominence possible
  x_local_maxima_hist_roi = []
  prom = 30
  while len(x_local_maxima_hist_roi) == 0 and prom >= 0:
    x_local_maxima_hist_roi, _ = scipy.signal.find_peaks(hist_roi_filtered[:(bins//6)], prominence=prom)
    prom -= 1

  if len(x_local_maxima_hist_roi) == 0 and VERBOSE:
    print('WARNING: No maximum in histogram!')

  # sum up peaks with in very close vicinity
  min_peak_distance = bins
  max_peak_distance = 0
  for i in range(len(x_local_maxima_hist_roi) - 1):

    dist = x_local_maxima_hist_roi[i+1] - x_local_maxima_hist_roi[i]

    if dist < min_peak_distance:
      min_peak_distance = dist
    if dist > max_peak_distance:
      max_peak_distance = dist

  if VERBOSE:
    print('Peaks: ', x_local_maxima_hist_roi)
    print(f'Min peak distance: {min_peak_distance}')
    print(f'Max peak distance: {max_peak_distance}')

  if max_peak_distance < bins // 256:
    if VERBOSE:
      print('No hole peak found!')
    return np.zeros_like(img).astype(np.uint8)

  # find the hole peak
  hole_peak_index = -1
  for i in range(len(x_local_maxima_hist_roi) - 1):
    if x_local_maxima_hist_roi[i+1] - x_local_maxima_hist_roi[i] == max_peak_distance:
      hole_peak_index = i
      break

  if hole_peak_index == -1:
    if VERBOSE:
      print('WARNING: No hole peak found!')
    return np.zeros_like(img).astype(np.uint8)

  hole_peak_value = x_local_maxima_hist_roi[hole_peak_index]
  hole_peak_value += bins // 256 * HOLE_AGGRESSIVENESS

  # apply threshold to the image
  channel_thresh = img.copy()
  channel_thresh[channel_thresh < hole_peak_value] = 255
  channel_thresh[channel_thresh >= hole_peak_value] = 0

  if VERBOSE:
    print(f'Hole area (percentage): {np.count_nonzero(channel_thresh) / channel_thresh.size:.4f}')

  return channel_thresh.astype(np.uint8)

## Function for CD31


def find_cd31(cd31_channel, scale_factor, bsize=0, thresh_c=-2):
  """
  :return: all contours. not filtered by size anymore
  """
  # TODO make adjustable to scale_factor
  num_close_iterations = 1
  close_kernel_size = 9

  num_open_iterations = 1
  open_kernel_size = 6

  # parameters to tweak
  blur_kernel_size = 3
  blur_kernel_sigma = 0.0

  if bsize == 0:
    # 20 µm is the average distance between two capillaries
    thresh_block_size = floor(20 / scale_factor)
  else:
    thresh_block_size = bsize

  # block size must be uneven
  if thresh_block_size % 2 == 0:
      thresh_block_size += 1

  # blur
  blurred = cv.GaussianBlur(cd31_channel, (blur_kernel_size,) * 2, blur_kernel_sigma)

  # Adaptive Threshold
  threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,
                                    thresh_block_size, thresh_c)

  # Closing
  close_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (close_kernel_size,) * 2)
  closed = cv.morphologyEx(threshold, cv.MORPH_CLOSE, close_kernel, iterations=num_close_iterations)

  # Opening horizontally and vertically
  open_kernels = (
      cv.getStructuringElement(cv.MORPH_ELLIPSE, (open_kernel_size, 2)),
      cv.getStructuringElement(cv.MORPH_ELLIPSE, (2, open_kernel_size,)),
  )
  opened_h = cv.morphologyEx(closed, cv.MORPH_OPEN, open_kernels[0], iterations=num_open_iterations)
  opened = cv.morphologyEx(opened_h, cv.MORPH_OPEN, open_kernels[1], iterations=num_open_iterations)


  # Contours
  contours, _ = cv.findContours(opened, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

  return contours

## Function for NG2

def find_ng2(ng2_channel, cd31_contours, scale_factor):

  coverages = []

  # Iterate over contours
  for i, cnt in enumerate(cd31_contours):

    # compute bounding rectangle
    bounding_rect = cv.boundingRect(cnt)

    # convert bounding rectangle coordinates and size to upper left and lower right corner
    # add margins of 10 px
    margin = 10
    upper_left = (bounding_rect[0] - margin, bounding_rect[1] - margin)
    lower_right = (bounding_rect[0] + bounding_rect[2] + margin, bounding_rect[1] + bounding_rect[3] + margin)

    # clip upper left to zero
    upper_left = (max(upper_left[0], 0), max(upper_left[1], 0))

    # capillary and surroundings
    # note: opencv will automatically clip. Will not be padded with zeros
    cap = ng2_channel[upper_left[1]:lower_right[1], upper_left[0]:lower_right[0]]

    # denoise
    blurred = cv.GaussianBlur(cap, (3, ) * 2, 0.0)

    # threshold
    bsize = floor(min(cap.shape) / 2.0)
    bsize = bsize + 1 if bsize % 2 == 0 else bsize
    threshold = cv.adaptiveThreshold(blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, bsize, 0.0)

    # Create zero-valued red and blue channels
    zeros = np.zeros_like(cap)

    # Apply capillary mask to ng2 channel
    cd31_mask = cv.fillPoly(zeros, [cnt], 255, offset=(-1 * upper_left[0], -1 * upper_left[1]))
    ng2_masked = cv.bitwise_and(threshold, threshold, mask=cd31_mask)

    # calculate percentage covered
    non_zero_cap = np.count_nonzero(cd31_mask)
    non_zero_per = np.count_nonzero(ng2_masked)
    percentage = non_zero_per / non_zero_cap * 100

    coverages.append(percentage)

  return coverages


