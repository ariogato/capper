import cv2 as cv
import numpy as np
from math import floor, ceil, pi, sqrt

class Image:

  class Capillary:

    def __init__(self, cd31_contour, ng2_coverage: float):
      self.cd31_contour = cd31_contour
      self.area_px2 = cv.contourArea(cd31_contour)
      self.ng2_coverage = ng2_coverage

      # This flag will be set by the evaluation and plotting.
      # E.g. if the diameter is out of range
      self.filtered_out = False

    def __iter__(self):
      return iter((self.cd31_contour, self.ng2_coverage))

    def centroid(self):

      M = cv.moments(self.cd31_contour)
      cx = int(M['m10'] / M['m00'])
      cy = int(M['m01'] / M['m00'])

      return (cx, cy)

  class Tile:

    def __init__(self, offset_x, offset_y, width, height, capillaries, hole_mask):
      self.offset_x = offset_x
      self.offset_y = offset_y
      self.width = width
      self.height = height
      self.capillaries = capillaries
      self.hole_mask = hole_mask

  def __init__(self, descriptor, filepath, img):
    """
    :param category: brain, heart, kidney, muscle
    """

    self.descriptor = descriptor
    self.filepath = filepath
    self.x_scale = 0.1624808538555503
    self.y_scale = 0.1624808538555503

    self.img = img
    self.img_auto_thresh = np.zeros_like(img)
    self.hole_mask = np.zeros_like(self.img).astype(np.uint8)

    self.capillaries: list[Image.Capillary] = []
    self.analyzed_area_px2 = 0
    self.hole_area_px2 = 0

    self.tiles = None

  def init_tiles(self, rows, cols):
    self.tiles = [[None] * cols for _ in range(rows)]

  def add_tile(self, row, col, tile):
    if self.tiles is None:
      raise BufferError('Tiles not initialized!')
    self.tiles[row][col] = tile

  def clear_capillary_filter(self):
    for cap in self.capillaries:
      cap.filtered_out = False

    for tile_row in self.tiles:
      for tile in tile_row:
        if tile is not None:
          for cap in tile.capillaries:
            cap.filtered_out = False

  def clear_analysis_data(self):
    self.img_auto_thresh = np.zeros_like(self.img)
    self.hole_mask = np.zeros_like(self.img).astype(np.uint8)
    self.capillaries = []
    self.tiles = None
    self.analyzed_area_px2 = 0
    self.hole_area_px2 = 0

def add_offset(contour, offset):
  """
  Adds (x, y) offset to each point in the contour.
  :param contour: one opencv contour
  :param offset: (x, y) integers
  :return: contour with added offset
  """

  offset_contour = contour.copy()

  for i in range(len(contour)):

    offset_contour[i][0][0] += offset[0]
    offset_contour[i][0][1] += offset[1]

  return offset_contour

def add_offset_multiple(contours, offset):
  """
  Wrapper to offset multiple contours
  :param contours: list of opencv contour
  :param offset: (x, y) integers
  :return: list of offset contours
  """

  offset_contours = []
  for cnt in contours:
    offset_contours.append(add_offset(cnt, offset))

  return offset_contours

def colliding_tiles(tile_width, tile_height, start_x, start_y, end_x, end_y):

  tile_upper_left = (
    floor(start_x / tile_width),
    floor(start_y / tile_height),
  )
  tile_lower_right = (
    floor(end_x / tile_width),
    floor(end_y / tile_height),
  )

  return tile_upper_left, tile_lower_right

