import xlsxwriter
import os
from datetime import datetime
import numpy as np
import scipy
from image import colliding_tiles

class WorksheetWrapper:

  def __init__(self, workbook, worksheet_name, row_offset=0, col_offset=0):
    self.workbook = workbook
    self.worksheet = workbook.add_worksheet(worksheet_name)
    self.bold = self.workbook.add_format({'bold': 1})

    self.start_row = row_offset
    self.start_col = col_offset

    self.row = self.start_row
    self.col = self.start_col

    self.last_filled_row = self.start_row
    self.last_filled_col = self.start_col

  def write(self, content, bold=False):

    if bold:
      self.worksheet.write(self.row, self.col, content, self.bold)
    else:
      self.worksheet.write(self.row, self.col, content)

    if self.row > self.last_filled_row:
      self.last_filled_row = self.row

    if self.col > self.last_filled_col:
      self.last_filled_col = self.col

    self.row += 1

  def next_column(self, distance=3):

    self.row = self.start_row
    self.col += distance

  def set_position(self, row, col):
    self.row = row
    self.col = col

  def set_start_position(self, row, col):
    self.start_row = row
    self.start_col = col

def worksheet_write_stats(worksheet: WorksheetWrapper, num_cols, xlsx_head_offset):
  """
  NOTE: This function is for german excel.
  :param worksheet:
  :param num_cols:
  :param xlsx_head_offset:
  :return:
  """

  alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'

  last_row = worksheet.last_filled_row

  # write headlines to the side
  worksheet.set_position(xlsx_head_offset + last_row - 1, 0)
  worksheet.set_start_position(xlsx_head_offset + last_row - 1, 0)
  worksheet.write('Mean', bold=True)
  worksheet.write('STABWN', bold=True)
  worksheet.write('N', bold=True)
  worksheet.write('SEM', bold=True)
  worksheet.next_column(distance=1)

  # write formulas
  for i in range(num_cols):

    xlsx_cell_range = f'{alphabet[i + 1]}{xlsx_head_offset}:{alphabet[i + 1]}{xlsx_head_offset + last_row - 1}'

    worksheet.write(f'=AVERAGE({xlsx_cell_range})', bold=True)
    worksheet.write(f'=STDEV({xlsx_cell_range})', bold=True)
    worksheet.write(f'=COUNT({xlsx_cell_range})', bold=True)
    worksheet.write(f'=STDEV({xlsx_cell_range})/SQRT(COUNT({xlsx_cell_range}))', bold=True)

    worksheet.next_column(distance=1)

def write_to_xlsx(path, current_accumulated_groups, accumulated_groups, groups, sliding_window_dimensions, scale_factor):

  samplewindow_size = (750, 750)

  # initialize xlsx file
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  filename = f"report_accumulated_groups_{timestamp}.xlsx"
  xlsx_path = os.path.join(path, filename)


  workbook = xlsxwriter.Workbook(xlsx_path)
  cd31_accumulated_worksheet = WorksheetWrapper(workbook, 'accumulated densities', 0, 1)
  ng2_accumulated_worksheet = WorksheetWrapper(workbook, 'accumulated coverages', 0, 1)

  # tile definitions
  tile_width, tile_height = sliding_window_dimensions

  # for the violin plot
  sw_densities = {g: [] for g in current_accumulated_groups}
  sw_coverages = {g: [] for g in current_accumulated_groups}

  for acc_group in current_accumulated_groups:

    print(acc_group)
    cd31_accumulated_worksheet.write(acc_group, bold=True)
    cd31_accumulated_worksheet.write('')
    ng2_accumulated_worksheet.write(acc_group, bold=True)
    ng2_accumulated_worksheet.write('')

    # all samples
    group_samples = accumulated_groups[acc_group]

    for sample in group_samples:

      images = groups[sample]

      print(f'Writing {sample} ({len(images)} images)')

      for image_object in groups[sample]:

        # hole algorithm only makes sense for heart tissue
        is_heart = any([s in image_object.descriptor for s in ['heart', 'sepsis', 'tac', 'transient']])

        img_num_caps = len([c for c in image_object.capillaries if not c.filtered_out])
        img_area_mm2 = (image_object.analyzed_area_px2 - (image_object.hole_area_px2 if is_heart else 0)) * \
                       (image_object.x_scale * image_object.y_scale) / 1_000_000

        # Exclude extreme outlier images
        if img_num_caps / img_area_mm2 < 1100.0 and is_heart:
          print(f'[Excluding {image_object.descriptor}]')
          continue

        num_tile_rows = len(image_object.tiles)
        num_tile_cols = len(image_object.tiles[0])

        # dimensions of the full image filled out by tiles. There's always some
        # margin pixels omitted due to image shape indivisible by tile size.
        tiled_width = num_tile_cols * tile_width
        tiled_height = num_tile_rows * tile_height

        # total number of sample windows across the tiled portion of the image
        samplewindow_rows = tiled_height // samplewindow_size[0]
        samplewindow_cols = tiled_width // samplewindow_size[1]

        for sw_row in range(samplewindow_rows):
          for sw_col in range(samplewindow_cols):

            samplewindow_capillaries = []

            # calculate sample window
            samplewindow_upper_left = (
              sw_col * samplewindow_size[1],
              sw_row * samplewindow_size[0],
            )

            samplewindow_lower_right = (
              samplewindow_upper_left[0] + samplewindow_size[1] - 1,
              samplewindow_upper_left[1] + samplewindow_size[0] - 1,
            )

            # determine tiles to assess in current samplewindow
            sw_tiles_upper_left, sw_tiles_lower_right = colliding_tiles(
              tile_width, tile_height,
              samplewindow_upper_left[0], samplewindow_upper_left[1],
              samplewindow_lower_right[0], samplewindow_lower_right[1],
            )

            # Iterate over tiles
            for tile_row in range(sw_tiles_upper_left[1], sw_tiles_lower_right[1] + 1):
              for tile_col in range(sw_tiles_upper_left[0], sw_tiles_lower_right[0] + 1):

                tile = image_object.tiles[tile_row][tile_col]

                # check if the tile capillaries' centroids are in the samplewindow
                for cap in tile.capillaries:

                  if cap.filtered_out:
                    continue

                  c_x, c_y = cap.centroid()

                  # add offset to centroid
                  c_x += tile.offset_x
                  c_y += tile.offset_y

                  # point collision detection
                  if samplewindow_upper_left[0] <= c_x <= samplewindow_lower_right[0] and samplewindow_upper_left[
                    1] <= c_y <= samplewindow_lower_right[1]:
                    samplewindow_capillaries.append(cap)

            # holes should only be considered for hearts
            sw_hole_area_px2 = 0
            if is_heart:
              sw_hole_mask = image_object.hole_mask[
                samplewindow_upper_left[1]:samplewindow_lower_right[1] + 1, samplewindow_upper_left[0]:
                                                                            samplewindow_lower_right[0] + 1]
              sw_hole_area_px2 = np.count_nonzero(sw_hole_mask)

            sw_num_caps = len(samplewindow_capillaries)
            sw_area_px2 = samplewindow_size[0] * samplewindow_size[1] - sw_hole_area_px2
            sw_area_mm2 = sw_area_px2 * (image_object.x_scale * image_object.y_scale) / 1_000_000

            # sample window (almost) entirely a hole
            # about the size of a capillary would be too small to reliably be evaluated
            min_area_px2 = np.power(8.0 / scale_factor, 2)
            if is_heart and sw_area_px2 < min_area_px2:

              # should not be the case, as the hole mask was applied before analysis
              if sw_num_caps > 0:
                print(f'Warning {image_object.descriptor} has capillaries in hole]')

              continue

            sw_density = sw_num_caps / sw_area_mm2

            sw_densities[acc_group].append(sw_density)
            sw_coverages[acc_group].extend([cap.ng2_coverage for cap in samplewindow_capillaries])

            if sw_density > 5000:
              print(f'Warning {image_object.descriptor}')
              print(f'\t{sw_num_caps} / {sw_area_mm2:.6f} mm^2 (= {sw_area_px2} px^2)')
              print(f'\t= {sw_density} mm^-2')

            # density in sample window
            cd31_accumulated_worksheet.write(sw_density)

            # NG2 coverage
            for cap in samplewindow_capillaries:
              ng2_accumulated_worksheet.write(cap.ng2_coverage)

    cd31_accumulated_worksheet.next_column(distance=1)
    ng2_accumulated_worksheet.next_column(distance=1)

  # statistics in excel sheet
  worksheet_write_stats(cd31_accumulated_worksheet, len(current_accumulated_groups), 3)
  worksheet_write_stats(ng2_accumulated_worksheet, len(current_accumulated_groups), 3)


  # print significance
  for k1, v1 in sw_densities.items():
    for k2, v2 in sw_densities.items():

      if k1 == k2:
        continue

      t_test = scipy.stats.ttest_ind(v1, v2)
      if t_test.pvalue > 0.05:
        print(f'{k1} vs {k2} not significant: {t_test.pvalue}')

  workbook.close()
