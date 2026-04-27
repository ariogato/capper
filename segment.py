import cv2 as cv
import numpy as np
from image_processing import find_holes, find_cd31, find_ng2, clip_convert_8b
from image import Image, add_offset_multiple

def segment(image_objects):
    """
    Analyzes the image objects passed by reference
    :param image_objects:
    :return:
    """

    # (width, height)
    sliding_window_dimensions = (500, 500)

    # iterate over all samples to test
    for image_object in image_objects:

        print(image_object.descriptor)

        # clear all previously aquired data
        image_object.clear_analysis_data()

        # read image
        img = image_object.img
        scale_factor = image_object.x_scale

        # split channels
        if (img.shape[2] > 3):
            b, g, r, _ = cv.split(img)
        else:
            b, g, r = cv.split(img)

        # get image dimensions
        img_height, img_width = img.shape[:2]

        # number of rows and columns of tiles
        tile_rows = img_height // sliding_window_dimensions[1]
        tile_cols = img_width // sliding_window_dimensions[0]

        # determine number of iterations
        max_i = img_width - sliding_window_dimensions[0]
        max_j = img_height - sliding_window_dimensions[1]

        # the last iterations in row, respectively column are smaller in image size
        last_i = max_i - max_i % sliding_window_dimensions[0]
        last_j = max_j - max_j % sliding_window_dimensions[1]

        # find holes in the area to be analyzed
        hole_mask = find_holes(
            r[0: (last_j + 1) * sliding_window_dimensions[1], 0: (last_i + 1) * sliding_window_dimensions[0]])
        hole_mask_inv = cv.bitwise_not(hole_mask)

        # apply inverse hole mask
        r = cv.bitwise_and(r, r, mask=hole_mask_inv)
        g = cv.bitwise_and(g, g, mask=hole_mask_inv)

        image_object.hole_mask = hole_mask.copy()
        image_object.hole_area_px2 = np.count_nonzero(hole_mask)
        image_object.init_tiles(tile_rows, tile_cols)

        # make the window slide
        for i in range(0, max_i, sliding_window_dimensions[0]):
            for j in range(0, max_j, sliding_window_dimensions[1]):

                # get region
                # j -> rows | y
                # i -> columns | x
                rows = slice(j, j + sliding_window_dimensions[1])
                cols = slice(i, i + sliding_window_dimensions[0])

                r_roi = r[rows, cols]
                g_roi = g[rows, cols]

                # grad_thresh of 0.7 is working for both in the mouse!
                r_roi_thresh = clip_convert_8b(r_roi, grad_thresh=0.05, VERBOSE=False)
                g_roi_thresh = clip_convert_8b(g_roi, grad_thresh=0.7)

                # proceed with regular algorithms
                cd31_channel_thresh = r_roi_thresh
                ng2_channel_thresh = g_roi_thresh

                # previously min_diam=4, max_diam=10
                cd31_contours = find_cd31(cd31_channel_thresh, scale_factor, thresh_c=-3)
                ng2_coverage = find_ng2(ng2_channel_thresh, cd31_contours, scale_factor)

                # Save contours and coverages to image object
                tile_capillaries = []
                offset_capillaries = []
                offset_contours = add_offset_multiple(cd31_contours, (i, j))
                for contour, offset_contour, coverage in zip(cd31_contours, offset_contours, ng2_coverage):
                    offset_capillaries.append(Image.Capillary(offset_contour, coverage))
                    tile_capillaries.append(Image.Capillary(contour, coverage))

                image_object.capillaries.extend(offset_capillaries)

                # save auto thresholded tile and area to image object
                image_object.img_auto_thresh[rows, cols, 1] = g_roi_thresh
                image_object.img_auto_thresh[rows, cols, 2] = r_roi_thresh
                image_object.analyzed_area_px2 += sliding_window_dimensions[0] * sliding_window_dimensions[1]

                # save tile
                image_object.add_tile(
                    row=j // sliding_window_dimensions[1],
                    col=i // sliding_window_dimensions[0],
                    tile=Image.Tile(i, j, sliding_window_dimensions[0], sliding_window_dimensions[1], tile_capillaries,
                                    hole_mask[rows, cols])
                )