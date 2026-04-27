import numpy as np
import scipy

## Adaptive size gating

def find_adaptive_min_diam(cap_areas_hist_um, bin_edges, MAX_DIAM):
  """
  The min diam is defined as the valley between the two size peaks.
  The first is a peak for debris and the second are the actual capillaries.

  Tested parameters for the histogram:
  np.histogram(cap_sizes, range=(0, 20), bins=1000, density=False)
  hist = scipy.signal.savgol_filter(hist, 101, 2)

  :param cap_areas_um: histogram of array of approx. capillary diameter in µm
  :param bin_edges: bin edges of the histogram (bin_edges[1:])
  :param MAX_DIAM: maximum diameter in µm
  :return: adaptive min diameter in µm. -1 if unsuccessful
  """

  #try to find the two peaks with the highest prominence possible
  found_peaks = []
  prom = 100.0

  while prom >= 0.0:

    # find peaks
    peaks, _ = scipy.signal.find_peaks(cap_areas_hist_um, prominence=prom)
    peaks = bin_edges[peaks]

    # filter out peaks that are too far
    peaks_filtered = peaks[peaks < MAX_DIAM]

    # kmeans of the peaks to check sufficient distance between them
    if len(peaks_filtered) >= 2:

      # kmeans to cluster all small peaks
      whitened = scipy.cluster.vq.whiten(peaks_filtered)
      kmeans = scipy.cluster.vq.kmeans(whitened, 2)[0]

      # sort by size
      kmeans_np = np.array([min(kmeans).item(), max(kmeans).item()])
      kmeans_np *= np.std(peaks_filtered)

      # sufficient distance between peaks: 0.5 µm
      if kmeans_np[1] - kmeans_np[0] > 0.5:

        # store and exit calculation
        found_peaks = kmeans_np
        break

    prom -= 0.2

  # No peaks could be found
  if prom < 0.0:
    return -1

  #print('Before kmeans: ', peaks_filtered)
  #print('After: ', found_peaks)

  # Find the most prominent negative peak between the found peaks
  # If many peaks are there, we take the one with smallest µm. To not miss capillaries
  # The valley must be at least 0.1 µm larger / smaller than the respective peaks
  padding = 0.2
  found_valleys = []
  prom = 50.0

  while prom >= 0.0 and len(found_valleys) == 0:

    valley_idxs, _ = scipy.signal.find_peaks(-1 * cap_areas_hist_um, prominence=prom)
    valleys_filtered = []

    for valley_idx in valley_idxs:
      if found_peaks[0] + padding < bin_edges[valley_idx] < found_peaks[1] - padding:
        valleys_filtered.append(bin_edges[valley_idx])

    if len(valleys_filtered) > 0:
      found_valleys = valleys_filtered

    prom -= 0.5

  return valleys_filtered[0]


def adaptive_size_gating(groups):

    # Filtering rules
    # diameter (approximated by assuming circle area): adaptive (fallback: 3.5) - 12.0 µm
    #
    # minimum ng2 coverage 0.1 %

    MIN_DIAM_RANGE = (3.5, 5.0)
    MAX_DIAM = 12.0

    # Populated groups
    non_empty_groups = [k for k, v in groups.items() if len(v) > 0]

    # Perform size gating
    for i in range(len(non_empty_groups)):

        group = non_empty_groups[i]
        img_objs = groups[group]

        if len(img_objs) == 0:
            continue

        scale_factor = img_objs[0].x_scale

        # convert to px^2
        MAX_DIAM_PX2 = ((MAX_DIAM / scale_factor) / 2) ** 2 * np.pi

        # Clear all previously set flags
        for img_obj in img_objs:
            img_obj.clear_capillary_filter()

        # gather capillary areas in px^2
        cap_sizes_px2 = []
        for img_obj in img_objs:
            for cap in img_obj.capillaries:
                # for plotting all sizes later
                cap_sizes_px2.append(cap.area_px2)

        # convert to µm diameter
        cap_sizes_px2 = np.array(cap_sizes_px2)
        cap_sizes_um = np.sqrt(cap_sizes_px2 / np.pi) * scale_factor * 2

        # calculate histogram
        hist, bin_edges = np.histogram(cap_sizes_um, range=(0, 20), bins=500, density=False)
        hist = scipy.signal.savgol_filter(hist, 25, 2)
        hist_dens, bin_edges_dens = np.histogram(cap_sizes_um, range=(0, 20), bins=500, density=True)
        hist_dens = scipy.signal.savgol_filter(hist_dens, 21, 2)
        hist_dens *= 1000.0

        # determine capillary size thresholds strategy
        adaptive_min_diam = find_adaptive_min_diam(hist_dens, bin_edges_dens[1:], MAX_DIAM)

        if adaptive_min_diam == -1:
            print(f'{group}: adaptive min diam not found')
            adaptive_min_diam = MIN_DIAM_RANGE[0]

        # adaptive diameter is clipped to 5 µm. Anything higher than this is most likely an algorithmic exception
        if adaptive_min_diam > MIN_DIAM_RANGE[1]:
            adaptive_min_diam = MIN_DIAM_RANGE[1]

        # brain is special because of the amount of background and the non uniform course of capillaries
        if 'brain' in group:
            adaptive_min_diam = 8.0

        print(f'{group}: {adaptive_min_diam:.2f}')

        adaptive_min_diam_px2 = ((adaptive_min_diam / scale_factor) / 2) ** 2 * np.pi

        for img_obj in img_objs:
            for cap in img_obj.capillaries:
                # Filter by size
                if not adaptive_min_diam_px2 <= cap.area_px2 <= MAX_DIAM_PX2:
                    cap.filtered_out = True

                # Filter by NG2 coverage
                if cap.ng2_coverage < 0.1:
                    cap.filtered_out = True

            # repeat filter for capillaries in tile objects
            for tile_row in img_obj.tiles:
                for tile in tile_row:
                    for cap in tile.capillaries:
                        # Filter by size
                        if not adaptive_min_diam_px2 <= cap.area_px2 <= MAX_DIAM_PX2:
                            cap.filtered_out = True

                        # Filter by NG2 coverage
                        if cap.ng2_coverage < 0.1:
                            cap.filtered_out = True
