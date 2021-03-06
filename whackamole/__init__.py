from math import pi
from sklearn.cluster import KMeans
import cv2 as cv
import numpy as np


def do_nothing(image):
    return image


def jaccard_index(y_real, y_pred):
    intersection = np.count_nonzero(cv.bitwise_and(y_real, y_pred))
    union = np.count_nonzero(cv.bitwise_or(y_real, y_pred))
    return intersection/union


def negative_threshold(threshold):
    white_pixels = 0
    black_pixels = 0
    for i in range(threshold.shape[0]):
        for j in range(threshold.shape[1]):
            if threshold[i][j] == 0:
                black_pixels = black_pixels + 1
            else:
                white_pixels = white_pixels + 1

        ret = cv.bitwise_not(threshold) if white_pixels > black_pixels else threshold
    return ret


def clustering(image):
    clusters = 4

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    # cv.imwrite("0_grayscale.png", gray)
    # convert gray image to 1D-array
    img = gray.reshape((gray.shape[0] * gray.shape[1], 1))

    # k-means to cluster pixels
    km = KMeans(n_clusters=clusters)
    km.fit(img)

    # cluster centers (main colours or main gray levels)
    colours = km.cluster_centers_
    main_colours = colours.astype(int)

    # gray centers from gray image clustering
    gray_levels = [int(main_colours[i][0]) for i in range(clusters)]

    # select best thresholding limit
    max_val = np.max(gray_levels)
    min_val = np.min(gray_levels)
    gray_levels.remove(max_val)
    max_val2 = np.max(gray_levels)
    gray_levels.remove(min_val)
    min_val2 = np.min(gray_levels)
    bound = int((min_val2 + max_val2)/2)

    # thresholding
    _, thresholded = cv.threshold(gray, bound, 255, cv.THRESH_BINARY_INV)
    # cv.imwrite("1_thresholded.png", thresholded)

    contours, _ = cv.findContours(thresholded, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # tmp = np.zeros((400, 600), dtype=np.uint8)
    # cv.drawContours(tmp, contours, -1, (255, 255, 255), -1, 4)
    # cv.imwrite("2_unfiltered_cc.png", tmp)
    i_height, i_width = gray.shape[0], gray.shape[1]

    final_contours = []
    for c in contours:
        br = cv.boundingRect(c)
        if br[0] < 10 or br[1] < 10 or br[0]+br[2]>i_width-10 or br[1]+br[3]>i_height-10:
            final_contours.append(c)

    ret1 = np.zeros(gray.shape, dtype=np.uint8)
    cv.drawContours(ret1, final_contours, -1, (255, 255, 255), -1, 4)
    # tmp = np.zeros((400, 600), dtype=np.uint8)
    # cv.drawContours(tmp, contours, -1, (255, 255, 255), -1, 4)
    # cv.imwrite("3_borders.png", ret1)

    thresholded1 = cv.subtract(thresholded, ret1)
    # cv.imwrite("4_borders_removed.png", thresholded1)

    contours, _ = cv.findContours(thresholded1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    final = None
    final_a = 0
    for c in contours:
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        circ = circularity(area, perimeter)
        if area > 400 and circ < 0.8:
            if area > final_a:
                final = c
                final_a = area

    ret3 = np.zeros((400, 600), dtype=np.uint8)
    if final is not None:
        cv.drawContours(ret3, [final], -1, (255, 255, 255), -1, 4)
        # cv.imwrite("5_final_result.png", ret3)

    return ret3


def preprocessing(image):
    copied = image.astype('uint8')
    gray_scale = cv.cvtColor(copied, cv.COLOR_RGB2GRAY)
    # Kernel for the morphological filtering
    kernel = cv.getStructuringElement(1, (17, 17))
    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv.morphologyEx(gray_scale, cv.MORPH_BLACKHAT, kernel)
    # cv.imwrite("blackhat.png", blackhat)
    # intensify the hair countours in preparation for the inpainting
    # algorithm
    _, thresh2 = cv.threshold(blackhat, 10, 255, cv.THRESH_BINARY)
    # cv.imwrite("hair_mask.png", thresh2)
    # inpaint the original image depending on the mask
    inpainted = cv.inpaint(copied, thresh2, 1, cv.INPAINT_TELEA)
    # cv.imwrite("inpaint.png", inpainted)
    # mean shift filtering
    return cv.pyrMeanShiftFiltering(inpainted, 8, 8)


def circularity(area, perimeter):
    if perimeter == 0:
        perimeter = 0.0001
    return (4*pi*area)/(perimeter**2)


def borders_mask(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    _, gray = cv.threshold(gray, 48, 255, cv.THRESH_BINARY_INV)
    contours, _ = cv.findContours(gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    i_height, i_width = gray.shape[0], gray.shape[1]

    final_contours = []
    for c in contours:
        br = cv.boundingRect(c)
        if br[0] < 10 or br[1] < 10 or br[0]+br[2] > i_width-10 or br[1]+br[3] > i_height-10:
            final_contours.append(c)

    ret = np.zeros(gray.shape, dtype=np.uint8)
    cv.drawContours(ret, final_contours, -1, (255, 255, 255), -1 , 4)
    return 255 - ret


def find_contours(image):
    # Conversion to HSV space and taking the saturation channel
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    img = cv.split(hsv)[1]
    # cv.imwrite("0_saturation.png", img)

    # Thresholding
    _, dst = cv.threshold(img, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imwrite("1_thresholded.png", dst)

    # Mask borders
    b_mask = borders_mask(image)
    # cv.imwrite("2_borders_mask.png", b_mask)

    dst = cv.bitwise_and(dst, b_mask)

    # Finding contours to remove holes
    contours, _ = cv.findContours(dst, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # tmp = np.zeros((400, 600), dtype=np.uint8)
    # cv.drawContours(tmp, contours, -1, (255, 255, 255), -1, 4)
    # cv.imwrite("3_unfiltered_cc.png", tmp)
    final = None
    final_ref = 1.0
    for c in contours:
        area = cv.contourArea(c)
        perimeter = cv.arcLength(c, True)
        circ = circularity(area, perimeter)
        if area > 400 and circ < 0.8:
            if area > final_ref:
                final = c
                final_ref = area

    mask = np.zeros((400, 600), dtype=np.uint8)
    if final is not None:
        cv.drawContours(mask, [final], -1, (255, 255, 255), -1, 4)
        # cv.imwrite("4_final_mask.png", mask)

    ### WATERSHED START
    ## FOREGROUND
    # Distance transform and thresholding
    fg = cv.distanceTransform(mask, cv.DIST_L2, cv.DIST_MASK_PRECISE)
    fg_min = np.amin(fg)
    fg_max = np.amax(fg)
    fg[:] = (fg - fg_min) * 255 // (fg_max - fg_min)
    fg = np.uint8(fg)
    _, fg = cv.threshold(fg, 100, 255, cv.THRESH_BINARY)

    ## BACKGROUND
    bg = cv.dilate(mask, None, iterations=4)
    bg[bg == 0] = 128
    bg[bg == 255] = 0

    ## MARKERS
    markers = cv.add(fg, bg)
    # cv.imwrite("5_watershed_mask.png", markers)
    markers32 = np.int32(markers)

    cv.watershed(image, markers32)
    w_mask = cv.convertScaleAbs(markers32)
    # cv.imwrite("6_watershed_mask.png", w_mask)
    w_mask[w_mask == 128] = 0
    w_mask[w_mask == 1] = 255
    w_mask[0, :] = 0
    w_mask[w_mask.shape[0] - 1, :] = 0
    w_mask[:, 0] = 0
    w_mask[:, w_mask.shape[1] - 1] = 0
    # cv.imwrite("7_final_result.png", w_mask)

    return w_mask
