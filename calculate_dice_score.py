import numpy as np
import cv2

k=2

for i in range(1,15):
    gt_path = 'glassmatrigel/01_GT_rename/mask' + str(i)+'.tif'
    res_path = 'results/glassmatrigel-01/final_result/' + str(i) + '.tif'

    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    res_image = cv2.imread(res_path, cv2.IMREAD_GRAYSCALE)

    gt_image = np.asarray(gt_image).astype(np.bool)
    res_image = np.asarray(res_image).astype(np.bool)

    if gt_image.shape != res_image.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = gt_image.sum() + res_image.sum()
    if im_sum == 0:
        print(0)

    # Compute Dice coefficient
    intersection = np.logical_and(gt_image, res_image)

    print(2. * intersection.sum() / im_sum)

for i in ["001","005","006","007","021","049","059","067","072","075","092","096","100","102","112"]:
    gt_path = 'PhC/01_GT/SEG/man_segBW' + i+'.tif'
    res_path = 'results/PhC-01/final_result/' + str(int(i)) + '.tif'

    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    res_image = cv2.imread(res_path, cv2.IMREAD_GRAYSCALE)

    gt_image = np.asarray(gt_image).astype(np.bool)
    res_image = np.asarray(res_image).astype(np.bool)

    if gt_image.shape != res_image.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = gt_image.sum() + res_image.sum()
    if im_sum == 0:
        print(0)

    # Compute Dice coefficient
    intersection = np.logical_and(gt_image, res_image)

    print(2. * intersection.sum() / im_sum)

for i in ["010","017","019","020","022","026","035","036","041","049","051","052","059","074","077","085","092","106","112"]:
    gt_path = 'PhC/02_GT/SEG/man_segBW' + i+'.tif'
    res_path = 'results/PhC-02/final_result/' + str(int(i)) + '.tif'

    gt_image = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    res_image = cv2.imread(res_path, cv2.IMREAD_GRAYSCALE)

    gt_image = np.asarray(gt_image).astype(np.bool)
    res_image = np.asarray(res_image).astype(np.bool)

    if gt_image.shape != res_image.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = gt_image.sum() + res_image.sum()
    if im_sum == 0:
        print(0)

    # Compute Dice coefficient
    intersection = np.logical_and(gt_image, res_image)

    print(2. * intersection.sum() / im_sum)