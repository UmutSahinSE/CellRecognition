import numpy as np
import cv2

k=2

for i in range(1,15):
    gt_init = 'glassmatrigel/01_GT_rename/mask'
    res_init = 'results/result'

    gt_path = gt_init+str(i)+'.tif'
    res_path = res_init+str(i)+'.tif'

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