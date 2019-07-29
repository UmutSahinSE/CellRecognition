# CellRecognition
A python script which is used for segmenting cells from provided phase contrast images. This is a university project.

## Stationary Object Detection Steps
1. To detect stationary objects, two images need to be picked by user. These images should both expose stationary objects as much as possible while having distinctly shaped&located cells.
1. Algorithm divides the first image into smaller windows and uses pattern matching using these windows as template on second image one by one. For a given window, if location of the highest similarity is close to location of that window, the location is declared to be a stationary object location.
1. For each image, pattern of each stationary object location is checked. If patterns located in this window between observed image and one of the picked images for stationary object detection are similar enough, that window on observed image is marked as stationary object.

## Segmentation Steps
1. Copying the original image, then applying smoothing and edge detection operations on that copy.
1. Copying the original image once more, then applying smoothing and background elimination operations on that copy. Background elimination simply checks if the difference among intensity of pixels in a window is high enough.
1. Doing an and operation on edge detected, background eliminated and stationary object eliminated images so that only relevant edges will remain. Stationary object elimination is explained under "Stationary Object Detection Steps" title.
1. Connecting close-distance disconnected edges.
1. Filling contours.
1. Eliminating small objects.

## Necessary Improvements
1. Algorithm must be optimized to run faster.
1. Algorithm must be improved in order to detect cells from wider range of images.
