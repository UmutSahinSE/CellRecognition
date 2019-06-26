from PIL import Image, ImageFilter
import cv2
import numpy as np
from skimage import morphology
import math

def combine_two_images(path1,path2, new_file_name):
    img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
    img_bwo = cv2.bitwise_or(img1,img2)
    cv2.imwrite(new_file_name, img_bwo)

def brightness_range_filter(path,lower,higher):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j]>lower and image[i, j]<higher:  # if intensity is in given range, mark as 1, else mark as 0
                image[i, j] = 255
            else:
                image[i, j] = 0
    cv2.imwrite('bw_image.tif', image)

def connect_close_edges(path,window_size, positive):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    y_range = []
    if positive == 1:  # decision of iterate range depending on connection on positive or negative axis
        y_range = range(image.shape[1] - window_size)
    else:
        y_range = range(window_size+1,image.shape[1]-1)


    for i in range(image.shape[0] - window_size):
        for j in y_range:
            if image[i][j] == 255:
                least_distance = 9999  # will going to use this operation once per point to avoid unnecessarily long execution time. Finding closest point that is not a neighbor
                closest_white = []  # location of point to connect with, relative to original point
                for k in range(window_size):
                    for l in range(window_size):
                        if image[i+k][j+positive*l] == 255 and least_distance > math.sqrt(pow(k,2)+pow(l,2))>1.5:
                            least_distance = math.sqrt(pow(k,2)+pow(l,2))
                            closest_white = [k,positive*l]
                if least_distance != 9999:
                    points = find_points_to_fill(i,j,closest_white)  # Bresenham's_Line_Algorithm to detect points to fill between two points
                    for p in points:
                        image[p[0],p[1]]=255

    cv2.imwrite('connected.tif', image)

def find_points_to_fill(i,j,closest_white):  # Bresenham's_Line_Algorithm http://www.roguebasin.com/index.php?title=Bresenham%27s_Line_Algorithm
    x1 = i
    y1 = j
    x2 = x1 + closest_white[0]
    y2 = y1 + closest_white[1]
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def eliminate_small_objects(path, min_size):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im = cv2.threshold(image, 175, 250, cv2.THRESH_BINARY)
    im = im[1]

    a = []
    for i in range(image.shape[0]):
        a.append([0 for j in range(image.shape[1])])

    for i in range(image.shape[0]):  # formatting image to boolean representation to be used in remove_small_objects function of skimage-morphology
        for j in range(image.shape[1]):
            if image[i][j] == 255:
                a[i][j] = 1

    a = np.array(a,bool)

    a = morphology.remove_small_objects(a, 350, connectivity=2)  # remove contours smaller than 350 pixels of area
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if a[i][j] == False:
                image[i][j] = 0  # formatting boolean representation back to intensity representation

    cv2.imwrite('big_only.tif', image)

def fill_holes(path, nth):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = image | im_floodfill_inv

    # Display images.
    cv2.imwrite('filled'+nth+'.tif', im_out)

def eliminate_convex_contours(path, convex_path):
    # before comparing original contours with convex_hulled ones, contours must be matched. As a result of
    # convex_hull operation, two contours may merge into one. To match the contours correctly, Center of gravity of
    # each contour is found. Center of gravity shouldn't change much after convex_hull operation, so closest centers
    # of gravity should represent matching contours https://www.pyimagesearch.com/2016/02/01/opencv-center-of-contour/
    image = cv2.imread(convex_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (696, 520))
    convex_contours, convex_hier = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    convex_contour_centers = []
    convex_areas = []
    for cnt in convex_contours:
        convex_areas.append(cv2.contourArea(cnt))
        M = cv2.moments(cnt)
        convex_contour_centers.append([int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])])
    print(convex_contour_centers)

    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (696, 520))
    contours, hier = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros(image.shape, np.uint8)
    contour_centers = []
    contour_areas = []
    for cnt in contours:
        contour_areas.append(cv2.contourArea(cnt))
        M = cv2.moments(cnt)
        contour_centers.append([int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])])
    print(contour_centers)

    which_convex_contour_will_be_compared_with = []

    for i in contour_centers:
        closest_convex_location = None
        least_distance = 999999999999
        for j in convex_contour_centers:
            distance = math.sqrt(math.pow(i[0]-j[0],2)+math.pow(i[1]-j[1],2))
            if distance < least_distance:
                least_distance = distance
                closest_convex_location = j
        which_convex_contour_will_be_compared_with.append(convex_contour_centers.index(closest_convex_location))

    print(which_convex_contour_will_be_compared_with)

    for i in range(len(which_convex_contour_will_be_compared_with)):
        # after matching contours, areas of original and convex_filled contour are compared and if area didn't increase as much, that gives a result that contour was originally convex enough and doesn't need further operation
        error = convex_areas[which_convex_contour_will_be_compared_with[i]]-contour_areas[i]
        percentage = error*100/convex_areas[which_convex_contour_will_be_compared_with[i]]
        print(percentage)
        if(percentage>35):
            cv2.drawContours(mask,[contours[i]],0,255,-1)
        else:
            cv2.drawContours(image, [contours[i]], 0, (0, 0, 0), 1)

    cv2.imwrite('remove_convex.tif', mask)

def convex_hull(path):  # convex hull algorithm applied on each contour https://medium.com/@harshitsikchi/convex-hulls-explained-baab662c4e94
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    contours, hier = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    uni_hull = []
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        uni_hull.append(hull)  # <- array as first element of list
        cv2.drawContours(image, uni_hull, -1, 255, 2)

    im_floodfill = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)

    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)

    # Combine the two images to get the foreground.
    im_out = image | im_floodfill_inv

    cv2.imwrite('convex.tif', im_out)

def remove_unwanted(path):  # in training data, this function is used to mask water bubbles and other objects that are not related with cells we are trying to detect
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print(image.shape)
    for i in range(440, 519):
        for j in range(450, 690):
            image[i, j] = 0
    for i in range(360, 519):
        for j in range(450, 460):
            image[i, j] = 0
    for i in range(20, 70):
        for j in range(490, 570):
            image[i, j] = 0
    for i in range(380, 420):
        for j in range(475, 505):
            image[i, j] = 0
    for i in range(360, 380):
        for j in range(450, 475):
            image[i, j] = 0
    for i in range(480, 519):
        for j in range(50, 100):
            image[i, j] = 0
    for i in range(415, 519):
        for j in range(575, 600):
            image[i, j] = 0
    cv2.imwrite(path, image)

############################################
########## start of execution ##############

for i in range(1,2):
    extra_0 = ""  # to adapt naming convention
    if i<10:
        extra_0 = "0"
    original_image = 'test/test0'+extra_0+str(i)+'.tif'
    image = Image.open(original_image)
    image = image.filter(ImageFilter.FIND_EDGES)  # find edges of image
    image.save('pil_edges.tif')

    brightness_range_filter("pil_edges.tif",28,255) # threshold edges in image and make it binary
    connect_close_edges("bw_image.tif", 12, 1)  # connect points with other points that are in 12*12 window in positive axis
    connect_close_edges("connected.tif", 12, -1)   # connect points with other points that are in 12*12 window in negative axis
    eliminate_small_objects("connected.tif", 5)  # connecting close edges creates contours. Small contours means they
    # are not cells and should be removed
    fill_holes("big_only.tif", "1")  # connecting edges may create contours with empty holes in them. this function fill these holes.
    convex_hull("filled1.tif")  # convex hull algorithm wraps the contours so that all of them are convex objects.
    eliminate_convex_contours("filled1.tif","convex.tif")  # compares original contours with convex_filled ones and
    # separates contours that gained more than 35% area as result of convex_hull operation
    connect_close_edges("remove_convex.tif", 50, 1)  # when relatively convex objects are separated, connecting edges
    # can be done with a larger window size without accidentally combining two cells
    connect_close_edges("connected.tif", 50, -1)
    fill_holes("connected.tif", "2")  # filling holes in new contours
    combine_two_images("filled1.tif","filled2.tif", "segresult_test0"+extra_0+str(i)+".tif")
    # merging previously separated relatively convex enough contours with newly reformed relatively concave contours
