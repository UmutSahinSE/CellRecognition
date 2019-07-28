from PIL import Image, ImageFilter
import cv2
import numpy as np
from skimage import morphology
import math
import matplotlib.pyplot as plt
from statistics import mode
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import match_template

def combine_two_images(path1,path2, new_file_name):
    img1 = cv2.imread(path1,cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(path2,cv2.IMREAD_GRAYSCALE)
    img_bwo = cv2.bitwise_or(img1,img2)
    cv2.imwrite(new_file_name, img_bwo)

def brightness_range_filter(path,lower,higher,image_index,file_init):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if image[i, j]>lower and image[i, j]<higher:  # if intensity is in given range, mark as 1, else mark as 0
                image[i, j] = 255
            else:
                image[i, j] = 0

    cv2.imwrite('bw_image.tif', image)

def connect_close_edges(image,window_size,index):
    y_range = []
    # if positive == 1:  # decision of iterate range depending on connection on positive or negative axis
    #     y_range = range(image.shape[1] - window_size)
    # else:
    #     y_range = range(window_size+1,image.shape[1]-1)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 255:
                if i < image.shape[0] - 2 and j < image.shape[1] - 2 and image[i+1][j+1] == 0:
                    k = 1
                    white_neighbor_flag = False
                    nearest_white = [0,0]
                    while k < window_size and not white_neighbor_flag:
                        k += 1
                        for l in range(1, k):
                            if i + k - l < image.shape[0] and j + l < image.shape[1] and image[i + k - l][j + l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [k-l,l]
                                break
                    while k > 0 and not white_neighbor_flag:
                        k-=1
                        for l in range(1,k):
                            if i + window_size - l < image.shape[0] and j + window_size - k + l < image.shape[1] and image[i + window_size - l][j + window_size - k + l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [window_size - l, window_size - k + l]
                                break
                    if white_neighbor_flag:
                        points = find_points_to_fill(i,j,nearest_white)  # Bresenham's_Line_Algorithm to detect points to fill between two points
                        for p in points:
                            if image[p[0],p[1]] != 255:
                                image[p[0],p[1]]=90
                if i > 2 and j > 2 and image[i-1][j-1] == 0:
                    k = 1
                    white_neighbor_flag = False
                    nearest_white = [0,0]
                    while k < window_size and not white_neighbor_flag:
                        k += 1
                        for l in range(1, k):
                            if i - k + l > 0 and j - l > 0 and image[i - k + l][j - l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [l-k,-l]
                                break
                    while k > 0 and not white_neighbor_flag:
                        k-=1
                        for l in range(1,k):
                            if i - window_size + l > 0 and j - window_size + k - l > 0 and image[i - window_size + l][j - window_size + k - l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [l - window_size, k - l - window_size]
                                break
                    if white_neighbor_flag:
                        points = find_points_to_fill(i,j,nearest_white)  # Bresenham's_Line_Algorithm to detect points to fill between two points
                        for p in points:
                            if image[p[0],p[1]] != 255:
                                image[p[0],p[1]]=90
                if i < image.shape[0] - 2 and j > 2 and image[i+1][j-1] == 0:
                    k = 1
                    white_neighbor_flag = False
                    nearest_white = [0,0]
                    while k < window_size and not white_neighbor_flag:
                        k += 1
                        for l in range(1, k):
                            if i + k - l < image.shape[0] and j - l > 0 and image[i + k - l][j - l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [k-l,-l]
                                break
                    while k > 0 and not white_neighbor_flag:
                        k-=1
                        for l in range(1,k):
                            if i + window_size - l < image.shape[0] and j - window_size + k - l > 0 and image[i + window_size - l][j - window_size + k - l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [window_size - l, k-window_size-l]
                                break
                    if white_neighbor_flag:
                        points = find_points_to_fill(i,j,nearest_white)  # Bresenham's_Line_Algorithm to detect points to fill between two points
                        for p in points:
                            if image[p[0],p[1]] != 255:
                                image[p[0],p[1]]=90
                if i > 2 and j < image.shape[1] - 2 and image[i-1][j+1] == 0:
                    k = 1
                    white_neighbor_flag = False
                    nearest_white = [0,0]
                    while k < window_size and not white_neighbor_flag:
                        k += 1
                        for l in range(1, k):
                            if i - k + l > 0 and j + l < image.shape[1] and image[i - k + l][j + l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [l-k,l]
                                break
                    while k > 0 and not white_neighbor_flag:
                        k-=1
                        for l in range(1,k):
                            if i - window_size + l > 0 and j + window_size - k + l < image.shape[1] and image[i - window_size + l][j + window_size - k + l] == 255:
                                white_neighbor_flag = True
                                nearest_white = [l - window_size, window_size - k + l]
                                break
                    if white_neighbor_flag:
                        points = find_points_to_fill(i,j,nearest_white)  # Bresenham's_Line_Algorithm to detect points to fill between two points
                        for p in points:
                            if image[p[0],p[1]] != 255:
                                image[p[0],p[1]]=90
                if i < image.shape[0] - 2 and image[i+1][j] == 0:
                    for k in range(2,window_size):
                        if i + k < image.shape[0] and image[i + k][j] == 255:
                            for l in range(1,k):
                                if image[i+l][j] != 255:
                                    image[i+l][j] = 90
                            break
                if j > 2 and image[i][j-1] == 0:
                    for k in range(2,window_size):
                        if j - k > 0 and image[i][j-k] == 255:
                            for l in range(1,k):
                                if image[i][j-l] != 255:
                                    image[i][j-l] = 90
                            break
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j] == 90:
                image[i][j] = 255

    return image

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

def eliminate_small_objects(image, min_size):
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

    a = morphology.remove_small_objects(a, min_size, connectivity=2)  # remove contours smaller than 350 pixels of area
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if a[i][j] == False:
                image[i][j] = 0  # formatting boolean representation back to intensity representation

    return image

def fill_holes(image):
    image[0] = [0]*len(image[0])
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
    return im_out

def eliminate_convex_contours(path, convex_path, persantage_threshold):
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
        if(percentage>persantage_threshold):
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

def mse(imageA, imageB):
    err = np.sum((imageA.astype("float") - imageB.astype("float")) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err

def detect_stationary(compare_image_path2,compare_image_path1,window_size):
    compare_image_path2 = cv2.imread(compare_image_path2, cv2.IMREAD_GRAYSCALE)
    compare_image1 = cv2.imread(compare_image_path1, cv2.IMREAD_GRAYSCALE)
    # background_template = io.imread("background_template.tif", as_gray=True)
    points = []

    for i in range(1, int((compare_image_path2.shape[0])/window_size)):
        for j in range(1, int((compare_image_path2.shape[1])/window_size)):
            template = compare_image_path2[i*window_size:i*window_size+window_size, j*window_size:j*window_size+window_size]
            # mse_res = mse(template,background_template)
            # if mse_res>6000:
                # io.imshow(template)
                # plt.show()
            res = cv2.matchTemplate(compare_image1, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.1 and abs(max_loc[0]-j*window_size)<window_size and abs(max_loc[1] -i*window_size)<window_size :
                points.append([max_loc[0],max_loc[1],template])
    return points

def test_stationary(current_image_path,points,image_to_be_changed,window_size):
    current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    for p in points:
        res = cv2.matchTemplate(current_image, p[2], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.1 and abs(max_loc[0] - p[0]) < window_size and abs(max_loc[1] - p[1]) < window_size:
            for i in range(window_size):
                for j in range(window_size):
                    image_to_be_changed[p[1]+i,p[0]+j]=0
    return image_to_be_changed

def detect_background(current_image,index,threshold,window_size):
    for i in range(0, int((current_image.shape[0])/window_size)):
        for j in range(0, int((current_image.shape[1])/window_size)):
            if j == int((current_image.shape[1])/window_size)-1:
                template = current_image[i * window_size:i * window_size + window_size, j * window_size:]
            else:
                template = current_image[i*window_size:i*window_size+window_size, j*window_size:j*window_size+window_size]
            mins = []
            maxes = []
            for m in range(len(template)):
                mins.append(min(template[m]))
                maxes.append(max(template[m]))
            if max(maxes) - min(mins) < threshold:
                for m in range(len(template)):
                    template[m] = [0] * len(template[m])
            # else:
            #     for m in range(len(template)):
            #         template[m] = [255] * len(template[m])


    template = current_image[i*window_size:]
    for m in range(len(template)):
        template[m] = [0] * len(template[m])
    return current_image


def do_segmentation(original_image_path,naming,i,do_eliminate_stationary,compare_img_path_1,compare_img_path_2):
    unedited_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    min_size = unedited_image.shape[0]
    bilateral_param_1 = 9
    bilateral_param_2 = 150
    detect_background_window_size = int(unedited_image.shape[0]/100)
    detect_background_min_diff = int(unedited_image.shape[0]/100)
    if detect_background_min_diff > 5:
        detect_background_min_diff = 5
    canny_param_1 = 0
    canny_param_2 = 7
    max_edge_connect_dist = int(unedited_image.shape[0]/100)
    detect_stationary_window_size = int(unedited_image.shape[0]/15)


    background_eliminated_image = unedited_image.copy()
    background_eliminated_image = cv2.bilateralFilter(background_eliminated_image, bilateral_param_1, bilateral_param_2, bilateral_param_2)
    background_eliminated_image = detect_background(background_eliminated_image, i, detect_background_min_diff,detect_background_window_size)
    cv2.imwrite('results/'+naming+'/back/'+str(i)+'.tif', background_eliminated_image)

    if do_eliminate_stationary:
        stationary_points = detect_stationary(compare_img_path_1,compare_img_path_2,detect_stationary_window_size)
        background_eliminated_image = test_stationary(original_image_path, stationary_points, background_eliminated_image,detect_stationary_window_size)
        for k in range(unedited_image.shape[0]):
            for l in range(unedited_image.shape[1]):
                if background_eliminated_image[k,l] != 0:
                    background_eliminated_image[k,l] = 1

    background_eliminated_image = cv2.resize(background_eliminated_image, (int(unedited_image.shape[1] / 2), int(unedited_image.shape[0] / 2)))
        # cv2.imwrite('results/resize' + str(i) + '.tif', background_and_stationary_eliminated_image)


    canny = unedited_image.copy()
    canny = cv2.bilateralFilter(canny, 9, 150, 150)
    canny = cv2.Canny(canny, canny_param_1, canny_param_2)
    canny = cv2.resize(canny, (int(canny.shape[1] / 2), int(canny.shape[0] / 2)))
    cv2.imwrite('results/'+naming+'/canny/' + str(i) + '.tif', canny)

    edgemerge = cv2.resize(unedited_image.copy(), (int(unedited_image.shape[1] / 2), int(unedited_image.shape[0] / 2)))

    for m in range(0, canny.shape[0]):
        for n in range(0, canny.shape[1]):
            if background_eliminated_image[m][n] == 0 or canny[m][n] == 0:
                edgemerge[m][n] = 0
            else:
                edgemerge[m][n] = 255

    cv2.imwrite('results/'+naming+'/edgemerge/' + str(i) + '.tif', edgemerge)

    # connected = cv2.imread('results/' + naming + '/edgemerge/' + str(i) + '.tif', cv2.IMREAD_GRAYSCALE)
    connected = connect_close_edges(edgemerge, max_edge_connect_dist,
                        i)  # connect points with other points that are in 12*12 window in positive axis
    cv2.imwrite('results/' + naming + '/connected/' + str(i) + '.tif', connected)

    filled = fill_holes(connected)  # connecting edges may create contours with empty holes in them. this function fill these holes.
    # # are not cells and should be removed
    cv2.imwrite('results/' + naming + '/filled_1/' + str(i) + '.tif', filled)
    #
    # filled = cv2.imread('results/' + naming + '/filled_1/' + str(i) + '.tif', cv2.IMREAD_GRAYSCALE)
    big_only = eliminate_small_objects(filled, min_size)  # connecting close edges creates contours. Small contours means they
    # #
    cv2.imwrite('results/' + naming + '/big_only/' + str(i) + '.tif', big_only)

    final_result = cv2.resize(big_only, (int(unedited_image.shape[1]), int(unedited_image.shape[0])))
    cv2.imwrite('results/' + naming + '/final_result/' + str(i) + '.tif', final_result)
    # connect_close_edges('results/big_only' + str(i) + '.tif', max_edge_connect_dist,
    #                     i)  # connect points with other points that are in 12*12 window in positive axis
    # fill_holes('results/connected' + str(i) + '.tif', "result",
    #            i)  # connecting edges may create contours with empty holes in them. this function fill these holes.
    # # are not cells and should be removed
    #
    # small_result = cv2.imread('results/filled' + str(i) + '_result.tif', cv2.IMREAD_GRAYSCALE)
    # result = cv2.resize(small_result, (int(unedited_image.shape[1]), int(unedited_image.shape[0])))
    # cv2.imwrite('results/result' + str(i) + '.tif', result)

    print(str(i) + " is finished")

############################################
########## start of execution ##############
# points = detect_stationary('glassmatrigel/01/frame28.tif','glassmatrigel/01/frame493.tif')


# for i in range(0,115):
#     file_init = 'PhC/01/t'       # 'training/train0'
#     extra_0 = ""  # to adapt naming convention
#     if i<10:
#         extra_0 = "00"
#     elif i<100:
#         extra_0 = "0"
#     original_image_path = file_init+extra_0+str(i)+'.tif'
#     do_segmentation(original_image_path,"PhC-01",i,True,"PhC/01/t017.tif","PhC/01/t114.tif")

# for i in range(1,15):
#     original_image_path = 'glassmatrigel/01rename/frame'+str(i)+'.tif'
#     do_segmentation(original_image_path,"glassmatrigel-01",i,False,None,None)

# for i in range(0,115):
#     file_init = 'PhC/02/t'       # 'training/train0'
#     extra_0 = ""  # to adapt naming convention
#     if i<10:
#         extra_0 = "00"
#     elif i<100:
#         extra_0 = "0"
#     original_image_path = file_init+extra_0+str(i)+'.tif'
#     do_segmentation(original_image_path,"PhC-02",i,True,"PhC/02/t077.tif","PhC/02/t114.tif")

for i in range(1,35):
    original_image_path = 'glassmatrigel/02rename/Frame'+str(i)+'.tif'
    do_segmentation(original_image_path,"glassmatrigel-02",i,False,None,None)
