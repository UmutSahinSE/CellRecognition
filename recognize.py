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

    # images = []
    # l = ""
    # if 10 > image_index:
    #     l = "0" + str(image_index)
    # else:
    #     l = str(image_index)
    # original_image = cv2.imread(file_init+l+'.tif',cv2.IMREAD_GRAYSCALE)
    # k = int(image_index / 10)
    # for i in range(1,3):
    #     l = ""
    #     if 10 > image_index+i:
    #         l = "0" + str(image_index+i)
    #     else:
    #         l = str(image_index+i)
    #     print(l)
    #     images.append(cv2.imread(file_init+l+'.tif',cv2.IMREAD_GRAYSCALE))
    #
    # for i in range(1, int((image.shape[0]-1)/3)):
    #     for j in range(1, int((image.shape[1]-1)/3)):
    #         later_sum_err = 0
    #         current_sum_err = 0
    #         for k in range(3):
    #             for l in range(3):
    #                 later_sum_err += abs(int(images[0][i*3+k][j*3+l]) - int(images[1][i*3+k][j*3+l]))
    #                 current_sum_err += abs(int(images[0][i * 3 + k][j * 3 + l]) - int(original_image[i * 3 + k][j * 3 + l]))
    #         if later_sum_err < 40 and current_sum_err < 40:
    #             for k in range(3):
    #                 for l in range(3):
    #                     if image[i * 3 + k][j * 3 + l] == 255:
    #                         image[i * 3 + k][j * 3 + l] = 90
    # def pattern_match(path,window_size):
    #     image = cv2.imread(path)
    #     print(image.shape)
    #     template = image[0:window_size, 0:window_size]
    #     cv2.imwrite('template.tif', template)
    #     template = cv2.imread('template.tif')
    #
    #     img = image.copy()
    #
    #     method = eval('cv2.TM_SQDIFF_NORMED')
    #
    #     # Apply template Matching
    #     res = cv2.matchTemplate(img, template, method)
    #     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    #     # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #         top_left = min_loc
    #     else:
    #         top_left = max_loc
    #     bottom_right = (top_left[0], top_left[1])
    #
    #     cv2.rectangle(img, top_left, bottom_right, 255, 2)
    #     plt.imshow(res, cmap='gray')
    #
    #     plt.gca().set_axis_off()
    #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                         hspace=0, wspace=0)
    #     plt.margins(0, 0)
    #     plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #     plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #
    #     plt.savefig("aaa.tif", bbox_inches='tight', pad_inches=0)


    # for i in range(1, image.shape[0] - 1):
    #     for j in range(1, image.shape[1] - 1):
    #         intensity_levels = []
    #         for k in images:
    #             intensity_levels.append(int(k[i,j]/5))
    #         # mode_value = max(set(intensity_levels), key = intensity_levels.count)
    #         # counter = 0
    #         # for k in intensity_levels:
    #         #     if -1 <= mode_value-k <= 1:
    #         #         counter+=1
    #         if max(intensity_levels) - min(intensity_levels) < 3 and -1 <= intensity_levels[0]-int(original_image[i,j]/5) <= 1 and image[i, j] == 255:
    #             image[i, j] = 90
    #         # min = 256
    #         # max = -1
    #         # sum = 0
    #         # for k in images:
    #         #     if int(k[i,j]) > max:
    #         #         max = int(k[i,j])
    #         #     if int(k[i,j]) < min:
    #         #         min = int(k[i, j])
    #         #     sum += int(k[i,j])
    #         # if max - sum/9 < 5 and sum/9 - min < 5 and image[i,j] == 255:
    #         #     image[i,j] = 90

    cv2.imwrite('bw_image.tif', image)

def connect_close_edges(path,window_size,index):
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
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



    # for i in range(image.shape[0] - window_size):
    #     for j in y_range:
    #         if image[i][j] == 255:
    #             if image[i + 1][j] is 255 or image[i - 1][j] is 255 or image[i + 1][j + positive] is 255 or image[i][
    #                 j + positive] is 255 or image[i - 1][j + positive] is 255:
    #                 continue
    #             least_distance = 9999  # will going to use this operation once per point to avoid unnecessarily long execution time. Finding closest point that is not a neighbor
    #             closest_white = []  # location of point to connect with, relative to original point
    #             for k in range(window_size):
    #                 for l in range(window_size):
    #                     if image[i+k][j+positive*l] == 255:
    #                         if least_distance > math.sqrt(pow(k,2)+pow(l,2)):
    #                             least_distance = math.sqrt(pow(k,2)+pow(l,2))
    #                             closest_white = [k,positive*l]
    #             if least_distance != 9999:
    #                 points = find_points_to_fill(i,j,closest_white)  # Bresenham's_Line_Algorithm to detect points to fill between two points
    #                 for p in points:
    #                     image[p[0],p[1]]=255

    cv2.imwrite('connected'+str(index)+'.tif', image)

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

def eliminate_small_objects(path, min_size, index):
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

    a = morphology.remove_small_objects(a, min_size, connectivity=2)  # remove contours smaller than 350 pixels of area
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if a[i][j] == False:
                image[i][j] = 0  # formatting boolean representation back to intensity representation

    cv2.imwrite('big_only' + str(index) + '.tif', image)

def fill_holes(path, nth, index):
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
    cv2.imwrite('filled' + str(index) + '_'+nth+'.tif', im_out)

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

def detect_stationary(current_image_path,compare_image_path1):
    current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    compare_image1 = cv2.imread(compare_image_path1, cv2.IMREAD_GRAYSCALE)
    # background_template = io.imread("background_template.tif", as_gray=True)
    points = []

    for i in range(1, int((current_image.shape[0])/200)):
        for j in range(1, int((current_image.shape[1])/200)):
            template = current_image[i*200:i*200+200, j*50:j*200+200]
            # mse_res = mse(template,background_template)
            # if mse_res>6000:
                # io.imshow(template)
                # plt.show()
            res = cv2.matchTemplate(compare_image1, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            if max_val > 0.1 and abs(max_loc[0]-j*200)<200 and abs(max_loc[1] -i*200)<200 :
                print(max_loc)
                print(str(j*20) + "," + str(i*20))
                points.append([max_loc[0],max_loc[1],template])
    return points

def test_stationary(current_image_path,points):
    current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
    for p in points:
        res = cv2.matchTemplate(current_image, p[2], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        if max_val > 0.1 and abs(max_loc[0] - p[0]) < 200 and abs(max_loc[1] - p[1]) < 200:
            for i in range(200):
                for j in range(200):
                    current_image[p[1]+i,p[0]+j]=0
    cv2.imwrite(current_image_path[22:24] + 'bbb.tif', current_image)
    #
    #         img = image.copy()
    #
    #         method = eval('cv2.TM_SQDIFF_NORMED')
    #
    #         # Apply template Matching
    #         res = cv2.matchTemplate(img, template, method)
    #         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    #
    #         # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    #         if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #             top_left = min_loc
    #         else:
    #             top_left = max_loc
    #         bottom_right = (top_left[0], top_left[1])
    #
    #         cv2.rectangle(img, top_left, bottom_right, 255, 2)
    #         plt.imshow(res, cmap='gray')
    #
    #         plt.gca().set_axis_off()
    #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0,
    #                             hspace=0, wspace=0)
    #         plt.margins(0, 0)
    #         plt.gca().xaxis.set_major_locator(plt.NullLocator())
    #         plt.gca().yaxis.set_major_locator(plt.NullLocator())
    #
    #         plt.savefig("aaa.tif", bbox_inches='tight', pad_inches=0)

def detect_background(current_image,index):
    # background_template = cv2.imread("background_template.tif", cv2.IMREAD_GRAYSCALE)
    # points = []
    # print(background_template[0,0])
    threshold = 5
    if index < 10:
        threshold = 3
    else:
        threshold = 3
    for i in range(0, int((current_image.shape[0])/10)):
        for j in range(0, int((current_image.shape[1])/10)):
            if j == int((current_image.shape[1])/10)-1:
                template = current_image[i * 10:i * 10 + 10, j * 10:]
            else:
                template = current_image[i*10:i*10+10, j*10:j*10+10]
            mins = []
            maxes = []
            for m in range(len(template)):
                mins.append(min(template[m]))
                maxes.append(max(template[m]))
            if max(maxes) - min(mins) < threshold:
                for m in range(len(template)):
                    template[m] = [0] * len(template[m])

    template = current_image[i*10:]
    for m in range(len(template)):
        template[m] = [0] * len(template[m])

    cv2.imwrite('back'+str(index)+'.tif', current_image)
    return current_image



############################################
########## start of execution ##############
# points = detect_stationary('glassmatrigel/01/frame28.tif','glassmatrigel/01/frame493.tif')

for i in range(14,15):
    file_init = 'glassmatrigel/01rename/frame'       # 'training/train0'
    extra_0 = ""  # to adapt naming convention
    # if i<10:
    #     extra_0 = "0"
    original_image = file_init+extra_0+str(i)+'.tif'
    # test_stationary(original_image,points)
    unedited_image = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)

    if i >= 8:
        back = unedited_image.copy()
        # image = cv2.GaussianBlur(image,(3,3),0)
        back = cv2.bilateralFilter(back, 9, 150, 150)
        back = detect_background(back,i)

        canny = unedited_image.copy()
        canny = cv2.bilateralFilter(canny, 9, 150, 150)
        canny = cv2.Canny(canny, 0, 5)
        cv2.imwrite('canny' + str(i) + '.tif', canny)

        for m in range(unedited_image.shape[0]):
            for n in range(unedited_image.shape[1]):
                if back[m,n] == 0 or canny[m,n] == 0:
                    unedited_image[m,n] = 0
                else:
                    unedited_image[m,n] = 255

        cv2.imwrite('edgemerge' + str(i) + '.tif', unedited_image)

        connect_close_edges('edgemerge' + str(i) + '.tif', 40,
                            i)  # connect points with other points that are in 12*12 window in positive axis

        # eliminate_small_objects('connected' + str(i) + '.tif', 1000,
        #                         i)  # connecting close edges creates contours. Small contours means they
        #
        # # are not cells and should be removed
        #
        # fill_holes('big_only' + str(i) + '.tif', "1",
        #            i)  # connecting edges may create contours with empty holes in them. this function fill these holes.

    elif 8 > i > 4:
        back = unedited_image.copy()
        # image = cv2.GaussianBlur(image,(3,3),0)
        back = cv2.bilateralFilter(back, 9, 150, 150)
        back = detect_background(back,i)
        #
        canny = unedited_image.copy()
        canny = cv2.bilateralFilter(canny, 9, 150, 150)
        canny = cv2.Canny(canny, 0, 6)
        cv2.imwrite('canny' + str(i) + '.tif', canny)

        for m in range(0, unedited_image.shape[0]):
            for n in range(0, unedited_image.shape[1]):
                if back[m][n] == 0 or canny[m][n] == 0:
                    unedited_image[m][n] = 0
                else:
                    unedited_image[m][n] = 255

        cv2.imwrite('edgemerge' + str(i) + '.tif', unedited_image)

        connect_close_edges('edgemerge' + str(i) + '.tif', 40,
                            i)  # connect points with other points that are in 12*12 window in positive axis

        # eliminate_small_objects('connected' + str(i) + '.tif', 1000,
        #                         i)  # connecting close edges creates contours. Small contours means they
        #
        # # are not cells and should be removed
        #
        # fill_holes('big_only' + str(i) + '.tif', "1",
        #            i)  # connecting edges may create contours with empty holes in them. this function fill these holes.

    elif i <= 4:
        back = unedited_image.copy()
        # image = cv2.GaussianBlur(image,(3,3),0)
        back = cv2.bilateralFilter(back, 9, 150, 150)
        back = detect_background(back, i)

        canny = unedited_image.copy()
        canny = cv2.bilateralFilter(canny, 9, 150, 150)
        canny = cv2.Canny(canny, 0, 10)
        cv2.imwrite('canny' + str(i) + '.tif', canny)

        for m in range(0, unedited_image.shape[0]):
            for n in range(0, unedited_image.shape[1]):
                if back[m][n] == 0 or canny[m][n] == 0:
                    unedited_image[m][n] = 0
                else:
                    unedited_image[m][n] = 255

        cv2.imwrite('edgemerge' + str(i) + '.tif', unedited_image)

        connect_close_edges('edgemerge' + str(i) + '.tif', 20,
                            i)  # connect points with other points that are in 12*12 window in positive axis

        eliminate_small_objects('connected' + str(i) + '.tif', 1000,
                                i)  # connecting close edges creates contours. Small contours means they

        # are not cells and should be removed

        fill_holes('big_only' + str(i) + '.tif', "1",
                   i)  # connecting edges may create contours with empty holes in them. this function fill these holes.

    # image = Image.open(original_image)  # original_image[15:17] + 'bbb.tif'
    # image = image.filter(ImageFilter.SMOOTH_MORE)
    # image = image.filter(ImageFilter.FIND_EDGES)  # find edges of image
    # image.save('pil_edges.tif')
    #
    # brightness_range_filter('edgemerge' + str(i) + '.tif',6,255,i,file_init) # threshold edges in image and make it binary



    # convex_hull("filled1.tif")  # convex hull algorithm wraps the contours so that all of them are convex objects.
    # eliminate_convex_contours("filled1.tif","convex.tif",35)  # compares original contours with convex_filled ones and

    # # separates contours that gained more than 35% area as result of convex_hull operation

    # connect_close_edges("remove_convex.tif", 50)  # when relatively convex objects are separated, connecting edges

    # # can be done with a larger window size without accidentally combining two cells

    # fill_holes("connected.tif", "2")  # filling holes in new contours
    # combine_two_images("filled1.tif","filled2.tif", "segresult_test0"+extra_0+str(i)+".tif")

    # # merging previously separated relatively convex enough contours with newly reformed relatively concave contours
