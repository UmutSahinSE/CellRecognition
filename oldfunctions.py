# def point_in_list(target, list):
#     for i in list:
#         if i[0] == target[0] and i[1] == target[1]:
#             return True
#     return False
#
# def find_center(list):
#     cluster_groups = []
#     while len(list) > 0:
#         i = list.pop()
#         new_group = []
#         for j in list:
#             if pow(i[0] - j[0], 2) + pow(i[1] - j[1], 2) < 5000:
#                 list.remove(j)
#                 new_group.append(j)
#         new_group.append(i)
#         cluster_groups.append(new_group)
#
#     means = []
#
#     for i in cluster_groups:
#         sumx = 0
#         sumy = 0
#         for j in i:
#             sumx += j[0]
#             sumy += j[1]
#         means.append([sumx / len(i), sumy / len(i)])
#
#     return means

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