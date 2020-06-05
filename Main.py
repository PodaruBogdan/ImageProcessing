<<<<<<< HEAD
import cv2
import numpy as np
import math

trackbars = None
#  all were 0 before
bottom_mask_margin = 0#51
top_mask_margin = 0
high_base_width = 0#473
low_base_width = 0#103
trap_height = 0#70
#  ----
a = 0#3
b = 1#473
last_arrow = ""
a11 = 0
a12 = 0
b11 = 0
b12 = 0

a21 = 0
a22 = 0
b21 = 0
b22 = 0


def on_a(val):
    global a
    a = val


def on_b(val):
    global b
    b = val


def on_bottom_trackbar(val):
    global bottom_mask_margin
    bottom_mask_margin = val


def on_high_trackbar(val):
    global high_base_width
    high_base_width = val


def on_low_trackbar(val):
    global low_base_width
    low_base_width = val


def on_height_trackbar(val):
    global trap_height
    trap_height = val


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 2 / 3)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines
    height, width, _ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, 0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
    right_fit_average = np.average(right_fit, 0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    return lane_lines


def roi(edges):
    ignore_mask_color = (255,) * 3
    mask = np.zeros_like(edges)
    height, width = edges.shape
    low_base = int(float(low_base_width) / 1000 * width)
    high_base = int(float(high_base_width) / 1000 * width)
    dy = int(float(trap_height) / 100 * height)
    vertices = np.array([[(low_base, height - bottom_mask_margin),
                          (high_base, dy),
                          (width - high_base, dy),
                          (width - low_base, height - bottom_mask_margin)]])
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 1 / 2 + trap_height),
        (width, height * 1 / 2 + trap_height),
        (width, height - bottom_mask_margin),
        (0, height - bottom_mask_margin),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges


def discard_color(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(im, 50, 150)


def isolate_white_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #  lift out all white colors
    # 168
    lower_white = np.array([0, 0, 168])
    # 30->111
    upper_white = np.array([172, 111, 255])
    #  render white mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask


def isolate_white_and_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)
    return mask_yw_image


def applySobelFilter(image, kernel_size):
    image_sobel = cv2.Sobel(image, cv2.CV_8U, 1, 0, kernel_size)
    return image_sobel


def remove_noise(image, kernel_size, iterations):
    gauss = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    for i in range(iterations):
        gauss = cv2.GaussianBlur(gauss, (kernel_size, kernel_size), 0)
    return gauss


def detect_edges(mask, low_threshold, upper_threshold):
    # try with 200-400
    edges = cv2.Canny(mask, low_threshold, upper_threshold)
    return edges


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    line_segments = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    global a11, a12, b11, b12, a21, a22, b21, b22
    _, width, _ = frame.shape
    left_line = None
    right_line = None
    if len(lines) == 2:
        left_line = lines[0]
        right_line = lines[1]
    elif len(lines) == 1:
        x1, _, _, _ = lines[0][0]
        if 0 < x1 < width / 2:
            left_line = lines[0]
        else:
            right_line = lines[0]

    if left_line is not None:
        x1, y1, x2, y2 = left_line[0]
        if a11 != x1 and a12 != x2 and b11 != y1 and b12 != y2:
            a11 = x1
            a12 = x2
            b11 = y1
            b12 = y2
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    else:
        cv2.line(line_image, (a11, b11), (a12, b12), line_color, line_width)

    if right_line is not None:
        x1, y1, x2, y2 = right_line[0]
        if a21 != x1 and a22 != x2 and b21 != y1 and b22 != y2:
            a21 = x1
            a22 = x2
            b21 = y1
            b22 = y2
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    else:
        cv2.line(line_image, (a21, b21), (a22, b22), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def apply_erode(image, kernel_size, iteration):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tmp = cv2.erode(image, kernel)
    for i in range(iteration):
        tmp = cv2.erode(tmp, kernel)
    return tmp


def apply_dilate(image, kernel_size, iteration):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tmp = cv2.dilate(image, kernel)
    for i in range(iteration):
        tmp = cv2.dilate(tmp, kernel)
    return tmp


def enhance_vertical(image, iterations):
    # kernel = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    # kernel = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]
    # kernel = [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]
    kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    # kernel = [[0,  1,  2], [-1,  0,  1], [-2, -1,  0]]
    kernel = np.asanyarray(kernel, np.float32)
    filtered = cv2.filter2D(image, -1, kernel)
    for i in range(iterations):
        filtered = cv2.filter2D(filtered, -1, kernel)
    return filtered


def pipeline2():
    cap = cv2.VideoCapture('road_video.mp4')
    while (cap.isOpened()):
        _, frame = cap.read()
        #  remove noise
        gauss = remove_noise(frame, 5, 2)
        #  lift out white color
        mask = isolate_white_lines(gauss)
        #   get rid of small particles
        erode = apply_erode(mask, 5, 0)
        #  enhance vertical lines => lane lines
        enhanced = enhance_vertical(erode, 1)
        #  fit lines through points
        canny = detect_edges(enhanced, 50, 150)
        #   get region of interest => trapezoid
        cropped_edges = roi(canny)
        #  obtain line segments
        lines = hough_lines(cropped_edges, 1, np.pi / 180, 10, 100, 10)
        #   filter line segments
        lane_lines = average_slope_intercept(frame, lines)
        #   draw final lines over original frame
        complete = display_lines(frame, lane_lines, (0, 255, 0), 3)

        cv2.imshow('frame', complete)
        cv2.imshow('original', cropped_edges)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def median_filter(image, kernel_size, iterations):
    tmp = cv2.medianBlur(image, kernel_size)
    for i in range(iterations):
        tmp = cv2.medianBlur(tmp, kernel_size)
    return tmp


def do_template_matching(image, templates):
    mscore = 0
    proper = None
    for name, template in templates.items():
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, score, min_loc, max_loc = cv2.minMaxLoc(res)
        if score > mscore:
            mscore = score
            proper = name
    if mscore > 0.25:
        print(str(score) + ' ' + proper)
    return (score, proper)


def get_template_roi(image):
    #  print(image.shape)
    height, width = image.shape
    cropped_edges = image[0:height, 150:width - 150]
    return cropped_edges


def get_template2(image):
    gauss = remove_noise(image, 5, 11)
    mask = isolate_white_lines(gauss)
    complete = cv2.resize(mask, (400, 80))  # Resize image
    # complete = cv2.cvtColor(complete, cv2.COLOR_BGR2GRAY)
    complete = get_template_roi(complete)
    cv2.imshow("t", complete)
    cv2.waitKey(0)
    return complete


def get_template(image):
    gauss = remove_noise(image, 5, 11)
    mask = isolate_white_lines(gauss)
    edges = detect_edges(mask, 50, 150)
    lines = hough_lines(edges, 1, np.pi / 180, 10, 5, 30)
    complete = line_image(image, lines, (255, 255, 255))
    complete = apply_dilate(complete, 3, 5)
    # 160 170
    complete = cv2.resize(complete, (400, 80))  # Resize image
    complete = cv2.cvtColor(complete, cv2.COLOR_BGR2GRAY)
    complete = get_template_roi(complete)
    # cv2.imshow('template',complete)
    # cv2.waitKey(0)
    return complete


def do_shape_matching(frame, templates):
    min_score1 = 100
    min_score2 = 100
    min_score3 = 100
    proper = None
    ret, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)

    for name, template in templates.items():
        ret, template = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)

        d1 = cv2.matchShapes(frame, template, cv2.CONTOURS_MATCH_I1, 0)
        d2 = cv2.matchShapes(frame, template, cv2.CONTOURS_MATCH_I2, 0)
        d3 = cv2.matchShapes(frame, template, cv2.CONTOURS_MATCH_I3, 0)
        # check for surface to be complete
        print("name= " + name + " d1=" + str(d1) + " d2= " + str(d2) + " d3= " + str(d3))
    print("\n")


def get_arrow_roi_trapezoid(frame):
    height, width = frame.shape
    tmp = frame[(height - 200):(height - 30), (int(width / 2) - 110):(int(width / 2) + 110)]
    height, width = tmp.shape
    mask = np.zeros_like(tmp)
    polygon = np.array([[
        ((int(width / 2) - 27), 0),
        ((int(width / 2) + 27), 0),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(tmp, mask)
    cv2.imshow('cropped', cropped)
    cv2.waitKey(1)
    return cropped


def get_arrow_roi(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    cropped = frame[(height - 200):(height - 30), (int(width / 2) - 110):(int(width / 2) + 110)]
    cv2.imshow('cropped', cropped)
    cv2.waitKey(1)
    return cropped


def pipeline(templates):
    cap = cv2.VideoCapture('road_video.mp4')
    while (cap.isOpened()):
        _, frame = cap.read()
        #  remove noise
        gauss = remove_noise(frame, 5, 2)
        #  lift out white color
        mask = isolate_white_lines(gauss)
        #  get region of interest => trapezoid
        cropped_edges = roi(mask)
        #  fit lines through points
        canny = detect_edges(cropped_edges, 50, 150)
        #  get roi for arrows
        cropped = get_arrow_roi_trapezoid(cropped_edges)
        #  search arrow in ROI for arrow
        arrow_type = get_arrow_type(cropped)
        #  enhance vertical lines => lane lines
        enhanced = enhance_vertical(canny, 2)
        #  obtain line segments
        lines = hough_lines(enhanced, 1, np.pi / 180, 30, 100, 1)
        #   filter line segments
        lane_lines = average_slope_intercept(frame, lines)
        #   draw final lines over original frame
        complete = display_lines(frame, lane_lines, (0, 255, 0), 3)
        for line in lane_lines:
            complete = cast_scanlines(complete, line, 10, cropped_edges, arrow_type)
        complete = cv2.resize(complete, (500, 400))  # Resize image
        cropped_edges = cv2.resize(cropped_edges, (500, 400))  # Resize image

        cv2.imshow('frame', complete)
        cv2.imshow('original', cropped_edges)
        if cv2.waitKey(2) & 0xFF == ord('s'):
            cv2.imwrite('cropped5.jpg', cropped)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def feature_matching(original, templates):
    mscore = 0
    proper = None
    for name, cropped in templates.items():
        minHessian = 400
        detector = cv2.xfeatures2d_SIFT.create(minHessian)
        keypoints1, descriptors1 = detector.detectAndCompute(original, None)
        keypoints2, descriptors2 = detector.detectAndCompute(cropped, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        ratio_thresh = 0.50
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        img_matches = np.empty((max(original.shape[0], cropped.shape[0]), original.shape[1] + cropped.shape[1], 3),
                               np.uint8)
        cv2.drawMatches(original, keypoints1, cropped, keypoints2, good_matches, img_matches,
                        cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(name, img_matches)
        cv2.waitKey(1)
        if len(good_matches) > mscore:
            mscore = len(good_matches)
            proper = name
        if proper is not None:
            print(str(mscore) + ' ' + proper)


def line_image(frame, lines, color):
    line_image = np.zeros_like(frame)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, 3)
    return line_image


def computeA(src):
    height, width = src.shape
    A = 0
    for r in range(height):
        for c in range(width):
            if src[r, c] > 0:
                A = A + 1
    return A


def centre_of_mass(src):
    height, width = src.shape
    A = computeA(src)
    sum1 = 0
    sum2 = 0
    for r in range(height):
        for c in range(width):
            if src[r, c] > 0:
                sum1 = sum1 + r
                sum2 = sum2 + c
    if A == 0:
        return (0, 0)
    else:
        return (sum1 / A, sum2 / A)


def elongation_angle(src):
    ret, src = cv2.threshold(src, 1, 255, cv2.THRESH_BINARY)
    height, width = src.shape
    nom = 0
    denom1 = 0
    denom2 = 0
    (ri, ci) = centre_of_mass(src)
    if ri != 0 and ci != 0:
        for r in range(height):
            for c in range(width):
                if src[r, c] > 0:
                    nom = nom + (r - ri) * (c - ci)
                    denom1 = denom1 + (c - ci) * (c - ci)
                    denom2 = denom2 + (r - ri) * (r - ri)
        phi = (math.atan2(2.0 * nom, (denom1 - denom2))) / 2.0
        print("phi= " + str(phi))
        if phi < 0:
            angle = 180.0 - abs(phi * 180.0 / math.pi)
        else:
            angle = phi * 180.0 / math.pi
        print(angle)
        if angle > 90:
            print("straight_right")
        elif angle < 90:
            print("straight_left")
        else:
            print("straight")
        return angle


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors = cv2.PCACompute(data_pts, mean)
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]) + math.pi  # orientation in radians
    print(angle)
    return angle


def something_complete(src, d1, d2):
    height, width = src.shape
    up = 0
    down = 0
    foundUp = False
    foundDown = False
    for r in range(height):
        for c in range(width):
            if src[r, c] == 255:
                foundUp = True
                if r >= d1:
                    up = r
                break
        if foundUp:
            break

    for r in range(height - 1, 0, -1):
        for c in range(width):
            if src[r, c] == 255:
                foundDown = True
                if height - 1 - r >= d2:
                    down = r
                break
        if foundDown:
            break
    if up != 0 and down != 0 and down - up >= 70:
        return True
    else:
        return False


def cond(src, r, c):
    height, width = src.shape
    if r + 1 > height or r - 1 < 0 or c + 1 > width or c - 1 < 0:
        return False
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    if src[r, c] == 255:
        for i in range(4):
            a = r + dx[i]
            b = c + dy[i]
            if not (a >= height or a < 0 or b >= width or b < 0):
                if src[a, b] == 255:
                    return True
    return False


def find_extreme_points(src):
    height, width = src.shape
    top = (0, 0)
    left = (0, width)
    right = (0, width)
    foundTop = False
    max = width
    if something_complete(src, 10, 10):
        r = 0
        while r < height - 1:
            for c in range(width):
                if src[r, c] > 0 and foundTop == False:
                    top = (r, c)
                    r = r + 1
                    foundTop = True
                    break
            for c1 in range(width):
                if cond(src, r, c1):
                    _, a = left
                    if c1 < a:
                        left = (r, c1)
            for c2 in range(width - 1, 0, -1):
                if cond(src, r, c2):
                    if width - c2 - 1 < max:
                        max = width - c2 - 1
                        right = (r, c2)
            r = r + 1
    return [top, left, right]


def verify_equation(p1, p2, pa, pb):
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = pa
    (x4, y4) = pb
    vala = (y1 - y2) * x3 + (x2 - x1) * y3 + (x1 * y2 - x2 * y1)
    valb = (y1 - y2) * x4 + (x2 - x1) * y4 + (x1 * y2 - x2 * y1)
    if vala < 0:
        a = -1
    else:
        a = 1
    if valb < 0:
        b = -1
    else:
        b = 1
    return (a, b)


def takeFirst(elem):
    return elem[0]


def d(_lambda, x):
    if x < 0:
        return 0
    else:
        return _lambda * math.exp(-_lambda * x)


def d2(x, a, b):
    return math.exp(a - x / b)


def length(line):
    x1, y1, x2, y2 = line[0]
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def get_next_point(m, p, sample):
    (x0, y0) = p
    tmp = math.sqrt(sample * sample / (m * m + 1))
    if m < 0:
        x1 = x0 + tmp
    else:
        x1 = x0 - tmp
    y1 = m * (x1 - x0) + y0
    return (x1, y1)


def get_slope(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return (y2 - y1) / (x2 - x1)


def dashed(img, points, type):
    encode = []

    height, width = img.shape
    for point in points:
        (x, y) = point
        x = int(x)
        y = int(y)
        found = False
        for i in range(-15, 15, 1):
            if x + i >= 0 and x + i < width and y >= 0 and y < height:
                if img[y, x + i] == 255:
                    found = True
                    encode.append(0)
                    break
        if found == False:
            encode.append(1)
    sum = 0
    for i in range(len(encode) - 1):
        if encode[i] != encode[i + 1]:
            sum = sum + 1
    if sum > 3:
        return "DASHED"
    else:
        return "SOLID"


def cast_scanlines(img, line, step, cropped, arrow_type):
    (height, width, _) = img.shape
    # d is distance between scanlines
    # it should decrease with adancing the lane
    # probability density function
    l1 = length(line)
    x1, y1, x2, y2 = line[0]
    if y1 > y2:
        p1 = (x1, y1)
        p2 = (x2, y2)
    else:
        p1 = (x2, y2)
        p2 = (x1, y1)
    if x1 > 0 and x1 < width / 2:
        type = "LEFT"
    else:
        type = "RIGHT"
    m1 = get_slope(p1, p2)
    pa = p1
    global a
    global b
    scanline_points = [pa]
    for i in range(0, int(l1), step):
        sample = d2(i, a, b)
        (xm, ym) = get_next_point(m1, pa, sample)
        pa = (xm, ym)
        xm = int(xm)
        ym = int(ym)
        if y1 >= ym >= y2 or y2 >= ym >= y1:
            scanline_points.append(pa)
            img = cv2.line(img, (xm - 20, ym), (xm + 20, ym), (0, 0, 255), 2)
    val = dashed(cropped, scanline_points, type)
    tmp_img = img.copy()
    if type == "RIGHT":
        cv2.putText(
            tmp_img,
            type + " : " + val,
            (1000, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0, 255),
            3)
    else:
        cv2.putText(
            tmp_img,
            type + " : " + val,
            (80, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0, 255),
            3)
    global last_arrow
    if last_arrow != arrow_type and arrow_type != None and arrow_type != "":
        last_arrow = arrow_type
    cv2.putText(
        tmp_img,
        last_arrow,
        (520, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0, 255),
        3)
    return tmp_img


def get_arrow_type(src):
    blur = cv2.GaussianBlur(src, (3, 3), 0.3)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    img = thresh
    if something_complete(img, 45, 25):
        mat = np.argwhere(img != 0)
        mat[:, [0, 1]] = mat[:, [1, 0]]
        mat = np.array(mat).astype(np.float32)
        m, e = cv2.PCACompute(mat, mean=np.array([]))
        center2 = tuple(m[0])
        endpoint1 = tuple(m[0] + e[0] * 100)  # major
        endpoint2 = tuple(m[0] + e[1] * 50)
        for center in find_extreme_points(img):
            a, b = center
            center = b, a
            cv2.circle(img, center, 5, 128)
        cv2.circle(img, center2, 5, 128)
        cv2.line(img, center2, endpoint1, 128)
        cv2.line(img, center2, endpoint2, 128)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        list_points = find_extreme_points(src)
        list_points.sort(key=takeFirst)
        (c1, c2) = center2
        (s1, s2) = list_points[0]
        (e1, e2) = endpoint1
        v1 = (s1 - c1, s2 - c2)
        v2 = (e1 - c1, e2 - c2)
        v11, v12 = v1
        v21, v22 = v2
        cos_alpha = (v11 * v21 + v12 * v22) / (math.sqrt(v11 * v11 + v12 * v12) * math.sqrt(v21 * v21 + v22 * v22))
        alpha = math.acos(cos_alpha) * 180 / math.pi
        print(alpha)

        if alpha > 90:
            val = "STRAIGHT RIGHT"
        elif 90 > alpha > 40:
            val = "STRAIGHT"
        else:
            val = "STRAIGHT LEFT"
        return val


def main():
    global trackbars
    trackbars = np.zeros((400, 400, 3), np.uint8)
    cv2.imshow('trackbars', trackbars)
    cv2.namedWindow('trackbars')
    cv2.createTrackbar('bm_margin', 'trackbars', 0, 100, on_bottom_trackbar)
    cv2.createTrackbar('hb_width', 'trackbars', 0, 1000, on_high_trackbar)
    cv2.createTrackbar('lb_width', 'trackbars', 0, 1000, on_low_trackbar)
    cv2.createTrackbar('trap_height', 'trackbars', 0, 100, on_height_trackbar)
    cv2.createTrackbar('a', 'trackbars', 0, 10, on_a)
    cv2.createTrackbar('b', 'trackbars', 1, 1000, on_b)

    straight = cv2.imread('arrows/straight.jpg')
    left = cv2.imread('arrows/left.jpg')
    right = cv2.imread('arrows/right.jpg')
    straight_left = cv2.imread('arrows/straight_left.jpg')
    straight_right = cv2.imread('arrows/straight_right.jpg')
    templates = {"straight": get_template(straight),
                 # "left":get_template(left),
                 # "right":get_template(right),
                 "straight_left": get_template(straight_left),
                 "straight_right": get_template(straight_right)}
    pipeline(templates)
    #cropped1 = cv2.imread("cropped3.jpg")
    #feature_matching(cropped1,templates)
    #cv2.waitKey(0)


if __name__ == "__main__":
    main()
=======
import cv2
import numpy as np
import math

trackbars = None
#  all were 0 before
bottom_mask_margin = 0#51
top_mask_margin = 0
high_base_width = 0#473
low_base_width = 0#103
trap_height = 0#70
#  ----
a = 0#3
b = 1#473
last_arrow = ""
a11 = 0
a12 = 0
b11 = 0
b12 = 0

a21 = 0
a22 = 0
b21 = 0
b22 = 0


def on_a(val):
    global a
    a = val


def on_b(val):
    global b
    b = val


def on_bottom_trackbar(val):
    global bottom_mask_margin
    bottom_mask_margin = val


def on_high_trackbar(val):
    global high_base_width
    high_base_width = val


def on_low_trackbar(val):
    global low_base_width
    low_base_width = val


def on_height_trackbar(val):
    global trap_height
    trap_height = val


def make_points(frame, line):
    height, width, _ = frame.shape
    slope, intercept = line
    y1 = height
    y2 = int(y1 * 2 / 3)
    x1 = max(-width, min(2 * width, int((y1 - intercept) / slope)))
    x2 = max(-width, min(2 * width, int((y2 - intercept) / slope)))
    return [[x1, y1, x2, y2]]


def average_slope_intercept(frame, line_segments):
    lane_lines = []
    if line_segments is None:
        return lane_lines
    height, width, _ = frame.shape
    left_fit = []
    right_fit = []
    boundary = 1 / 3
    left_region_boundary = width * (1 - boundary)
    right_region_boundary = width * boundary
    for line_segment in line_segments:
        for x1, y1, x2, y2 in line_segment:
            if x1 == x2:
                continue
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]
            if slope < 0:
                if x1 < left_region_boundary and x2 < left_region_boundary:
                    left_fit.append((slope, intercept))
            else:
                if x1 > right_region_boundary and x2 > right_region_boundary:
                    right_fit.append((slope, intercept))
    left_fit_average = np.average(left_fit, 0)
    if len(left_fit) > 0:
        lane_lines.append(make_points(frame, left_fit_average))
    right_fit_average = np.average(right_fit, 0)
    if len(right_fit) > 0:
        lane_lines.append(make_points(frame, right_fit_average))
    return lane_lines


def roi(edges):
    ignore_mask_color = (255,) * 3
    mask = np.zeros_like(edges)
    height, width = edges.shape
    low_base = int(float(low_base_width) / 1000 * width)
    high_base = int(float(high_base_width) / 1000 * width)
    dy = int(float(trap_height) / 100 * height)
    vertices = np.array([[(low_base, height - bottom_mask_margin),
                          (high_base, dy),
                          (width - high_base, dy),
                          (width - low_base, height - bottom_mask_margin)]])
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(edges, mask)
    return masked_image


def region_of_interest(edges):
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[
        (0, height * 1 / 2 + trap_height),
        (width, height * 1 / 2 + trap_height),
        (width, height - bottom_mask_margin),
        (0, height - bottom_mask_margin),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped_edges = cv2.bitwise_and(edges, mask)
    return cropped_edges


def discard_color(image):
    im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(im, 50, 150)


def isolate_white_lines(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #  lift out all white colors
    # 168
    lower_white = np.array([0, 0, 168])
    # 30->111
    upper_white = np.array([172, 111, 255])
    #  render white mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    return mask


def isolate_white_and_yellow(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(gray, 200, 255)
    mask_yw = cv2.bitwise_or(mask_white, mask_yellow)
    mask_yw_image = cv2.bitwise_and(gray, mask_yw)
    return mask_yw_image


def applySobelFilter(image, kernel_size):
    image_sobel = cv2.Sobel(image, cv2.CV_8U, 1, 0, kernel_size)
    return image_sobel


def remove_noise(image, kernel_size, iterations):
    gauss = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    for i in range(iterations):
        gauss = cv2.GaussianBlur(gauss, (kernel_size, kernel_size), 0)
    return gauss


def detect_edges(mask, low_threshold, upper_threshold):
    # try with 200-400
    edges = cv2.Canny(mask, low_threshold, upper_threshold)
    return edges


def hough_lines(image, rho, theta, threshold, min_line_len, max_line_gap):
    line_segments = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), min_line_len, max_line_gap)
    return line_segments


def display_lines(frame, lines, line_color=(0, 255, 0), line_width=2):
    line_image = np.zeros_like(frame)
    global a11, a12, b11, b12, a21, a22, b21, b22
    _, width, _ = frame.shape
    left_line = None
    right_line = None
    if len(lines) == 2:
        left_line = lines[0]
        right_line = lines[1]
    elif len(lines) == 1:
        x1, _, _, _ = lines[0][0]
        if 0 < x1 < width / 2:
            left_line = lines[0]
        else:
            right_line = lines[0]

    if left_line is not None:
        x1, y1, x2, y2 = left_line[0]
        if a11 != x1 and a12 != x2 and b11 != y1 and b12 != y2:
            a11 = x1
            a12 = x2
            b11 = y1
            b12 = y2
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    else:
        cv2.line(line_image, (a11, b11), (a12, b12), line_color, line_width)

    if right_line is not None:
        x1, y1, x2, y2 = right_line[0]
        if a21 != x1 and a22 != x2 and b21 != y1 and b22 != y2:
            a21 = x1
            a22 = x2
            b21 = y1
            b22 = y2
        cv2.line(line_image, (x1, y1), (x2, y2), line_color, line_width)
    else:
        cv2.line(line_image, (a21, b21), (a22, b22), line_color, line_width)
    line_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return line_image


def apply_erode(image, kernel_size, iteration):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tmp = cv2.erode(image, kernel)
    for i in range(iteration):
        tmp = cv2.erode(tmp, kernel)
    return tmp


def apply_dilate(image, kernel_size, iteration):
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tmp = cv2.dilate(image, kernel)
    for i in range(iteration):
        tmp = cv2.dilate(tmp, kernel)
    return tmp


def enhance_vertical(image, iterations):
    # kernel = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    # kernel = [[-1, 2, -1], [-1, 2, -1], [-1, 2, -1]]
    # kernel = [[-1, -1, -1], [2, 2, 2], [-1, -1, -1]]
    kernel = [[1, 0, -1], [2, 0, -2], [1, 0, -1]]
    # kernel = [[0,  1,  2], [-1,  0,  1], [-2, -1,  0]]
    kernel = np.asanyarray(kernel, np.float32)
    filtered = cv2.filter2D(image, -1, kernel)
    for i in range(iterations):
        filtered = cv2.filter2D(filtered, -1, kernel)
    return filtered


def pipeline2():
    cap = cv2.VideoCapture('road_video.mp4')
    while (cap.isOpened()):
        _, frame = cap.read()
        #  remove noise
        gauss = remove_noise(frame, 5, 2)
        #  lift out white color
        mask = isolate_white_lines(gauss)
        #   get rid of small particles
        erode = apply_erode(mask, 5, 0)
        #  enhance vertical lines => lane lines
        enhanced = enhance_vertical(erode, 1)
        #  fit lines through points
        canny = detect_edges(enhanced, 50, 150)
        #   get region of interest => trapezoid
        cropped_edges = roi(canny)
        #  obtain line segments
        lines = hough_lines(cropped_edges, 1, np.pi / 180, 10, 100, 10)
        #   filter line segments
        lane_lines = average_slope_intercept(frame, lines)
        #   draw final lines over original frame
        complete = display_lines(frame, lane_lines, (0, 255, 0), 3)

        cv2.imshow('frame', complete)
        cv2.imshow('original', cropped_edges)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def median_filter(image, kernel_size, iterations):
    tmp = cv2.medianBlur(image, kernel_size)
    for i in range(iterations):
        tmp = cv2.medianBlur(tmp, kernel_size)
    return tmp


def do_template_matching(image, templates):
    mscore = 0
    proper = None
    for name, template in templates.items():
        res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        min_val, score, min_loc, max_loc = cv2.minMaxLoc(res)
        if score > mscore:
            mscore = score
            proper = name
    if mscore > 0.25:
        print(str(score) + ' ' + proper)
    return (score, proper)


def get_template_roi(image):
    #  print(image.shape)
    height, width = image.shape
    cropped_edges = image[0:height, 150:width - 150]
    return cropped_edges


def get_template2(image):
    gauss = remove_noise(image, 5, 11)
    mask = isolate_white_lines(gauss)
    complete = cv2.resize(mask, (400, 80))  # Resize image
    # complete = cv2.cvtColor(complete, cv2.COLOR_BGR2GRAY)
    complete = get_template_roi(complete)
    cv2.imshow("t", complete)
    cv2.waitKey(0)
    return complete


def get_template(image):
    gauss = remove_noise(image, 5, 11)
    mask = isolate_white_lines(gauss)
    edges = detect_edges(mask, 50, 150)
    lines = hough_lines(edges, 1, np.pi / 180, 10, 5, 30)
    complete = line_image(image, lines, (255, 255, 255))
    complete = apply_dilate(complete, 3, 5)
    # 160 170
    complete = cv2.resize(complete, (400, 80))  # Resize image
    complete = cv2.cvtColor(complete, cv2.COLOR_BGR2GRAY)
    complete = get_template_roi(complete)
    # cv2.imshow('template',complete)
    # cv2.waitKey(0)
    return complete


def do_shape_matching(frame, templates):
    min_score1 = 100
    min_score2 = 100
    min_score3 = 100
    proper = None
    ret, frame = cv2.threshold(frame, 128, 255, cv2.THRESH_BINARY)

    for name, template in templates.items():
        ret, template = cv2.threshold(template, 128, 255, cv2.THRESH_BINARY)

        d1 = cv2.matchShapes(frame, template, cv2.CONTOURS_MATCH_I1, 0)
        d2 = cv2.matchShapes(frame, template, cv2.CONTOURS_MATCH_I2, 0)
        d3 = cv2.matchShapes(frame, template, cv2.CONTOURS_MATCH_I3, 0)
        # check for surface to be complete
        print("name= " + name + " d1=" + str(d1) + " d2= " + str(d2) + " d3= " + str(d3))
    print("\n")


def get_arrow_roi_trapezoid(frame):
    height, width = frame.shape
    tmp = frame[(height - 200):(height - 30), (int(width / 2) - 110):(int(width / 2) + 110)]
    height, width = tmp.shape
    mask = np.zeros_like(tmp)
    polygon = np.array([[
        ((int(width / 2) - 27), 0),
        ((int(width / 2) + 27), 0),
        (width, height),
        (0, height),
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    cropped = cv2.bitwise_and(tmp, mask)
    cv2.imshow('cropped', cropped)
    cv2.waitKey(1)
    return cropped


def get_arrow_roi(frame):
    height = frame.shape[0]
    width = frame.shape[1]
    cropped = frame[(height - 200):(height - 30), (int(width / 2) - 110):(int(width / 2) + 110)]
    cv2.imshow('cropped', cropped)
    cv2.waitKey(1)
    return cropped


def pipeline(templates):
    cap = cv2.VideoCapture('road_video.mp4')
    while (cap.isOpened()):
        _, frame = cap.read()
        #  remove noise
        gauss = remove_noise(frame, 5, 2)
        #  lift out white color
        mask = isolate_white_lines(gauss)
        #  get region of interest => trapezoid
        cropped_edges = roi(mask)
        #  fit lines through points
        canny = detect_edges(cropped_edges, 50, 150)
        #  get roi for arrows
        cropped = get_arrow_roi_trapezoid(cropped_edges)
        #  search arrow in ROI for arrow
        arrow_type = get_arrow_type(cropped)
        #  enhance vertical lines => lane lines
        enhanced = enhance_vertical(canny, 2)
        #  obtain line segments
        lines = hough_lines(enhanced, 1, np.pi / 180, 30, 100, 1)
        #   filter line segments
        lane_lines = average_slope_intercept(frame, lines)
        #   draw final lines over original frame
        complete = display_lines(frame, lane_lines, (0, 255, 0), 3)
        for line in lane_lines:
            complete = cast_scanlines(complete, line, 10, cropped_edges, arrow_type)
        complete = cv2.resize(complete, (500, 400))  # Resize image
        cropped_edges = cv2.resize(cropped_edges, (500, 400))  # Resize image

        cv2.imshow('frame', complete)
        cv2.imshow('original', cropped_edges)
        if cv2.waitKey(2) & 0xFF == ord('s'):
            cv2.imwrite('cropped5.jpg', cropped)
        if cv2.waitKey(2) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def feature_matching(original, templates):
    mscore = 0
    proper = None
    for name, cropped in templates.items():
        minHessian = 400
        detector = cv2.xfeatures2d_SIFT.create(minHessian)
        keypoints1, descriptors1 = detector.detectAndCompute(original, None)
        keypoints2, descriptors2 = detector.detectAndCompute(cropped, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
        ratio_thresh = 0.50
        good_matches = []
        for m, n in knn_matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        img_matches = np.empty((max(original.shape[0], cropped.shape[0]), original.shape[1] + cropped.shape[1], 3),
                               np.uint8)
        cv2.drawMatches(original, keypoints1, cropped, keypoints2, good_matches, img_matches,
                        cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv2.imshow(name, img_matches)
        cv2.waitKey(1)
        if len(good_matches) > mscore:
            mscore = len(good_matches)
            proper = name
        if proper is not None:
            print(str(mscore) + ' ' + proper)


def line_image(frame, lines, color):
    line_image = np.zeros_like(frame)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_image, (x1, y1), (x2, y2), color, 3)
    return line_image


def computeA(src):
    height, width = src.shape
    A = 0
    for r in range(height):
        for c in range(width):
            if src[r, c] > 0:
                A = A + 1
    return A


def centre_of_mass(src):
    height, width = src.shape
    A = computeA(src)
    sum1 = 0
    sum2 = 0
    for r in range(height):
        for c in range(width):
            if src[r, c] > 0:
                sum1 = sum1 + r
                sum2 = sum2 + c
    if A == 0:
        return (0, 0)
    else:
        return (sum1 / A, sum2 / A)


def elongation_angle(src):
    ret, src = cv2.threshold(src, 1, 255, cv2.THRESH_BINARY)
    height, width = src.shape
    nom = 0
    denom1 = 0
    denom2 = 0
    (ri, ci) = centre_of_mass(src)
    if ri != 0 and ci != 0:
        for r in range(height):
            for c in range(width):
                if src[r, c] > 0:
                    nom = nom + (r - ri) * (c - ci)
                    denom1 = denom1 + (c - ci) * (c - ci)
                    denom2 = denom2 + (r - ri) * (r - ri)
        phi = (math.atan2(2.0 * nom, (denom1 - denom2))) / 2.0
        print("phi= " + str(phi))
        if phi < 0:
            angle = 180.0 - abs(phi * 180.0 / math.pi)
        else:
            angle = phi * 180.0 / math.pi
        print(angle)
        if angle > 90:
            print("straight_right")
        elif angle < 90:
            print("straight_left")
        else:
            print("straight")
        return angle


def getOrientation(pts, img):
    sz = len(pts)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i, 0] = pts[i, 0, 0]
        data_pts[i, 1] = pts[i, 0, 1]

    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors = cv2.PCACompute(data_pts, mean)
    cntr = (int(mean[0, 0]), int(mean[0, 1]))
    angle = math.atan2(eigenvectors[0, 1], eigenvectors[0, 0]) + math.pi  # orientation in radians
    print(angle)
    return angle


def something_complete(src, d1, d2):
    height, width = src.shape
    up = 0
    down = 0
    foundUp = False
    foundDown = False
    for r in range(height):
        for c in range(width):
            if src[r, c] == 255:
                foundUp = True
                if r >= d1:
                    up = r
                break
        if foundUp:
            break

    for r in range(height - 1, 0, -1):
        for c in range(width):
            if src[r, c] == 255:
                foundDown = True
                if height - 1 - r >= d2:
                    down = r
                break
        if foundDown:
            break
    if up != 0 and down != 0 and down - up >= 70:
        return True
    else:
        return False


def cond(src, r, c):
    height, width = src.shape
    if r + 1 > height or r - 1 < 0 or c + 1 > width or c - 1 < 0:
        return False
    dx = [-1, 1, 0, 0]
    dy = [0, 0, -1, 1]
    if src[r, c] == 255:
        for i in range(4):
            a = r + dx[i]
            b = c + dy[i]
            if not (a >= height or a < 0 or b >= width or b < 0):
                if src[a, b] == 255:
                    return True
    return False


def find_extreme_points(src):
    height, width = src.shape
    top = (0, 0)
    left = (0, width)
    right = (0, width)
    foundTop = False
    max = width
    if something_complete(src, 10, 10):
        r = 0
        while r < height - 1:
            for c in range(width):
                if src[r, c] > 0 and foundTop == False:
                    top = (r, c)
                    r = r + 1
                    foundTop = True
                    break
            for c1 in range(width):
                if cond(src, r, c1):
                    _, a = left
                    if c1 < a:
                        left = (r, c1)
            for c2 in range(width - 1, 0, -1):
                if cond(src, r, c2):
                    if width - c2 - 1 < max:
                        max = width - c2 - 1
                        right = (r, c2)
            r = r + 1
    return [top, left, right]


def verify_equation(p1, p2, pa, pb):
    (x1, y1) = p1
    (x2, y2) = p2
    (x3, y3) = pa
    (x4, y4) = pb
    vala = (y1 - y2) * x3 + (x2 - x1) * y3 + (x1 * y2 - x2 * y1)
    valb = (y1 - y2) * x4 + (x2 - x1) * y4 + (x1 * y2 - x2 * y1)
    if vala < 0:
        a = -1
    else:
        a = 1
    if valb < 0:
        b = -1
    else:
        b = 1
    return (a, b)


def takeFirst(elem):
    return elem[0]


def d(_lambda, x):
    if x < 0:
        return 0
    else:
        return _lambda * math.exp(-_lambda * x)


def d2(x, a, b):
    return math.exp(a - x / b)


def length(line):
    x1, y1, x2, y2 = line[0]
    return math.sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2))


def get_next_point(m, p, sample):
    (x0, y0) = p
    tmp = math.sqrt(sample * sample / (m * m + 1))
    if m < 0:
        x1 = x0 + tmp
    else:
        x1 = x0 - tmp
    y1 = m * (x1 - x0) + y0
    return (x1, y1)


def get_slope(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return (y2 - y1) / (x2 - x1)


def dashed(img, points, type):
    encode = []

    height, width = img.shape
    for point in points:
        (x, y) = point
        x = int(x)
        y = int(y)
        found = False
        for i in range(-15, 15, 1):
            if x + i >= 0 and x + i < width and y >= 0 and y < height:
                if img[y, x + i] == 255:
                    found = True
                    encode.append(0)
                    break
        if found == False:
            encode.append(1)
    sum = 0
    for i in range(len(encode) - 1):
        if encode[i] != encode[i + 1]:
            sum = sum + 1
    if sum > 3:
        return "DASHED"
    else:
        return "SOLID"


def cast_scanlines(img, line, step, cropped, arrow_type):
    (height, width, _) = img.shape
    # d is distance between scanlines
    # it should decrease with adancing the lane
    # probability density function
    l1 = length(line)
    x1, y1, x2, y2 = line[0]
    if y1 > y2:
        p1 = (x1, y1)
        p2 = (x2, y2)
    else:
        p1 = (x2, y2)
        p2 = (x1, y1)
    if x1 > 0 and x1 < width / 2:
        type = "LEFT"
    else:
        type = "RIGHT"
    m1 = get_slope(p1, p2)
    pa = p1
    global a
    global b
    scanline_points = [pa]
    for i in range(0, int(l1), step):
        sample = d2(i, a, b)
        (xm, ym) = get_next_point(m1, pa, sample)
        pa = (xm, ym)
        xm = int(xm)
        ym = int(ym)
        if y1 >= ym >= y2 or y2 >= ym >= y1:
            scanline_points.append(pa)
            img = cv2.line(img, (xm - 20, ym), (xm + 20, ym), (0, 0, 255), 2)
    val = dashed(cropped, scanline_points, type)
    tmp_img = img.copy()
    if type == "RIGHT":
        cv2.putText(
            tmp_img,
            type + " : " + val,
            (1000, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0, 255),
            3)
    else:
        cv2.putText(
            tmp_img,
            type + " : " + val,
            (80, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0, 255),
            3)
    global last_arrow
    if last_arrow != arrow_type and arrow_type != None and arrow_type != "":
        last_arrow = arrow_type
    cv2.putText(
        tmp_img,
        last_arrow,
        (520, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0, 255),
        3)
    return tmp_img


def get_arrow_type(src):
    blur = cv2.GaussianBlur(src, (3, 3), 0.3)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)
    img = thresh
    if something_complete(img, 45, 25):
        mat = np.argwhere(img != 0)
        mat[:, [0, 1]] = mat[:, [1, 0]]
        mat = np.array(mat).astype(np.float32)
        m, e = cv2.PCACompute(mat, mean=np.array([]))
        center2 = tuple(m[0])
        endpoint1 = tuple(m[0] + e[0] * 100)  # major
        endpoint2 = tuple(m[0] + e[1] * 50)
        for center in find_extreme_points(img):
            a, b = center
            center = b, a
            cv2.circle(img, center, 5, 128)
        cv2.circle(img, center2, 5, 128)
        cv2.line(img, center2, endpoint1, 128)
        cv2.line(img, center2, endpoint2, 128)
        cv2.imshow("image", img)
        cv2.waitKey(0)
        list_points = find_extreme_points(src)
        list_points.sort(key=takeFirst)
        (c1, c2) = center2
        (s1, s2) = list_points[0]
        (e1, e2) = endpoint1
        v1 = (s1 - c1, s2 - c2)
        v2 = (e1 - c1, e2 - c2)
        v11, v12 = v1
        v21, v22 = v2
        cos_alpha = (v11 * v21 + v12 * v22) / (math.sqrt(v11 * v11 + v12 * v12) * math.sqrt(v21 * v21 + v22 * v22))
        alpha = math.acos(cos_alpha) * 180 / math.pi
        print(alpha)

        if alpha > 90:
            val = "STRAIGHT RIGHT"
        elif 90 > alpha > 40:
            val = "STRAIGHT"
        else:
            val = "STRAIGHT LEFT"
        return val


def main():
    global trackbars
    trackbars = np.zeros((400, 400, 3), np.uint8)
    cv2.imshow('trackbars', trackbars)
    cv2.namedWindow('trackbars')
    cv2.createTrackbar('bm_margin', 'trackbars', 0, 100, on_bottom_trackbar)
    cv2.createTrackbar('hb_width', 'trackbars', 0, 1000, on_high_trackbar)
    cv2.createTrackbar('lb_width', 'trackbars', 0, 1000, on_low_trackbar)
    cv2.createTrackbar('trap_height', 'trackbars', 0, 100, on_height_trackbar)
    cv2.createTrackbar('a', 'trackbars', 0, 10, on_a)
    cv2.createTrackbar('b', 'trackbars', 1, 1000, on_b)

    straight = cv2.imread('arrows/straight.jpg')
    left = cv2.imread('arrows/left.jpg')
    right = cv2.imread('arrows/right.jpg')
    straight_left = cv2.imread('arrows/straight_left.jpg')
    straight_right = cv2.imread('arrows/straight_right.jpg')
    templates = {"straight": get_template(straight),
                 # "left":get_template(left),
                 # "right":get_template(right),
                 "straight_left": get_template(straight_left),
                 "straight_right": get_template(straight_right)}
    pipeline(templates)
    #cropped1 = cv2.imread("cropped3.jpg")
    #feature_matching(cropped1,templates)
    #cv2.waitKey(0)


if __name__ == "__main__":
    main()
>>>>>>> 9b56a5c82cc53b70b6d8edf2e7fde547a639f270
