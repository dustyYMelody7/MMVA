import numpy as np
import cv2

def sorted_boxes(dt_boxes):
    """
    Sort ocr boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected ocr boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
            (_boxes[i + 1][0][0] < _boxes[i][0][0]):
            tmp = _boxes[i]
            _boxes[i] = _boxes[i + 1]
            _boxes[i + 1] = tmp
    return _boxes

def get_rotate_crop_image(img, points):
    img_height, img_width = img.shape[0:2]
    left = int(np.min(points[:, 0]))
    right = int(np.max(points[:, 0]))
    # 限制0
    if left < 0:
        left = 0
    right += 1
    top = int(np.min(points[:, 1]))
    bottom = int(np.max(points[:, 1]))
    # 限制0
    if top < 0:
        top = 0
    bottom += 1
    # print("img_crop:", left, right, top, bottom)
    img_crop = img[top:bottom, left:right, :].copy()
    points[:, 0] = points[:, 0] - left
    points[:, 1] = points[:, 1] - top
    img_crop_width = int(np.linalg.norm(points[0] - points[1]))
    img_crop_height = int(np.linalg.norm(points[0] - points[3]))
    pts_std = np.float32([[0, 0], [img_crop_width, 0],\
        [img_crop_width, img_crop_height], [0, img_crop_height]])

    M = cv2.getPerspectiveTransform(points, pts_std)
    dst_img = cv2.warpPerspective(
        img_crop,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_REPLICATE)
    dst_img_height, dst_img_width = dst_img.shape[0:2]
    if dst_img_height * 1.0 / dst_img_width >= 1.5:
        dst_img = np.rot90(dst_img)
    return dst_img

