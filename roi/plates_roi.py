#!/usr/bin/env python
# coding: utf-8

import cv2
import matplotlib.pyplot as plt
import numpy as np
from yolo_opencv import get_objects
import os

def apply_threshold_to_img(img, threshold_type="binary"):
    if threshold_type == "binary":
        _,img = cv2.threshold(img,150, 255,cv2.THRESH_BINARY)
    elif threshold_type == "adaptive_gaussian":
        img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    return img

def get_license_plates(filename):
    plates = []
    image = cv2.imread("raw_plates/" + filename)
    if image is not None:    
        gray = image

        aspect_ratio = 1
        new_size = 800
        gray = cv2.cvtColor(cv2.resize(gray, (int(new_size * aspect_ratio), int(new_size * (1/aspect_ratio)))),
                                       cv2.COLOR_BGR2GRAY) #convert to grey scale
        gray = gray[int(gray.shape[1]/2):, :]
        gray_orig = gray

        gray = cv2.bilateralFilter(gray, 30, 30, 30)

        gray = apply_threshold_to_img(gray, "adaptive_gaussian")

        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = apply_threshold_to_img(gray, "binary")

        kernel = np.ones((2,2), np.uint8)
        gray = cv2.dilate(gray, kernel, iterations=1)

        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = apply_threshold_to_img(gray, "adaptive_gaussian")

        gray = cv2.dilate(gray, kernel, iterations=1)

        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = apply_threshold_to_img(gray, "adaptive_gaussian")

        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = apply_threshold_to_img(gray, "adaptive_gaussian")

        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = cv2.GaussianBlur(gray,(5,5),cv2.BORDER_DEFAULT)
        gray = apply_threshold_to_img(gray, "binary")

        high_thresh, thresh_im = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        lowThresh = 0.5*high_thresh

        edged = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)

        edged = cv2.Canny(edged, lowThresh, high_thresh) #Perform Edge detection

        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (10,10))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (10,10))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (10,10))
        edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, (10,10))

        _, contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        biggest_area = 0
        best_candidate = None

        best_candidate = None
        best_ratio = 0

        candidates = []

        i = 0
        for c in contours:
            c = cv2.convexHull(c, False)
            epsilon = 0.018*cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            if len(approx) == 4:
                rect = cv2.minAreaRect(approx)
                box = cv2.boxPoints(rect)
                box = np.int0(approx)
                candidates.append((cv2.contourArea(approx), box))
                i = i + 1

        if len(candidates) > 0:
            if len(candidates) > 1:
                candidates.sort(reverse=True, key=lambda x: x[0])
                candidates = [c for c in candidates[:2]]

            for i, c in enumerate(candidates):
                contour_mask = np.zeros(gray.shape,np.uint8)
                x, y, w, h = cv2.boundingRect(c[1])
                cv2.rectangle(contour_mask, (x, y), (x+w, y+h), (255, 255, 255), -1);

                img1_bg = cv2.bitwise_and(gray_orig,gray_orig,mask = contour_mask)

                (x, y) = np.where(contour_mask == 255)
                (topx, topy) = (np.min(x), np.min(y))
                (bottomx, bottomy) = (np.max(x), np.max(y))
                cropped_img_gray = gray_orig[topx:bottomx+1, topy:bottomy+1]

                cropped_img_gray = cv2.resize(cropped_img_gray, (500, 300))

                plates.append(cropped_img_gray)
    return plates