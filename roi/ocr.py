#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import glob
import pytesseract
import pandas as pd 
from PIL import Image
import imutils

pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/4.1.0/bin/tesseract'

def rotateImage(img):
    img = imutils.rotate(img, 2)
    return img

def adaptiveThreshold(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    threshGauss = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)

    return threshGauss

def resize(img):
    ratio = 500.0 / img.shape[1]
    dim = (500, int(img.shape[0] * ratio))

    img = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)

    return img

def addBorder(img):
    bordersize = 14
    img = cv2.copyMakeBorder(img, top=bordersize, bottom=bordersize, left=bordersize, right=bordersize,
                                borderType=cv2.BORDER_CONSTANT, value=[255, 255, 255])

    return img

def cleanOCR(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, lines=np.array([]),
                            minLineLength=100, maxLineGap=80)

    a, b, c = lines.shape
    for i in range(a):
        x = lines[i][0][0] - lines[i][0][2]
        y = lines[i][0][1] - lines[i][0][3]
        if x != 0:
            if abs(y / x) < 1:
                cv2.line(image, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (255, 255, 255),
                            1, cv2.LINE_AA)

    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    gray = cv2.morphologyEx(image, cv2.MORPH_CLOSE, se)

    # OCR
    config = '-l eng --oem 1 --psm 3'
    text = pytesseract.image_to_string(gray, config=config)

    validChars = ['A', 'B', 'G', 'H', 'M', 'N', 'R', 'S',
                    'T', 'U', 'V' 'X', 'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    cleanText = []

    for char in text:
        if char in validChars:
            cleanText.append(char)

    plate = ''.join(cleanText)
    # print(plate)

    return plate

def extract_plate_text(img):
    img = rotateImage(img)
    img = adaptiveThreshold(img)
    img = resize(img)
    img = addBorder(img)

    cv2.imwrite("test.jpg", img)

    text = cleanOCR(img)
    return text, img

img = cv2.imread("plates/plate_faces_26_ 0_1_.png")
print(extract_plate_text(img))