#!/usr/bin/env python
# coding: utf-8


import cv2
import matplotlib.pyplot as plt
import numpy as np
from yolo_opencv import get_objects_video
import os

def get_frame_objects(video_filename, sample_rate=1):
    frames = get_objects_video(video_filename, sample_rate)

    clean_frames = []
    
    for frame in frames:
        video_objects = []
        for i, obj in enumerate(frame):
            if obj[1].shape[0] > 0 and obj[1].shape[1] > 0 and obj[1].shape[2] > 0:
                video_objects.append(obj)
        clean_frames.append(video_objects)
    return clean_frames