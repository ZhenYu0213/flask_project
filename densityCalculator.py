from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
from itertools import combinations


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Box:
    def __init__(self, x, y, xmin, ymin, xmax, ymax):
        self.x = x
        self.y = y
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax


def CalculateBoxesArea(centroid_dict):
    """
        calculate the total area of a object
    """
    totalArea = float(0.0)
    for key, box in centroid_dict.items():
        data = Box(box[0], box[1], box[2], box[3], box[4], box[5])
        boxArea = (data.xmax-data.xmin)*(data.ymax-data.ymin)
        totalArea += boxArea
    return totalArea


def isOverlap(box1, box2):
    print(box1)
    print(box2)
    boxA = Box(box1[0], box1[1], box1[2], box1[3], box1[4], box1[5])
    boxB = Box(box2[0], box2[1], box2[2], box2[3], box2[4], box2[5])
    # determine the minimum point of overlap --(x,y)
    left_top = (max(boxA.xmin, boxB.xmin), max(boxA.ymin, boxB.ymin))
    # determine the maximum point of overlap --(x,y)
    right_buttom = (min(boxA.xmax, boxB.xmax), min(boxA.ymax, boxB.ymax))
    p_left_top = Point(left_top)
    p_right_buttom = Point(right_buttom)
    # calculate area of two boxes
    Area1 = (boxA.xmax-boxA.xmin)*(boxA.ymax-boxA.ymin)
    Area2 = (boxB.xmax-boxB.xmin)*(boxB.ymax-boxB.ymin)
    if p_right_buttom.x > p_left_top.x and p_right_buttom.y > p_left_top.y:  # if overlap
        Intersection = (p_right_buttom.x-p_left_top.x) * \
            (p_right_buttom.y-p_left_top.y)
        Union = (Area1+Area2-Intersection)
        return Intersection
    else:
        return 0


def convertBack(x, y, w, h):
    # ================================================================
    # 2.Purpose : Converts center coordinates to rectangle coordinates
    # ================================================================
    """
    :param:
    x, y = midpoint of bbox
    w, h = width, height of the bbox

    :return:
    xmin, ymin, xmax, ymax
    """
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax


def DetermineUnionOfBoundedBoxes(detections, img):
    """
    Determine union of objects, which were in the same class
    """
    if len(detections) > 0:
        centroid_dict = dict() 						# Function creates a dictionary and calls it centroid_dict
        objectId = 0
        detectionList = []
        for detection in detections:
            nameTag = detection[0]
            detectionList.append(detection)
            if nameTag == 'tree':
                x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                # Create dictionary of tuple with 'objectId' as the index center points and bbox
                centroid_dict[objectId] = (
                    int(x), int(y), xmin, ymin, xmax, ymax)
                objectId += 1
            elif nameTag == 'planet':
                x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                # Create dictionary of tuple with 'objectId' as the index center points and bbox
                centroid_dict[objectId] = (
                    int(x), int(y), xmin, ymin, xmax, ymax)
                objectId += 1
            elif nameTag == 'beach':
                x, y, w, h = detection[2][0],\
                    detection[2][1],\
                    detection[2][2],\
                    detection[2][3]
                xmin, ymin, xmax, ymax = convertBack(
                    float(x), float(y), float(w), float(h))
                # Create dictionary of tuple with 'objectId' as the index center points and bbox
                centroid_dict[objectId] = (
                    int(x), int(y), xmin, ymin, xmax, ymax)
                objectId += 1
        # [TODO] calculate total area without intersection
        # while i < len(detectionList):
        #     j = i+1
        #     while j < len(detectionList):
        #         Interactions += isOverlap(centroid_dict[i], centroid_dict[j])
        #         j += 1
        #     i += 1
        # draw box
        #totalArea = CalculateBoxesArea(centroid_dict)
        Interactions = float(0.0)
        i = 0
        for key, box in centroid_dict.items():
            area = Box(box[0], box[1], box[2], box[3], box[4], box[5])
            print("Draw box.................")
            cv2.rectangle(img, (area.xmin, area.ymin),
                          (area.xmax, area.ymax), (0, 255, 0), 2)

        #totalUnion = totalArea-Interactions
        # print(totalUnion)
    return img


netMain = None
metaMain = None
altNames = None


def YOLO():
    """
    Perform Object detection
    """
    global metaMain, netMain, altNames
    configPath = "./cfg/yolo-obj.cfg"
    weightPath = "./backup/yolo-obj_8000.weights"
    metaPath = "./data/obj.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture("./data/testVideo.mp4")
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    new_height, new_width = frame_height // 2, frame_width // 2
    # print("Video Reolution: ",(width, height))

    out = cv2.VideoWriter(
        "./Demo/test_output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0,
        (new_width, new_height))

    # print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(new_width, new_height, 3)

    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        # Check if frame present :: 'ret' returns True if frame present, otherwise break the loop.
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (new_width, new_height),
                                   interpolation=cv2.INTER_LINEAR)
        #classNameList = ['tree', 'planet', 'beach']

        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())
        detections = darknet.detect_image(
            netMain, metaMain, darknet_image, thresh=0.25)
        image = DetermineUnionOfBoundedBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))
        cv2.imshow('Demo', image)
        cv2.waitKey(3)
        out.write(image)

    cap.release()
    out.release()
    print(":::Video Write Completed")


if __name__ == "__main__":
    YOLO()
