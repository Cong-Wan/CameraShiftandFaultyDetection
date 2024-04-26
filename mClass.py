#!/usr/bin/env python
import cv2, random
import os.path as osp
import numpy as np
import matplotlib.pyplot as plt



class CameraShiftDetector:
    def __init__(self,
                    splitN=[3, 4], 
                    threshold=4,
                    minMatchedPoints=2,
                    nfeatures=15000 
                        ) -> None:

        self.splitN = splitN
        self.threshold = threshold
        self.minMatchedPoints = minMatchedPoints
        self.nFeatures = nfeatures

        self.orb = cv2.ORB_create(self.nFeatures)

        # Create FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    def match(self, standardImPath, comparisonImPath, savePath,saveFeaturePoints=False):

        self.standardIm = cv2.imread(standardImPath)
        self.comparisonIm = cv2.imread(comparisonImPath)

        if self.standardIm.shape == self.comparisonIm.shape: #, "The current picture size does not match the standard picture size!"

            standardGray = self.preProcessIm(self.standardIm)
            comparisonGray = self.preProcessIm(self.comparisonIm)

            standardKps, standardFeature = self.orb.detectAndCompute(standardGray, None)
            comparisonKps, comparisonFeature = self.orb.detectAndCompute(comparisonGray, None)

            matches = self.flann.knnMatch(standardFeature.astype(np.float32), comparisonFeature.astype(np.float32), k=2)
            
            matched2CalculateDistance = []
            for m, _ in matches:
                
                pt1 = standardKps[m.queryIdx].pt
                pt2 = comparisonKps[m.trainIdx].pt

                pt1_x, pt1_y = pt1
                pt2_x, pt2_y = pt2

                if self.filterPoints(pt1, pt2):
                    matched2CalculateDistance.append([[pt1_x, pt1_y], [pt2_x,pt2_y]])

            # offsets of matched points
            offsets = [self.calculateDistance(m[0], m[1]) for m in matched2CalculateDistance]
            
            lessthresNumber = len([d for d in offsets if d < float(self.threshold)])

            # Doesn't move
            if lessthresNumber >= self.minMatchedPoints:
                return "NoMove"
            
            else:
                
                self.saveReuslt(savePath, matched2CalculateDistance, saveFeaturePoints=False)
                return "Moved"
                

        else:
            maxHeight = max(self.standardIm.shape[0], self.comparisonIm.shape[0])
            totalWidth = self.standardIm.shape[1] + self.comparisonIm.shape[1]

            merged = 255 * np.ones((maxHeight, totalWidth, 3), dtype=np.uint8)
            merged[0:self.standardIm.shape[0], 0:self.standardIm.shape[1]] = self.standardIm
            merged[0:self.comparisonIm.shape[0], self.comparisonIm.shape[1]:] = self.comparisonIm
            
            cv2.imwrite(savePath, merged)
            return "ShapeError"
    def saveReuslt(self, savePath, matched2CalculateDistance, saveFeaturePoints=False, saveGid=True):        

        result = np.zeros((self.imH, self.imW*2, 3), dtype=np.uint8)
        
        result[:self.imH, :self.imW] = self.standardIm
        result[:self.imH, self.imW:] = self.comparisonIm

        if saveFeaturePoints:
            for matched in matched2CalculateDistance:
                distance = self.calculateDistance(matched[0], matched[1])

                if distance < float(self.threshold):

                    cv2.circle(result, (int(matched[0][0]), int(matched[0][1])), 4, (0, 0, 255) ,2)
                    cv2.circle(result, (int(matched[1][0] + self.standardIm.shape[1]), int(matched[1][1])), 4, (0, 0, 255) ,2)
                    
                    cv2.line(result, (int(matched[0][0]), int(matched[0][1])), (int(matched[1][0]) + self.imW, int(matched[1][1])), (0, 0, 255), 1)
                    cv2.putText(result, str(distance), (int((matched[0][0] + self.imW + matched[1][0]) / 2), int(matched[0][1])), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                else:
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                    cv2.circle(result, (int(matched[0][0]), int(matched[0][1])), 2, color ,2, cv2.FILLED)
                    cv2.circle(result, (int(matched[1][0] + self.imW), int(matched[1][1])), 2, color ,2, cv2.FILLED)
        
        # 画网格线以便更好的人工筛查是否移动
        if saveGid:
            splitH = [int((self.imH//10) * (i+1)) for i in range(10)]
            splitW = [int((self.imW//10) * (i+1)) for i in range(10)]

            for i in range(len(splitH)):

                cv2.line(result, (0, splitH[i]), (self.imW, splitH[i]), (152, 131, 154), 1)
                cv2.line(result, (self.imW, splitH[i]), (self.imW*2, splitH[i]), (152, 131, 154), 1)

            for i in range(len(splitW)):

                cv2.line(result, (splitW[i], 0), (splitW[i], self.imH), (152, 131, 154), 1)
                cv2.line(result, (splitW[i]+self.imW, 0), (splitW[i]+self.imW, self.imH), (152, 131, 154), 1)
        
        cv2.imwrite(savePath, result)
     
    def preProcessIm(self, im):
        self.imH, self.imW = im.shape[:2]
        
        croppedIm = im[int(0.1*self.imH):int(0.9*self.imH), :]
        gray = cv2.cvtColor(croppedIm, cv2.COLOR_BGR2GRAY)

        return gray

    def filterPoints(self, p1, p2, eps = 0.00001):
        """
        p1/p2 is tuple
        eps avoid divide zero
        """
        # 判断两个匹配点是否在同一区域
        pt1_x, pt1_y = p1
        pt2_x, pt2_y = p2
        pt1_ = self.identifyPointRegion(pt1_x, pt1_y)
        pt2_ = self.identifyPointRegion(pt2_x, pt2_y)

        # 计算两个匹配点之间的tan(θ)值, tan(5°) ≈ 0.0874759
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = round(dy / (dx + eps), 7)
    
        if -0.0874759 <= angle <= 0.0874759 and pt1_ == pt2_:
            return True
        else:
            return False
        
    # Identify the region of the photo where the point is located
    def identifyPointRegion(self, x, y):
        region_x = x // (self.imW // int(self.splitN[0]))
        region_y = y // (self.imH // int(self.splitN[1]))

        return region_x, region_y
    
    # Calculate the Euclidean distance between two points
    def calculateDistance(self, p1, p2):
        point1, point2 = np.array(p1), np.array(p2)
        return np.linalg.norm(point1 - point2)

    
class CameraScreenFaultyDetector:
    def __init__(self, 
                 slideWindowSize=20, 
                 threshold=0.7
                 ) -> None:
        
        self.slideWindowSize = slideWindowSize
        self.threshold = threshold
    
    def analyse(self, imPath):
        # 滑动窗口分析直方图的结果，如果一个滑动窗口内的总和是大于图像总像素点的百分之八十，设为True
        # flag1 = False

        # 判断边缘信息值，如果小于0.009则为不正常，设为True
        # flag2 = False

        im = cv2.imread(imPath)
        
        h, w = im.shape[:2]
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        blackFlag, otherFlag= self.analyseHistogram(im, h, w, threshold=self.threshold, slideWindowSize=self.slideWindowSize)
        
        # 边缘信息只判断解码错误
        edgeFlag = self.analyseEdge(gray, h, w)

        return blackFlag, otherFlag, edgeFlag
    def analyseHistogram(self,im, h, w, threshold,slideWindowSize):
        otherFlag = False
        blackFlag = False
        
        hist = cv2.calcHist([im],[0],None,[256],[0,256])
        
        blackCount1 = np.sum(hist[0:21])
        blackCount2 = np.sum(hist[21:41])
        if blackCount1 > (h * w*threshold) or blackCount2 > (h * w*threshold):
            blackFlag = True
        
        else:
            for i in range(41, 256):
                if i < 256 - slideWindowSize:
                    pc = np.sum(hist[i:i+slideWindowSize])
                    
                    if pc > h * w*threshold:
                        otherFlag = True
                        break
            
        return blackFlag, otherFlag
    
    def analyseEdge(self, gray, h, w):
        flag = False
        blur = cv2.GaussianBlur(gray, (7, 7), 4)
        edge = cv2.Canny(blur, 127, 200)
        
        # 计算边缘像素的数量
        edgeCount = np.count_nonzero(edge)
        
        if edgeCount / (w * h) > 0.1:# or edgeCount / (self.w * self.h) < 0.005:
            flag = True
        return flag

    
