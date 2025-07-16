#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@File    :   Main.py
@Time    :   2024/08/06 10:14:28
@Author  :   Wilbur 
@Version :   3.0
@Description : 修复模式A的BUG
'''


import argparse, logging, json, os
from mClass import CameraScreenFaultyDetector, CameraShiftDetector
import os.path as osp
from glob import glob


IMG_FORMAT = ["jpg", "png", "JPG", "bmp", "jpeg"]


def parse_args():
    parser = argparse.ArgumentParser()  
    parser.add_argument("--mode",type=str, default="B", help="\
                            模式A: 只对一批图做故障屏检测\
                            模式B: 对所有NVR设备图做移位检测,再对检测出移位的做故障屏分析\
                            模式C: 只对单一NVR设备做移位检测")    
    parser.add_argument("--StandardDir", type=str, default="/home/Disk-6t/InspectionScreenshot/20250714", help="\
        如果选择模式A, 则该路径为待测的图片的路径\
        如果选择模式B和C, 则该路径为标准图的路径,其中的子文件夹是nvr设备,子文件夹内是图片")
    parser.add_argument("--ComparisonDir", type=str, default="/home/Disk-6t/InspectionScreenshot/20250716", help="采样图的路径，其中的子文件夹是nvr设备，子文件夹内是图片")
    parser.add_argument("--SavePath",type=str, default="/home/Disk-6t/InspectionScreenshot/20250716/20250716_run_result", help="检测出异常的图片存放地址，异常图片有NVR—id + 通道号拼成")
    parser.add_argument("--threshold",type=int, default=4, help="Variance of camera jitter, the value is 0-100")
    parser.add_argument("--splitN", type=list, default=[3, 3], help="Split image to N*N region")
    parser.add_argument("--nFeatures", type=int, default=15000, help="Max feature points of ORB detection")
    
    return parser.parse_args()


def main():
    logging.basicConfig(filename=osp.join(osp.dirname(osp.abspath(__file__)), 'AlgorithmError.log'), filemode='a', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
    os.makedirs(args.SavePath, exist_ok=True)
    
    shiftLogPath = osp.join(args.SavePath, "Shift.log")
    blackScreenLogPath = osp.join(args.SavePath, "BlackScreen.log")
    snowScreenLogPath = osp.join(args.SavePath, "SnowScreen.log")
    otherScreenLogPath = osp.join(args.SavePath, "OtherScreen.log")
    errorLogPath = osp.join(args.SavePath, "Error.log")
    
    sl = open(shiftLogPath, 'w')
    bsl = open(blackScreenLogPath, 'w')
    ssl = open(snowScreenLogPath, 'w')
    osl = open(otherScreenLogPath, 'w')
    el = open(errorLogPath, "w")
    
    csd = CameraShiftDetector(
        splitN=args.splitN,
        threshold=args.threshold,
        nfeatures=args.nFeatures
    )
    
    csfd = CameraScreenFaultyDetector()
    
    # 只做故障屏检测
    if args.mode == "A":
        imPathList = [imP for imP in glob(osp.join(args.StandardDir, "*/*.*"))]
        for imP in imPathList:

            try:
                blackFlag, otherFlag, edgeFlag = csfd.analyse(imP)
                if blackFlag:
                    bsl.write(imP + "\n")
                elif edgeFlag:
                    ssl.write(imP + "\n")
                elif otherFlag:
                    osl.write(imP + "\n")
            except Exception as e:
                logging.error("Exception occurred", exc_info=True)
                el.write(imP+"\n")
                
    # 对所有NVR设备图做移位检测,再对检测出移位的做故障屏分析
    elif args.mode == "B":
        standardImPathList = [imP for imP in glob(osp.join(args.StandardDir, "*/*.*")) if osp.basename(imP).rsplit(".", 1)[-1] in IMG_FORMAT]
        for simP in standardImPathList:
            cimP = osp.join(args.ComparisonDir, osp.basename(osp.dirname(simP)), osp.basename(simP))
            resultImPath = osp.join(args.SavePath, osp.basename(osp.dirname(simP)) + "_" + osp.basename(simP).rsplit(".", 1)[0] + ".jpg")
            orignalResultImPath = osp.join(args.SavePath, osp.basename(osp.dirname(simP)) + "_" + osp.basename(simP).rsplit(".", 1)[0] + ".png")
            moveFlag = csd.match(simP, cimP, resultImPath, orignalResultImPath)
            if moveFlag == "Moved":
                blackFlag, otherFlag, edgeFlag = csfd.analyse(cimP)
                if blackFlag:
                    bsl.write(cimP+"\n")
                elif edgeFlag:
                    ssl.write(cimP + "\n")
                elif otherFlag:
                    osl.write(cimP + "\n")
                else:
                    sl.write(resultImPath + "\n")
            elif moveFlag == "ShapeError":
                el.write(simP + "\n")
    
    # 对单个NVR设备图做移位检测,再对检测出移位的做故障屏分析
    elif args.mode == "C":
        standardImPathList = [imP for imP in glob(osp.join(args.StandardDir, "*.*")) if osp.basename(imP).rsplit(".", 1)[-1] in IMG_FORMAT]
        for simP in standardImPathList:
            cimP = osp.join(args.ComparisonDir, osp.basename(simP))
            resultImPath = osp.join(args.SavePath, osp.basename(osp.dirname(simP)) + "_" + osp.basename(simP).rsplit(".", 1)[0] + ".jpg")
            orignalResultImPath = osp.join(args.SavePath, osp.basename(osp.dirname(simP)) + "_" + osp.basename(simP).rsplit(".", 1)[0] + ".png")
            moveFlag = csd.match(simP, cimP, resultImPath, orignalResultImPath, saveFeaturePoints=True)
            if moveFlag == "Moved":
                blackFlag, otherFlag, edgeFlag = csfd.analyse(cimP)
                if blackFlag:
                    bsl.write(cimP+"\n")
                elif edgeFlag:
                    ssl.write(cimP + "\n")
                elif otherFlag:
                    osl.write(cimP + "\n")
                else:
                    sl.write(resultImPath + "\n")
            elif moveFlag == "ShapeError":
                el.write(simP + "\n")
    
    else:
        raise "Error mode must be A or B or C"
    
    sl.close()
    bsl.close()
    osl.close()
    ssl.close()
    el.close()
    
    data = [{"type": "error", "name": "采样图不存在或与标准图尺寸不匹配", "path": errorLogPath},
            {"type": "BlackScreen", "name": "黑屏", "path": blackScreenLogPath},
            {"type": "SnowScreen", "name": "雪花屏", "path": snowScreenLogPath},
            {"type": "otherScreen", "name": "其它故障屏", "path": otherScreenLogPath},
            {"type": "Shift", "name": "摄像头位移", "path": shiftLogPath}]
    sumLogPath = osp.join(args.SavePath, "SumLog.json")
    
    with open(sumLogPath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
        
if __name__ == "__main__":
    args = parse_args()
    main()