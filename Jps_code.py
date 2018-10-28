from tkinter import *
import os
import os.path
import math
from tkinter.filedialog import *
from tkinter.simpledialog import *
import operator
import numpy as np
import struct
import threading
from wand.image import *
from wand.color import Color
from wand.drawing import Drawing
import sqlite3
import pymysql
import csv
import xlrd
from xlsxwriter import Workbook
import xlsxwriter
import matplotlib.pyplot as plt
import glob
import json
import tensorflow as tf
import pandas as pd

## 함수 선언
def loadImage(fname):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, VIEW_X, VIEW_Y
    fsize = os.path.getsize(fname)
    inH = inW = int(math.sqrt(fsize))
    inImage = np.zeros([inH, inW], dtype=np.int32)
    fp = open(fname, 'rb')
    for  i  in range(inH) :
        for  k  in  range(inW) :
            inImage[i][k] =  int(ord(fp.read(1)))
    fp.close()
    
def openFile():
    global window, canvas, paper, filename,inImage, outImage,inW, inH, outW, outH, photo, gif, VIEW_X, VIEW_Y
    filename = askopenfilename(parent=window,
                               filetypes=(("그림파일", "*.raw;*.gif;*.jpg;*.png;*.tif;*.bmp"), ("모든파일", "*.*")))
    if pLabel != None:
        pLabel.destroy()
    if filename[-3:] != "raw":
        gif = True
        loadImage_gif(filename)
        equal_gif()
        return
    else: gif = False
    loadImage(filename) # 파일 -> 입력메모리
    equal() # 입력메모리 -> 출력메모리
    
def display_geniune():
    global VIEW_X, VIEW_Y
    brt = askinteger('출력 비율을 정하세요. ', '64, 128, 256, 512, 1024..', minvalue=64, maxvalue=4096)
    VIEW_X = VIEW_Y = brt
    display_first()

def display():
    global window, canvas, PLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    global VIEW_X, VIEW_Y
    if gif == True:
        display_gif()
        return
    if  canvas != None:
        canvas.destroy()
    if pLabel != None:
        pLabel.destroy()
    if VIEW_X >= outW or VIEW_Y >= outH:
        VIEW_X = outW
        VIEW_Y = outH
    step = int(outW / VIEW_X)
    window.geometry(str(VIEW_X * 2) + 'x' + str(VIEW_Y * 2))
    canvas = Canvas(window, width=VIEW_X, height=VIEW_Y)
    paper = PhotoImage(width=VIEW_X, height=VIEW_Y)
    canvas.create_image((VIEW_X/2, VIEW_Y/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH, step) :
            for k in range(0, outW, step) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data, data, data), (int(k/step), int(i/step)))
    threading.Thread(target=putPixel).start()
    canvas.pack(expand=1, anchor=CENTER)
    status.configure(text="이미지 정보: " + str(outW) + " X " + str(outH) + " / 출력 정보: " + str(VIEW_X) + " X " + str(VIEW_Y))
    status.pack()
    
def display_first():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    global VIEW_X, VIEW_Y
    if gif == True:
        display_first_gif()
        return
    if  canvas != None :
        canvas.destroy()
    if VIEW_X >= outW or VIEW_Y >= outH:
        VIEW_X = outW
        VIEW_Y = outH
    step = int(outW / VIEW_X)
    window.geometry(str(VIEW_X * 2) + 'x' + str(VIEW_Y * 2))
    canvas = Canvas(window, width=VIEW_X, height=VIEW_Y)
    paper = PhotoImage(width=VIEW_X, height=VIEW_Y)
    paper_copy = paper.copy()
    canvas.create_image((VIEW_X/2, VIEW_Y/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH, step) :
            for k in range(0, outW, step) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data, data, data), (int(k/step), int(i/step)))
                paper_copy.put('#%02x%02x%02x' % (data, data, data), (int(k/step), int(i/step)))
    threading.Thread(target=putPixel).start()        
    canvas.pack(expand=1, anchor=CENTER)
    status.configure(text="이미지 정보: " + str(outW) + " X " + str(outH) + " / 출력 정보: " + str(VIEW_X) + " X " + str(VIEW_Y))
    status.pack()
    
def display_copy():
    global window, canvas, pLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    global VIEW_X, VIEW_Y
    if gif == True:
        display_copy_gif()
        return
    if  canvas != None :
        canvas.destroy()
    if VIEW_X >= outW or VIEW_Y >= outH:
        VIEW_X = outW
        VIEW_Y = outH
    window.geometry(str(VIEW_X*2) + 'x' + str(VIEW_Y * 2))
    canvas = Canvas(window, width=VIEW_X, height=VIEW_Y)
    canvas.create_image((VIEW_X/2, VIEW_Y/2), image=paper, state='normal')
    canvas.pack(side=RIGHT)
    photo = PhotoImage()
    pLabel = Label(window, image=photo)
    pLabel.pack(side=LEFT)
    pLabel.configure(image=paper_copy)
    
def rollback():
    if gif == True:
        rollback_gif()
        return
    global window, canvas, paper, PLabel, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if pLabel != None:
        pLabel.destroy()
    loadImage(filename)
    equal()
            
def equal():
    if gif == True:
        equal_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW = inW
    outH = inH
    outImage = np.zeros([inW, inH], dtype=np.int32)
    np.array(inImage)
    for i in range(inH):
        for j in range(inW):
            outImage[i][j] = inImage[i][j]           
    display_first()
    
def saveFile():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    draw = Drawing()  # 빈 판
    # 빈 판에 컬러 -> #000000 ~ #FFFFFF
    saveFp = asksaveasfile(parent=window, mode='w', defaultextension='.png'
                           , filetypes=(("그림파일", "*.gif;*.jpg;*.png;*.tif;*.bmp"), ("모든파일", "*.*")))
    for i in range(outW):
        for j in range(outH):
            dataR = outImage[i][j][0]
            dataG = outImage[i][j][1]
            dataB = outImage[i][j][2]
            hexStr = '#'
            if dataR > 15:
                hexStr += hex(dataR)[2:]
            else:
                hexStr += ('0' + hex(dataR)[2:])
            if dataG > 15:
                hexStr += hex(dataG)[2:]
            else:
                hexStr += ('0' + hex(dataG)[2:])
            if dataB > 15:
                hexStr += hex(dataB)[2:]
            else:
                hexStr += ('0' + hex(dataB)[2:])
            draw.fill_color = Color(hexStr)
            draw.color(j, i, 'replace')
    with Image(filename=filename) as img:
        draw(img)
        img.save(filename=saveFp.name)
    print("SAVE OK")

def exitFile() :
    global window, canvas, paper, filename,inImage, outImage,inW, inH, outW, outH
    pass

def addImage(num):
    if gif == True:
        addImage_gif(num)
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    outW = inW
    outH = inH
    outImage = np.zeros([inH, inW], dtype=np.int32)
        
    if num == 1:
        brt = askinteger('밝게하기', '밝게할 값', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] + brt > 255:
                    outImage[i][k] = 255
                else:
                    outImage[i][k] = inImage[i][k] + brt
                    
    elif num == 2:
        brt = askinteger('어둡게하기', '어둡게할 값', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] - brt < 0:
                    outImage[i][k] = 0
                else:
                    outImage[i][k] = inImage[i][k] - brt
    elif num == 3:
        brt = askinteger('밝게하기', '곱할 값', minvalue=1, maxvalue=10)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] * brt > 255:
                    outImage[i][k] = 255
                else:
                    outImage[i][k] = inImage[i][k] * brt
    elif num == 4:
        brt = askinteger('어둡게하기', '나눌 값', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = int(inImage[i][k] / brt)
                
    elif num == 5: # AND
        brt = askinteger('AND', '상수', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] & brt > 255:
                    outImage[i][k] = 255
                elif inImage[i][k] & brt < 0:
                    outImage[i][k] = 0
                else:                  
                    outImage[i][k] = inImage[i][k] & brt
    elif num == 6: # OR
        brt = askinteger('OR', '상수', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] | brt > 255:
                    outImage[i][k] = 255
                elif inImage[i][k] | brt < 0:
                    outImage[i][k] = 0
                else:                  
                    outImage[i][k] = inImage[i][k] | brt

    elif num == 7: # XOR
        brt = askinteger('OR', '상수', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = inImage[i][k] ^ brt
                if outImage[i][k] > 255: outImage[i][k] = 255
                elif outImage[i][k] < 0: outImage[i][k] = 0
                    
                    
    elif num == 8: # 반전
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = 255 - inImage[i][k]
                
    elif num == 9: # 감마
        brt = askfloat('감마', '소수', minvalue=0, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = int(inImage[i][k] * (1/brt))
                if outImage[i][k] > 255: outImage[i][k] = 255
                elif outImage[i][k] < 0: outImage[i][k] = 0
                       
           
    elif num == 10: # parabola(cap)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = int(-255 * ((inImage[i][k] / 127 - 1) ** 2) + 255)
                if outImage[i][k] > 255: outImage[i][k] = 255
                elif outImage[i][k] < 0: outImage[i][k] = 0

    elif num == 11: # parabola(cap)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = int(255 * ((inImage[i][k] / 127 - 1) ** 2))
                if outImage[i][k] > 255: outImage[i][k] = 255
                elif outImage[i][k] < 0: outImage[i][k] = 0
    
    elif num == 12: # binary
        brt = askinteger('임계치', '정수(1~255)', minvalue=0, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] > brt: outImage[i][k] = 255
                elif inImage[i][k] <= brt: outImage[i][k] = 0

    elif num == 13: # 범위강조
        brt1 = askinteger('첫 번째 범위 수', '정수(1~255)', minvalue=0, maxvalue=255)
        brt2 = askinteger('두 번째 범위 수', '정수(1~255)', minvalue=0, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k] > brt1 and inImage[i][k] < brt2:
                    outImage[i][k] = 255              
    display()
    
def a_average(num): # 입력 // 출력 평균
    if gif == True:
        a_average_gif(num)
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    sumList1, sumList2 = [], []
    if num == 2:
        brt = askinteger('절사 수치', '정수(1~100)%', minvalue=1, maxvalue=100)
        cut = 255*(brt/100)
    for i in range(inH):
        for k in range(inW):
            if num == 1:
                sumList1.append(inImage[i][k])
                sumList2.append(outImage[i][k])
            if num == 2:
                if inImage[i][k] > int(cut) and inImage[i][k] < 255-int(cut):
                    sumList1.append(inImage[i][k])
                if outImage[i][k] > int(cut) and outImage[i][k] < 255-int(cut):
                    sumList2.append(outImage[i][k])                   
    inAvg = sum(sumList1) / len(sumList1)
    outAvg = sum(sumList2) / len(sumList2)
    subWindow = Toplevel(window)
    subWindow.geometry('200x100')
    label1 = Label(subWindow, text="입력 평균: " + str(inAvg))
    label2 = Label(subWindow, text="출력 평균: " + str(outAvg))
    label1.pack()
    label2.pack()
    
def a_minmax():
    if gif == True:
        a_minmax_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    allDict1, allDict2 = {}, {}
    for i in range(inH):
        for j in range(inW):
            if inImage[i][j] in allDict1:
                allDict1[inImage[i][j]] += 1
            else:
                allDict1[inImage[i][j]] = 1
            if outImage[i][j] in allDict2:
                allDict2[outImage[i][j]] += 1
            else:
                allDict2[outImage[i][j]] = 1                
    sortList1 = sorted(allDict1.items(), key=operator.itemgetter(1))
    sortList2 = sorted(allDict2.items(), key=operator.itemgetter(1))
    subWindow = Toplevel(window)
    subWindow.geometry('200x100')
    label1 = Label(subWindow, text="입력 최대&최소: " + str(sortList1[-1]) + str(sortList1[0]))
    label2 = Label(subWindow, text="출력 최대&최소: " + str(sortList2[-1]) + str(sortList2[0]))
    label1.pack()
    label2.pack()            

def direct(num):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW = inW
    outH = inH
    outImage = []
    tmpList = []
    for i in range(outH):
        tmpList = []
        for k in range(outW):
            tmpList.append(0)
        outImage.append(tmpList)        
    for  i  in  range(inH) :
        for  k  in  range(inW) :
            if num == 1:
                outImage[outW-1-i][k] = inImage[i][k]
            if num == 2:
                outImage[i][outH-1-k] = inImage[i][k]
    display()
    
def panImage():
    global panYN
    panYN = True

def mouseClick(event):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    global sx, sy, ex, ey, panYN
    if not panYN:
        return
    if gif:
        mouseClick_gif(event)
        return
    sx = event.x
    sy = event.y

def mouseDrop(event):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    global sx, sy, ex, ey, panYN
    if not panYN:
        return
    if gif:
        mouseDrop_gif(event)
        return
    ex = event.x
    ey = event.y
    mx = sx - ex
    my = sy - ey
    outW = inW
    outH = inH
    outImage = []
    tmpList = []
    for i in range(outH):
        tmpList = []
        for k in range(outW):
            tmpList.append(0)
        outImage.append(tmpList)
    np.array(outImage)
    for  i  in  range(inH) :
        for  k  in  range(inW):
            if 0 < i-mx < outH and 0 < k-my < outW:
                outImage[i-mx][k-my] = inImage[i][k]
    panYN = False
    display()
    
def zoomOut():
    if gif:
        zoomOut_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger('축소하기', '축소할 배수-->', minvalue=2, maxvalue=32)
    outW = int(inW/scale);  outH = int(inH/scale);
    outImage = [];   tmpList = []
    for i in range(outH):  # 출력메모리 확보(0으로 초기화)
        tmpList = []
        for k in range(outW):
            tmpList.append(0)
        outImage.append(tmpList)
    for  i  in  range(inH) :
        for  k  in  range(inW) :
             outImage[int(i/scale)][int(k/scale)] = inImage[i][k]
    display()

def zoomIn(num):
    if gif:
        zoomIn_gif(num)
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger("확대하기", "배수", minvalue=2, maxvalue=32)
    outW = inW * scale
    outH = outH * scale
    outImage = np.zeros([outH, outW], dtype=np.int32)
    if num == 1:
        for  i  in  range(inH):
            for  k  in  range(inW):
                outImage[int(i*scale)][int(k*scale)] = inImage[i][k]
    if num == 2:
        for i in range(outH):
            for k in range(outW):
                outImage[int(i)][int(k)] = inImage[int(i/scale)][int(k/scale)]
    display()
    
def embossing():
    if gif:
        embossing_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger("scale(3,5,7,9)", "정수", minvalue=1, maxvalue=10)
    outW, outH = inW, outH
    if int(scale) % 2 == 0: embossing()
    # mask matrix
    mask = np.zeros([scale, scale], dtype=np.int32)
    mask_num = np.array([-1, 1, 0])
    n = 0
    for i in range(scale):
        for j in range(scale):
            if i == j:
                mask[i][j] = mask_num[n]
                n += 1
                if n > 2: n = 0 
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            outImage[i][j] = np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2)] * mask) + 127
            if outImage[i][j] > 255: outImage[i][j] = 255
            elif outImage[i][j] < 0: outImage[i][j] = 0
    display()

def blurring(num):
    if gif:
        blurring_gif(num)
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger("scale(3,5,7,9)", "정수", minvalue=1, maxvalue=10)
    outW, outH = inW, outH
    if int(scale) % 2 == 0: embossing()
    # mask matrix
    mask = np.ones([scale, scale], dtype=np.float32)
    mask = mask * (1/np.square(scale))
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            outImage[i][j] = int(np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2)] * mask))
            if num == 1:
                outImage[i][j] = inImage[i][j] - outImage[i][j]
            if outImage[i][j] > 255: outImage[i][j] = 255
            elif outImage[i][j] < 0: outImage[i][j] = 0
    display()
    
def gausian_blurring():
    if gif:
        gausian_blurring_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW, outH = inW, outH
    # mask matrix
    scale = 3
    mask = np.array([[1./16., 1./8., 1./16.], [1./8., 1./4., 1./8.], [1./16., 1./8., 1./16.]])
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            outImage[i][j] = int(np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2)] * mask))
            if outImage[i][j] > 255: outImage[i][j] = 255
            elif outImage[i][j] < 0: outImage[i][j] = 0
    display()

def sharpening(num):
    if gif:
        sharpening_gif(num)
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW, outH = inW, outH
    # mask matrix
    scale = 3
    if num == 1:
        mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    if num == 2:
        mask = np.array([[-1./9., -1./9., -1./9.], [-1./9., 8./9., -1./9.], [-1./9., -1./9., -1./9.]])
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            outImage[i][j] = int(np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2)] * mask))
            if outImage[i][j] > 255: outImage[i][j] = 255
            elif outImage[i][j] < 0: outImage[i][j] = 0
    display()
    
''' ######################### GIF 처리 공간 ######################### '''
def loadImage_gif(fname) :
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    photo = Image(filename=filename)
    inW = photo.width
    inH = photo.height
    inImage = []
    tmpList = []
    for i in range(inH):
        tmpList = []
        for k in range(inW) :
            tmpList.append(np.array([0, 0, 0]))
        inImage.append(tmpList)
    blob = photo.make_blob(format='RGB')
    for  i  in range(inH):
        for  k  in  range(inW):
            r, g, b = blob[(i * 3 * inH) + (k * 3) + 0], blob[(i * 3 * inH) + (k * 3) + 1], blob[
                (i * 3 * inH) + (k * 3) + 2]
            inImage[i][k] = [r, g, b]
    inImage = np.array(inImage)
    photo = None

def saveFile_gif():
    global window, canvas, paper, filename,inImage, outImage,inW, inH, outW, outH
    saveFp = asksaveasfile(parent=window, mode='wb',
                               defaultextension="*.gif", filetypes=(("GIF파일", "*.gif"), ("모든파일", "*.*")))
    for i in range(outW):
        for k in range(outH):
            saveFp.write( struct.pack(outImage[i][k][0], outImage[i][k][1], outImage[i][k][2]))
    saveFp.close()

def display_gif():
    global window, canvas, pLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    global VIEW_X, VIEW_Y
    if  canvas != None:
        canvas.destroy()
    if pLabel != None:
        pLabel.destroy()
    if VIEW_X >= outW or VIEW_Y >= outH:
        VIEW_X = outW
        VIEW_Y = outH
    step = int(outW / VIEW_X)
    window.geometry(str(VIEW_X * 2) + 'x' + str(VIEW_Y * 2))
    canvas = Canvas(window, width=VIEW_X, height=VIEW_Y)
    paper = PhotoImage(width=VIEW_X, height=VIEW_Y)
    canvas.create_image((VIEW_X/2, VIEW_Y/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH, step) :
            for k in range(0, outW, step) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data[0], data[1], data[2]), (int(k/step), int(i/step)))
    threading.Thread(target=putPixel).start()
    canvas.pack(expand=1, anchor=CENTER)
    status.configure(text="이미지 정보: " + str(outW) + " X " + str(outH) + " / 출력 정보: " + str(VIEW_X) + " X " + str(VIEW_Y))
    status.pack()

def display_first_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    global VIEW_X, VIEW_Y
    if  canvas != None :
        canvas.destroy()
    if VIEW_X >= outW or VIEW_Y >= outH:
        VIEW_X = outW
        VIEW_Y = outH
    step = int(outW / VIEW_X)
    window.geometry(str(VIEW_X * 2) + 'x' + str(VIEW_Y * 2))
    canvas = Canvas(window, width=VIEW_X, height=VIEW_Y)
    paper = PhotoImage(width=VIEW_X, height=VIEW_Y)
    paper_copy = paper.copy()
    canvas.create_image((VIEW_X/2, VIEW_Y/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH, step) :
            for k in range(0, outW, step) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data[0], data[1], data[2]), (int(k/step), int(i/step)))
                paper_copy.put('#%02x%02x%02x' % (data[0], data[1], data[2]), (int(k/step), int(i/step)))
    threading.Thread(target=putPixel).start()        
    canvas.pack(expand=1, anchor=CENTER)
    status.configure(text="이미지 정보: " + str(outW) + " X " + str(outH) + " / 출력 정보: " + str(VIEW_X) + " X " + str(VIEW_Y))
    status.pack()
        
def display_copy_gif():
    global window, canvas, pLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    global VIEW_X, VIEW_Y
    if  canvas != None :
        canvas.destroy()
    if VIEW_X >= outW or VIEW_Y >= outH:
        VIEW_X = outW
        VIEW_Y = outH
    window.geometry(str(VIEW_X*2) + 'x' + str(VIEW_Y * 2))
    canvas = Canvas(window, width=VIEW_X, height=VIEW_Y)
    canvas.create_image((VIEW_X/2, VIEW_Y/2), image=paper, state='normal')
    canvas.pack(side=RIGHT)
    photo = PhotoImage()
    pLabel = Label(window, image=photo)
    pLabel.pack(side=LEFT)
    pLabel.configure(image=paper_copy)
    
def rollback_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if pLabel != None:
        pLabel.destroy()
    loadImage_gif(filename)
    equal_gif()    
        
def equal_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW = inW
    outH = inH
    outImage = []
    tmpList = []
    for i in range(outH):
        tmpList = []
        for k in range(outW):
            tmpList.append(np.array([0, 0, 0]))
        outImage.append(tmpList)        
    for  i  in  range(inH):
        for  k  in  range(inW):
            outImage[i][k] = inImage[i][k]
    outImage = np.array(outImage)
    display_first_gif()
    
def addImage_gif(num):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    outW = inW
    outH = inH
    outImage = []
    tmpList = []
    for i in range(outH):
        tmpList = []
        for k in range(outW):
            tmpList.append(np.array([0, 0, 0]))
        outImage.append(tmpList)
    outImage = np.array(outImage)
        
    if num == 1:
        brt = askinteger('밝게하기', '밝게할 값', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
               outImage[i][k] = inImage[i][k] + brt
               outImage[i][k][outImage[i][k] > 255] = 255
               outImage[i][k][outImage[i][k] < 0] = 0
               
    elif num == 2:
        brt = askinteger('어둡게하기', '어둡게할 값', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = inImage[i][k] - brt
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0
            
    elif num == 3:
        brt = askinteger('밝게하기', '곱할 값', minvalue=1, maxvalue=10)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = inImage[i][k] * brt
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0

    elif num == 4:
        brt = askinteger('어둡게하기', '나눌 값', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = np.array(inImage[i][k] / brt, dtype=np.int32)
                
    elif num == 5: # AND
        brt = askinteger('AND', '상수', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                print(type(inImage))
                outImage[i][k] = inImage[i][k] & brt
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0
                
    elif num == 6: # OR
        brt = askinteger('OR', '상수', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = inImage[i][k] | brt
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0

    elif num == 7: # XOR
        brt = askinteger('OR', '상수', minvalue=1, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = inImage[i][k] ^ brt
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0
                    
    elif num == 8: # 반전
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = 255 - inImage[i][k]
                
    elif num == 9: # 감마
        brt = askfloat('감마', '소수', minvalue=0, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = np.array(inImage[i][k] * (1/brt), dtype=np.int32)
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0
                       
           
    elif num == 10: # parabola(cap)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = np.array(-255 * ((inImage[i][k] / 127 - 1) ** 2) + 255, dtype=np.int32)
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0

    elif num == 11: # parabola(cap)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                outImage[i][k] = np.array(255 * ((inImage[i][k] / 127 - 1) ** 2), dtype=np.int32)
                outImage[i][k][outImage[i][k] > 255] = 255
                outImage[i][k][outImage[i][k] < 0] = 0
    
    elif num == 12: # binary
        brt = askinteger('임계치', '정수(1~255)', minvalue=0, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                inImage[i][k][inImage[i][k] > brt] = 255
                inImage[i][k][inImage[i][k] <= brt] = 0
                outImage[i][k] = inImage[i][k]

    elif num == 13: # 범위강조
        brt1 = askinteger('첫 번째 범위 수', '정수(1~255)', minvalue=0, maxvalue=255)
        brt2 = askinteger('두 번째 범위 수', '정수(1~255)', minvalue=0, maxvalue=255)
        for  i  in  range(inH) :
            for  k  in  range(inW) :
                if inImage[i][k][0] > brt1 and inImage[i][k][0] < brt2: 
                    outImage[i][k][0] = 255
                else: outImage[i][k][0] = inImage[i][k][0]
                if inImage[i][k][1] > brt1 and inImage[i][k][1] < brt2: 
                    outImage[i][k][1] = 255
                else: outImage[i][k][0] = inImage[i][k][0]
                if inImage[i][k][2] > brt1 and inImage[i][k][2] < brt2: 
                    outImage[i][k][2] = 255
                else: outImage[i][k][0] = inImage[i][k][0]    
    display_gif()
    
def a_average_gif(num): # 입력 // 출력 평균
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    sumListR1, sumListG1, sumListB1 = [], [], []
    sumListR2, sumListG2, sumListB2 = [], [], []
    if num == 2:
        brt = askinteger('절사 수치', '정수(1~100)%', minvalue=1, maxvalue=100)
        cut = 255*(brt/100)
    for i in range(inH):
        for k in range(inW):
            if num == 1:
                sumListR1.append(inImage[i][k][0])
                sumListR2.append(outImage[i][k][0])
                sumListG1.append(inImage[i][k][1])
                sumListG2.append(outImage[i][k][1])
                sumListB1.append(inImage[i][k][2])
                sumListB2.append(outImage[i][k][2])
            if num == 2:
                if inImage[i][k][0] > int(cut) and inImage[i][k][0] < 255-int(cut):
                    sumListR1.append(inImage[i][k][0])
                if outImage[i][k][0] > int(cut) and outImage[i][k][0] < 255-int(cut):
                    sumListR2.append(outImage[i][k][0])    
                if inImage[i][k][1] > int(cut) and inImage[i][k][1] < 255-int(cut):
                    sumListG1.append(inImage[i][k][1])
                if outImage[i][k][1] > int(cut) and outImage[i][k][1] < 255-int(cut):
                    sumListG2.append(outImage[i][k][1]) 
                if inImage[i][k][2] > int(cut) and inImage[i][k][2] < 255-int(cut):
                    sumListB1.append(inImage[i][k][2])
                if outImage[i][k][2] > int(cut) and outImage[i][k][2] < 255-int(cut):
                    sumListB2.append(outImage[i][k][2]) 
    inAvgR = sum(sumListR1) / len(sumListR1)
    outAvgR = sum(sumListR2) / len(sumListR2)
    inAvgG = sum(sumListG1) / len(sumListG1)
    outAvgG = sum(sumListG2) / len(sumListG2)
    inAvgB = sum(sumListB1) / len(sumListB1)
    outAvgB = sum(sumListB2) / len(sumListB2)
    subWindow = Toplevel(window)
    subWindow.geometry('500x100')
    label1 = Label(subWindow, text="R입력 평균: " + str(inAvgR)[:7]+ " G입력 평균: " + str(inAvgG)[:7] + " B입력 평균: " + str(inAvgB)[:7])
    label2 = Label(subWindow, text="R출력 평균: " + str(outAvgR)[:7]+ " G출력 평균: " + str(outAvgG)[:7] + " B출력 평균: " + str(outAvgB)[:7])
    label1.pack()
    label2.pack()        

def a_minmax_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    allDictR1, allDictR2, allDictG1, allDictG2, allDictB1, allDictB2 = {}, {}, {}, {}, {}, {}
    for i in range(inH):
        for j in range(inW):
            if inImage[i][j][0] in allDictR1:
                allDictR1[inImage[i][j][0]] += 1
            else:
                allDictR1[inImage[i][j][0]] = 1
            if inImage[i][j][1] in allDictG1:
                allDictG1[inImage[i][j][1]] += 1
            else:
                allDictG1[inImage[i][j][1]] = 1
            if inImage[i][j][2] in allDictB1:
                allDictB1[inImage[i][j][2]] += 1
            else:
                allDictB1[inImage[i][j][2]] = 1
            if outImage[i][j][0] in allDictR2:
                allDictR2[outImage[i][j][0]] += 1
            else:
                allDictR2[outImage[i][j][0]] = 1
            if outImage[i][j][1] in allDictG2:
                allDictG2[outImage[i][j][1]] += 1
            else:
                allDictG2[outImage[i][j][1]] = 1
            if outImage[i][j][2] in allDictB2:
                allDictB2[outImage[i][j][2]] += 1
            else:
                allDictB2[outImage[i][j][2]] = 1
          
    sortListR1 = sorted(allDictR1.items(), key=operator.itemgetter(1))
    sortListR2 = sorted(allDictR2.items(), key=operator.itemgetter(1))
    sortListG1 = sorted(allDictG1.items(), key=operator.itemgetter(1))
    sortListG2 = sorted(allDictG2.items(), key=operator.itemgetter(1))
    sortListB1 = sorted(allDictB1.items(), key=operator.itemgetter(1))
    sortListB2 = sorted(allDictB2.items(), key=operator.itemgetter(1))
    subWindow = Toplevel(window)
    subWindow.geometry('500x100')
    label1 = Label(subWindow, text="R입력 최대&최소: " + str(sortListR1[-1]) + str(sortListR1[0]) + "\n" +
                   " G입력 최대&최소: " + str(sortListG1[-1]) + str(sortListG1[0]) + "\n" +
                   " B입력 최대&최소: " + str(sortListB1[-1]) + str(sortListB1[0]))
    label2 = Label(subWindow, text="R출력 최대&최소: " + str(sortListR2[-1]) + str(sortListR2[0]) + "\n" +
                   " G출력 최대&최소: " + str(sortListG2[-1]) + str(sortListG2[0]) + "\n" +
                   " B출력 최대&최소: " + str(sortListB2[-1]) + str(sortListB2[0]))
    label1.pack()
    label2.pack()

    
def mouseClick_gif(event):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    global sx, sy, ex, ey, panYN
    if not panYN:
        return
    sx = event.x
    sy = event.y

def mouseDrop_gif(event):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    global sx, sy, ex, ey, panYN
    if not panYN:
        return
    ex = event.x
    ey = event.y
    mx = sx - ex
    my = sy - ey
    outW = inW
    outH = inH
    outImage = []
    tmpList = []
    for i in range(outH):
        tmpList = []
        for k in range(outW):
            tmpList.append(np.array([0, 0, 0]))
        outImage.append(tmpList)
    np.array(outImage)
    for  i  in  range(inH) :
        for  k  in  range(inW):
            if 0 < i-mx < outH and 0 < k-my < outW:
                outImage[i-mx][k-my] = inImage[i][k]
    panYN = False
    display_gif()
    
def zoomOut_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger('축소하기', '축소할 배수-->', minvalue=2, maxvalue=32)
    outW = int(inW/scale);  outH = int(inH/scale);
    outImage = [];   tmpList = []
    for i in range(outH):  # 출력메모리 확보(0으로 초기화)
        tmpList = []
        for k in range(outW):
            tmpList.append(np.array([0, 0, 0]))
        outImage.append(tmpList)
    for  i  in  range(inH) :
        for  k  in  range(inW) :
             outImage[int(i/scale)][int(k/scale)] = inImage[i][k]
    display_gif()

def zoomIn_gif(num):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger("확대하기", "배수", minvalue=2, maxvalue=32)
    outW = inW * scale
    outH = outH * scale
    outImage = []
    tmpList = []
    for i in range(outH):
        tmpList = []
        for k in range(outW):
            tmpList.append(np.array([0, 0, 0]))
        outImage.append(tmpList) 
    if num == 1:
        for  i  in  range(inH):
            for  k  in  range(inW):
                outImage[int(i*scale)][int(k*scale)] = inImage[i][k]
    if num == 2:
        for i in range(outH):
            for k in range(outW):
                outImage[int(i)][int(k)] = inImage[int(i/scale)][int(k/scale)]
    display_gif()
    
def embossing_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger("scale(3,5,7,9)", "정수", minvalue=1, maxvalue=10)
    outW, outH = inW, inH
    if int(scale) % 2 == 0: embossing_gif()
    # mask matrix
    mask = np.zeros([scale, scale], dtype=np.int32)
    mask_num = np.array([-1, 1, 0])
    n = 0
    # rgb를 hsi로 미리 모두 변환
    inImage = rgb2hsi(inImage)
    outImage = np.array(outImage, dtype=np.float64)
    for i in range(scale):
        for j in range(scale):
            if i == j:
                mask[i][j] = mask_num[n]
                n += 1
                if n > 2: n = 0 
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            for k in range(3):
                outImage[i][j][k] = np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2), k] * mask)
    # hsi를 rgb로 변환
    outImage = hsi2rgb(outImage) + 127
    outImage[outImage > 255] = 255
    outImage[outImage < 0] = 0
    display_gif()

def blurring_gif(num):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    scale = askinteger("scale(3,5,7,9)", "정수", minvalue=1, maxvalue=10)
    outW, outH = inW, inH
    if int(scale) % 2 == 0: embossing()
    # mask matrix
    mask = np.ones([scale, scale], dtype=np.float32)
    mask = mask * (1/np.square(scale))
    # rgb를 hsi로 미리 모두 변환
    inImage = rgb2hsi(inImage)
    outImage = np.array(outImage, dtype=np.float64)
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            for k in range(3):
                outImage[i][j][k] = np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2), k] * mask)
    # hsi를 rgb로 변환
    outImage = hsi2rgb(outImage)
    outImage[outImage > 255] = 255
    outImage[outImage < 0] = 0
    display_gif()
    
def gausian_blurring_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW, outH = inW, inH
    # mask matrix
    scale = 3
    mask = np.array([[1./16., 1./8., 1./16.], [1./8., 1./4., 1./8.], [1./16., 1./8., 1./16.]])
    # rgb를 hsi로 미리 모두 변환
    inImage = rgb2hsi(inImage)
    outImage = np.array(outImage, dtype=np.float64)
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            for k in range(3):
                outImage[i][j][k] = np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2), k] * mask)
    # hsi를 rgb로 변환
    outImage = hsi2rgb(outImage)
    outImage[outImage > 255] = 255
    outImage[outImage < 0] = 0
    display_gif()

def sharpening_gif(num):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW, outH = inW, inH
    # mask matrix
    scale = 3
    if num == 1:
        mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    if num == 2:
        mask = np.array([[-1./9., -1./9., -1./9.], [-1./9., 8./9., -1./9.], [-1./9., -1./9., -1./9.]])
    # rgb를 hsi로 미리 모두 변환
    inImage = rgb2hsi(inImage)
    outImage = np.array(outImage, dtype=np.float64)
    for i in range(int(scale/2), inH - int(scale/2)):
        for j in range(int(scale/2), inW - int(scale/2)):
            for k in range(3):
                outImage[i][j][k] = np.sum(inImage[i - int(scale/2) : i + scale - int(scale/2), j - int(scale/2) : j + scale - int(scale/2), k] * mask)
     # hsi를 rgb로 변환
    outImage = hsi2rgb(outImage)
    outImage[outImage > 255] = 255
    outImage[outImage < 0] = 0
    display_gif()

def rgb2hsi(rgb):
    '''
    RGB 행렬을 받아 모두 연산처리
    '''
    rgb = np.array(rgb, dtype=np.float64)
    hsi = np.zeros(np.shape(rgb), dtype=np.float64)
    # 2중 for문으로 h, s, i를 각각 연산
    for k in range(np.shape(rgb)[0]):
        for j in range(np.shape(rgb)[1]):
            r, g, b = rgb[k][j][0], rgb[k][j][1], rgb[k][j][2]
            h, s, i = 0, 0, 0
            i = np.mean(rgb[k][j])
            if r == g == b:
                s, h = 0, 0
                hsi[k][j] = [h, s, i]
                continue
            s = 1 - (3 / np.sum(rgb[k][j]) * np.min(rgb[k][j]))
            angle = 0.5 * ((r - g) + (r - b)) / np.sqrt((r - g) * (r - g) + ((r - b) * (g - b)))
            h = np.arccos(angle)
            if b >= g:
                h = np.pi*2 - h
            hsi[k][j] = [h, s, i]
    return hsi

def hsi2rgb(hsi):
    '''
    hsi 행렬을 rgb행렬로 변환
    '''
    rgb = np.zeros(np.shape(hsi), dtype=np.float64)
    # 2중 for문으로 모든 각각의 hsi를 연산
    for k in range(np.shape(hsi)[0]):
        for j in range(np.shape(hsi)[1]):
            h, s, i = hsi[k][j][0], hsi[k][j][1], hsi[k][j][2]
            r, g, b = 0, 0, 0
            if s == 0:
                rgb[k][j] = [i, i, i]
                continue
            elif h <= np.pi/3 * 2:
                b = 1/3 * (1 - s)
                r = 1/3 * (1 + (s * np.cos(h) / np.cos(np.pi/3 - h)))
                g = 1 - (r + b)
            elif np.pi/3 * 2 < h <= np.pi/3 * 4:
                h = h - np.pi/3 * 2
                g = 1/3 * (1 + (s * np.cos(h) / np.cos(np.pi/3 - h)))
                r = 1/3 * (1 - s)
                b = 1 - (r + g)
            elif np.pi/3 * 4 < h <= np.pi * 2:
                h = h - np.pi/3 * 4
                g = 1/3 * (1 - s)
                b = 1/3 * (1 + (s * np.cos(h) / np.cos(np.pi/3 - h)))
                r = 1 - g - b
            rgb[k][j] = np.array([r, g, b]) * i * 3
    return np.array(rgb, dtype=np.int32)
    
def saveRGBCSV():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    output_file = asksaveasfile(parent=window, mode='w',
                                defaultextension="*.csv", filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    output_file = output_file.name
    header = ['Column', 'Row', 'Value']
    with open(output_file, 'w', newline='') as filewriter:
        csvWriter = csv.writer(filewriter)
        csvWriter.writerow(header)
        for row in range(outW):
            for col in range(outH):
                data = outImage[row][col]
                row_list = [row, col, data]
                csvWriter.writerow(row_list)
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()

def loadCSV(fname):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    fsize = -1
    fp = open(fname, 'r')
    for f in fp:
        fsize += 1
    fp.close()
    inH = inW = int(math.sqrt(fsize))  # 입력메모리 크기 결정! (중요)
    inImage = [];
    tmpList = []
    for i in range(inH):  # 입력메모리 확보(0으로 초기화)
        tmpList = []
        for k in range(inW):
            tmpList.append(0)
        inImage.append(tmpList)
    # 파일 --> 메모리로 데이터 로딩
    fp = open(fname, 'r')  # 파일 열기(바이너리 모드)
    csvFP = csv.reader(fp)
    next(csvFP)
    for row_list in csvFP:
        row = int(row_list[0])
        col = int(row_list[1])
        if len(row_list[2]) > 4:
            value = row_list[2][1:-1].split()  # string 상태의 RGB LIST를 다시 LIST 형태로 변환
            value = np.array(value, dtype=np.int32) # string을 int32으로 타입 변환
            global gif
            gif = True
        else:
            value = int(row_list[2])
        inImage[row][col] = value
    fp.close()
    
def openCSV():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    filename = askopenfilename(parent=window,
                               filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    loadCSV(filename)  # 파일 --> 입력메모리
    equal()  # 입력메모리--> 출력메모리
    
def saveSQLite():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    global input_file
    saveFp = asksaveasfilename(parent=window)
    con = sqlite3.connect(saveFp)  # 데이터베이스 지정(또는 연결)
    cur = con.cursor()  # 연결 통로 생성 (쿼리문을 날릴 통로)
    # 열이름 리스트 만들기
    inImage = outImage.copy()
    fname = os.path.basename(filename).split(".")[0]
    try:
        sql = "DELETE FROM imageTable WHERE filename = '" + fname + "'"
        cur.execute(sql)
    except:
        pass
    try:
        sql = "CREATE TABLE imageTable(filename CHAR(20), resolution smallint" + \
              ", row  smallint,  col  smallint, value CHAR(20))"
        cur.execute(sql)
        con.commit()
    except:
        pass
    for i in range(inW):
        for k in range(inH):
            sql = "INSERT INTO imageTable VALUES('" + fname + "'," + str(inW) + \
                  "," + str(i) + "," + str(k) + "," + "'" + str(inImage[i][k]) + "'" + ")"
            cur.execute(sql)  # str은 ' ' 앞뒤로 중요 (query)
    con.commit()
    cur.close()
    con.close()  # 데이터베이스 연결 종료
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def loadSQLite():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    openFp = askopenfilename(parent=window)
    con = sqlite3.connect(openFp)  # 데이터베이스 지정(또는 연결)
    cur = con.cursor()  # 연결 통로 생성 (쿼리문을 날릴 통로)
    inImage = []
    try:
        sql = "SELECT DISTINCT filename, resolution FROM imageTable"
        cur.execute(sql)
        tableNameList = []
        while True:
            row = cur.fetchone()
            if row == None:
                break
            tableNameList.append(row[0] + ":" + str(row[1]))
        ##
        def selectTable():
            global window, filename, inImage, inW, inH, outW, outH
            index = listbox.curselection()[0]
            subWindow.destroy()
            fname, res = tableNameList[index].split(':')
            filename = fname
            sql = "SELECT * FROM imageTable WHERE filename = " + "'" + fname + "'"
            cur.execute(sql)
            row = cur.fetchone()
            inW = inH = int(row[1])
            tmpList = []
            for i in range(inH):
                tmpList = []
                for k in range(inW):
                    tmpList.append(0)
                inImage.append(tmpList)
            for i in range(inW * inH):
                if len(row[4]) > 4:
                    inImage[int(row[2])][int(row[3])] = row[4][1:-1].split() # str을 다시 list로 (rgb)
                    global gif
                    gif = True
                else:
                    inImage[int(row[2])][int(row[3])] = int(row[4])
                row = cur.fetchone()
            inImage = np.array(inImage, dtype=np.int32)
            cur.close()
            con.close()
            equal()
            ##
        subWindow = Toplevel(window)
        listbox = Listbox(subWindow)
        button = Button(subWindow, text='선택', command=selectTable)
        listbox.pack()
        button.pack()
        for sName in tableNameList:
            listbox.insert(END, sName)
        subWindow.lift()
        ##
    except:
        cur.close()
        con.close()
        print("error")
        ##

def savemySql():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    # DB쿼리 / imageDB 생성
    ip = askstring("ip주소", "192.168.226.131")
    userName = askstring("user name", "DB에서 생성된 유저")
    password = askstring("password", "password: 1234")
    db = askstring("DB name", "사용할 DB")
    con = pymysql.connect(host=ip, user=userName, password=password, db=db, charset='utf8')
    cur = con.cursor()  # 연결 통로 생성 (쿼리문을 날릴 통로)
    # 열이름 리스트 만들기
    inImage = outImage.copy()  # *변화된 outImage를 inImage로 변환
    fname = os.path.basename(filename).split(".")[0]
    sql = "DELETE FROM imageTable WHERE filename = '" + fname + "'"  # 기존 데이터 삭제
    try:
        #        print(sql)
        cur.execute(sql)
        con.commit()  # commit 중요
    except:
        pass
    try:
        sql = "CREATE TABLE imageTable( filename CHAR(20), resolution smallint" + \
              ", row  smallint,  col  smallint, value  char(20))"
        cur.execute(sql)
    except:
        pass

    for i in range(inW):
        for k in range(inH):
            sql = "INSERT INTO imageTable VALUES('" + fname + "'," + str(inW) + \
                  "," + str(i) + "," + str(k) + "," + "'" + str(inImage[i][k]) + "'" + ")"
            cur.execute(sql)
    con.commit()
    cur.close()
    con.close()  # 데이터베이스 연결 종료
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def loadmySql():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    # pymysql 연결 // linux >> DB 연결
    ip = askstring("ip주소", "192.168.226.131")
    userName = askstring("user name", "DB에서 생성된 유저")
    password = askstring("password", "password: 1234")
    db = askstring("DB name", "사용할 DB")
    con = pymysql.connect(host=ip, user=userName, password=password, db=db, charset='utf8')
    cur = con.cursor()  # 연결 통로 생성 (쿼리문을 날릴 통로)
    inImage = []
    try:
        # 중복치 제거 후 수집 >> distinct
        sql = "SELECT DISTINCT filename, resolution FROM imageTable"
        cur.execute(sql)
        tableNameList = []
        while True:
            row = cur.fetchone()
            if row == None:
                break
            tableNameList.append(row[0] + ":" + str(row[1]))
        ##
        def selectTable():
            global window, filename, inImage, inW, inH, outW, outH
            index = listbox.curselection()[0]
            subWindow.destroy()
            fname, res = tableNameList[index].split(':')
            filename = fname
            sql = "SELECT * FROM imageTable WHERE filename = " + "'" + fname + "'"
            cur.execute(sql)
            row = cur.fetchone()
            inW = inH = int(row[1])
            tmpList = []
            for i in range(inH):
                tmpList = []
                for k in range(inW):
                    tmpList.append(0)
                inImage.append(tmpList)
            for i in range(inW * inH):
                # ROW = [filename, resolution, rownum, colnum, grayscale]
                if len(row[4]) > 4:
                    inImage[int(row[2])][int(row[3])] = row[4][1:-1].split()
                    global gif
                    gif = True
                else:
                    inImage[int(row[2])][int(row[3])] = int(row[4])
                row = cur.fetchone()
            inImage = np.array(inImage, dtype=np.int32)
            cur.close()
            con.close()
            equal()
            ##
        subWindow = Toplevel(window)
        listbox = Listbox(subWindow)
        button = Button(subWindow, text='선택', command=selectTable)
        listbox.pack()
        button.pack()
        for sName in tableNameList:
            listbox.insert(END, sName)
        subWindow.lift()
        ##
    except:
        cur.close()
        con.close()
        print("error")

def sqlExcel1():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    # save할 파일 결정
    outfilename = asksaveasfile(parent=window, mode='wb',
                                defaultextension="*.xlsx", filetypes=(("xlsx파일", "*.xlsx"), ("모든파일", "*.*")))
    wb = Workbook(outfilename)
    ws = wb.add_worksheet(os.path.basename(filename))
    with open(filename, 'rb') as fReader:
        for i in range(inW):
            for j in range(inH):
                data = inImage[i][j]  # 저장되어 있던 inImage에서 data 추출
                ws.write(i, j, str(data))  # index마다 쓰기
    wb.close()
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def sqlExcel2():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    outfilename = asksaveasfile(parent=window, mode='wb',
                                defaultextension="*.xlsx", filetypes=(("xlsx파일", "*.xlsx"), ("모든파일", "*.*")))
    wb = Workbook(outfilename)
    ws = wb.add_worksheet(os.path.basename(filename))
    with open(filename, 'rb') as fReader:
        # 워크시트의 열 너비 / 행 높이 지정
        ws.set_column(0, inW, 1.0)  # 0.34
        for row in range(inH):
            ws.set_row(row, 9.5)  # 0.35
        for i in range(inW):
            for j in range(inH):
                data = inImage[i][j]
                if data  or 'str' in type(data):
                    if data > 15:
                        hexStr = '#' + (hex(data)[2:]) * 3
                    else:
                        hexStr = '#' + ('0' + hex(data)[2:]) * 3
                else:
                    if data[0] <= 15:  # 15 이하일 경우, 1자리 수이기 때문에 0을 추가
                        hexStr = '#' + ('0' + hex(data[0])[2:])
                    else:
                        hexStr = '#' + (hex(data[0])[2:])  # 16진수 변환 후, R(2자리)
                    if data[1] <= 15:
                        hexStr += ('0' + hex(data[1])[2:])  # G(2자리)
                    else:
                        hexStr += hex(data[1])[2:]
                    if data[2] <= 15:
                        hexStr += ('0' + hex(data[2])[2:])  # B(2자리)
                    else:
                        hexStr += hex(data[2])[2:]
                cell_format = wb.add_format()  # RGB코드는 #을 앞에
                cell_format.set_bg_color(hexStr)
                ws.write(i, j, '', cell_format)
    wb.close()
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()

def sqlExcel3():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    inImage = []
    input_file = askopenfilename(parent=window, filetypes=(("EXCEL파일", "*.xls;*.xlsx"), ("모든파일", "*.*")))
    workbook = xlrd.open_workbook(input_file)
    sheetCount = workbook.nsheets
    for sheet in workbook.sheets():
        sRow = sheet.nrows
        sCol = sheet.ncols
        for i in range(sRow):
            tmpList = []
            for j in range(sCol):
                if len(sheet.cell_value(i,j)) > 4:
                    value = sheet.cell_value(i, j)[1:-1].split()
                    global gif
                    gif = True
                else:
                    value = sheet.cell_value(i, j)
                tmpList.append(value)
            inImage.append(tmpList)
    inImage = np.array(inImage, dtype=np.int32)
    inW = len(inImage)
    inH = len(inImage[0])
    equal()
    
def a_histogram_plt():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    if gif:
        a_histogram_plt_gif()
        return
    countList = [0] * 256
    normList = [0] * 256
    for i in range(outH):
        for k in range(outW):
            value = outImage[i][k]
            countList[value] += 1
    plt.plot(countList)
    plt.show()
    
def a_histogram_plt_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    countListR, countListG, countListB = [0] * 256, [0] * 256, [0] * 256
    for i in range(outH):
        for k in range(outW):
            r, g, b = outImage[i][k][0], outImage[i][k][1], outImage[i][k][2]
            countListR[r] += 1
            countListG[g] += 1
            countListB[b] += 1
    plt.ion()
    plt.plot(countListR, 'r')
    plt.plot(countListG, 'g')
    plt.plot(countListB, 'b')
    plt.show()
    

def stretch():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    if gif:
        stretch_gif()
        return
    maxVal, minVal, high = 0, 255, 255
    for i in range(inH):
        for j in range(inW):
            data = inImage[i][j]
            if data > maxVal:
                maxVal = data
            if data < minVal:
                minVal = data
    for i in range(inH):
        for j in range(inW):
            value = int((inImage[i][j] - minVal) / (maxVal - minVal) * high)
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            inImage[i][j] = value
    equal()

def stretch_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    maxValR, maxValG, maxValB, minValR, minValG, minValB, high = 0, 0, 0, 255, 255, 255, 255
    for i in range(inH):
        for j in range(inW):
            dataR, dataG, dataB = inImage[i][j][0], inImage[i][j][1], inImage[i][j][2]
            if dataR > maxValR:
                maxValR = dataR
            if dataG > maxValG:
                maxValG = dataG
            if dataB > maxValB:
                maxValB = dataB
            if dataR < minValR:
                minValR = dataR
            if dataG < minValG:
                minValG = dataG
            if dataB < minValB:
                minValB = dataB
    for i in range(inH):
        for j in range(inW):
            valueR = int((inImage[i][j][0] - minValR) / (maxValR - minValR) * high)
            valueG = int((inImage[i][j][1] - minValG) / (maxValG - minValG) * high)
            valueB = int((inImage[i][j][2] - minValB) / (maxValB - minValB) * high)
            outImage[i][j] = [valueR, valueG, valueB]
            outImage[i][j][outImage[i][j] > 255] = 255
            outImage[i][j][outImage[i][j] < 0] = 0
    display_gif()
    
def endin():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    if gif:
        endin_gif()
        return
    maxVal, minVal, high = 0, 255, 255
    for i in range(inH):
        for j in range(inW):
            data = inImage[i][j]
            if data > maxVal:
                maxVal = data
            if data < minVal:
                minVal = data
    limit = askinteger('endin', '범위', minvalue=1, maxvalue=127)
    maxVal -= limit
    minVal += limit
    for i in range(inH):
        for j in range(inW):
            value = int((inImage[i][j] - minVal) / (maxVal - minVal) * high)
            if value < 0:
                value = 0
            elif value > 255:
                value = 255
            inImage[i][j] = value
    equal()
    
def endin_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    maxValR, maxValG, maxValB, minValR, minValG, minValB, high = 0, 0, 0, 255, 255, 255, 255
    for i in range(inH):
        for j in range(inW):
            dataR, dataG, dataB = inImage[i][j][0], inImage[i][j][1], inImage[i][j][2]
            if dataR > maxValR:
                maxValR = dataR
            if dataG > maxValG:
                maxValG = dataG
            if dataB > maxValB:
                maxValB = dataB
            if dataR < minValR:
                minValR = dataR
            if dataG < minValG:
                minValG = dataG
            if dataB < minValB:
                minValB = dataB
    limit = askinteger('endin', '범위', minvalue=1, maxvalue=127)
    maxValR -= limit
    maxValG -= limit
    maxValB -= limit
    minValR += limit
    minValG += limit
    minValB += limit
    for i in range(inH):
        for j in range(inW):
            valueR = int((inImage[i][j][0] - minValR) / (maxValR - minValR) * high)
            valueG = int((inImage[i][j][1] - minValG) / (maxValG - minValG) * high)
            valueB = int((inImage[i][j][2] - minValB) / (maxValB - minValB) * high)
            outImage[i][j] = [valueR, valueG, valueB]
            outImage[i][j][outImage[i][j] > 255] = 255
            outImage[i][j][outImage[i][j] < 0] = 0
    display_gif()
    
def equalize():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    if gif:
        equalize_gif()
        return
    maxVal, minVal, high = 0, 255, 255
    hist = [0] * 255;
    cum = [0] * 255;
    norm = [0] * 255
    outW = inW;
    outH = inH;
    outImage = [];
    tmpList = []
    for i in range(outH):  # 출력메모리 확보(0으로 초기화)
        tmpList = []
        for k in range(outW):
            tmpList.append(0)
        outImage.append(tmpList)
    for i in range(inH):
        for j in range(inW):
            value = inImage[i][j]
            hist[value] += 1
    sVal = 0
    for i in range(len(hist)):
        sVal += hist[i]
        cum[i] = sVal
    for i in range(len(cum)):
        norm[i] = cum[i] / (outW * outH) * high
    for i in range(inH):
        for j in range(inW):
            index = inImage[i][j]
            outImage[i][j] = int(norm[index])
            print(inImage[i][j], outImage[i][j])
            if outImage[i][j] < 0:
                outImage[i][j] = 0
            elif outImage[i][j] > 255:
                outImage[i][j] = 255
    display()
    
def equalize_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    maxValR, maxValG, maxValB, minValR, minValG, minValB, high = 0, 0, 0, 255, 255, 255, 255
    histR, histG, histB = [0] * 256, [0] * 256, [0] * 256
    cumR, cumG, cumB = [0] * 256, [0] * 256, [0] * 256
    normR, normG, normB = [0] * 256, [0] * 256, [0] * 256
    outW = inW;  outH = inH;
    outImage = [];   tmpList = []
    for i in range(outH):  # 출력메모리 확보(0으로 초기화)
        tmpList = []
        for k in range(outW):
            tmpList.append([0, 0, 0])
        outImage.append(tmpList)
    for i in range(outH):
        for k in range(outW):
            r, g, b = inImage[i][k][0], inImage[i][k][1], inImage[i][k][2]
            histR[r] += 1
            histG[g] += 1
            histB[b] += 1
    sValR, sValG, sValB = 0, 0, 0
    for i in range(len(histR)):
        sValR += histR[i]
        sValG += histG[i]
        sValB += histB[i]
        cumR[i] = sValR
        cumG[i] = sValG
        cumB[i] = sValB
    for i in range(len(cumR)):
        normR[i] = cumR[i] / (outW * outH) * high
        normG[i] = cumG[i] / (outW * outH) * high
        normB[i] = cumB[i] / (outW * outH) * high
    for i in range(inH):
        for j in range(inW):
            index = inImage[i][j]
            outImage[i][j] = np.array([int(normR[index[0]]), int(normG[index[1]]), int(normB[index[2]])], dtype=np.int32)
            outImage[i][j][outImage[i][j] > 255] = 255
            outImage[i][j][outImage[i][j] < 0] = 0
    display_gif()
    
def rotate():
    global inImage, outImage, inH, inW, outH, outW, window, canvas, paper, filename
    if gif:
        rotate_gif()
        return
    degree = askinteger('각도', '값 입력', minvalue=0, maxvalue=360)
    # 출력 파일의 크기 결정.
    radian90 = (90 - degree) * np.pi / 180.0
    radian = degree * np.pi / 180.0
    outW = int(inH * math.cos(radian90) + inW * math.cos(radian))
    outH = int(inH * math.cos(radian) + inW * math.cos(radian90))
    # outW = inW; outH = inH
    # 출력 영상 메모리 확보
    outImage = []
    for i in range(0, outW):
        tmpList = []
        for k in range(0, outH):
            tmpList.append(0)
        outImage.append(tmpList)
    # inImage2 크기를 outImage와 동일하게
    inImage2 = []
    for i in range(0, outW):
        tmpList = []
        for k in range(0, outH):
            tmpList.append(255)
        inImage2.append(tmpList)
    # inImage --> inImage2의 중앙으로
    gap = int((outW - inW) / 2)
    for i in range(0, inW):
        for k in range(0, inH):
            inImage2[i + gap][k + gap] = inImage[i][k]
    cx = int(outW / 2)
    cy = int(outH / 2)
    for i in range(0, outW):
        for k in range(0, outH):
            xs = i
            ys = k
            xd = int(math.cos(radian) * (xs - cx)
                     - math.sin(radian) * (ys - cy)) + cx
            yd = int(math.sin(radian) * (xs - cx)
                     + math.cos(radian) * (ys - cy)) + cy
            if 0 <= xd < outW and 0 <= yd < outH:
                outImage[xs][ys] = inImage2[xd][yd]
            else:
                outImage[xs][ys] = 255
    display()
    
def rotate_gif():
    global inImage, outImage, inH, inW, outH, outW, window, canvas, paper, filename
    degree = askinteger('각도', '값 입력', minvalue=0, maxvalue=360)
    # 출력 파일의 크기 결정.
    radian90 = (90 - degree) * np.pi / 180.0
    radian = degree * np.pi / 180.0
    outW = int(inH * math.cos(radian90) + inW * math.cos(radian))
    outH = int(inH * math.cos(radian) + inW * math.cos(radian90))
    # outW = inW; outH = inH
    # 출력 영상 메모리 확보
    outImage = []
    for i in range(0, outW):
        tmpList = []
        for k in range(0, outH):
            tmpList.append([0, 0, 0])
        outImage.append(tmpList)
    # inImage2 크기를 outImage와 동일하게
    inImage2 = []
    for i in range(0, outW):
        tmpList = []
        for k in range(0, outH):
            tmpList.append([255, 255, 255])
        inImage2.append(tmpList)
    # inImage --> inImage2의 중앙으로
    gap = int((outW - inW) / 2)
    for i in range(0, inW):
        for k in range(0, inH):
            inImage2[i + gap][k + gap] = inImage[i][k]
    cx = int(outW / 2)
    cy = int(outH / 2)
    for i in range(0, outW):
        for k in range(0, outH):
            xs = i
            ys = k
            xd = int(math.cos(radian) * (xs - cx)
                     - math.sin(radian) * (ys - cy)) + cx
            yd = int(math.sin(radian) * (xs - cx)
                     + math.cos(radian) * (ys - cy)) + cy
            if 0 <= xd < outW and 0 <= yd < outH:
                outImage[xs][ys] = inImage2[xd][yd]
            else:
                outImage[xs][ys] = [255, 255, 255]
    display_gif()
    
def morphing():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    if gif:
        morphing_gif()
        return
    outW, outH = inW, inH
    filename2 = askopenfilename(parent=window,
                               filetypes=(("RAW파일", "*.raw"), ("모든파일", "*.*")))
    if filename2 == '' or filename2 == None:
        return
    inImage2 = []
    fsize2 = os.path.getsize(filename2)
    inH2 = inW2 = int(math.sqrt(fsize2))
    if inH2 != inH:
        return
    outImage = []
    fp2 = open(filename2, 'rb')
    for i in range(outH):
        tmpList = []
        for j in range(outW):
            data = int(ord(fp2.read(1)))
            tmpList.append(data)
        inImage2.append(tmpList)
    fp2.close()
    for i in range(outH):
        tmpList = []
        for j in range(outW):
            tmpList.append(0)
        outImage.append(tmpList)
    value = askinteger('합성 비율', '두 번째 영상 가중치', minvalue=1, maxvalue=100)
    w1 = (1 - value/100)
    w2 = value/100
    for i in range(outH):
        for j in range(outW):
            data = int(inImage[i][j] * w1) + int(inImage2[i][j] * w2)
            if data > 255:
                data = 255
            elif data < 0:
                data = 0
            outImage[i][j] = data
    display()
    
def morphing_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH
    outW, outH = inW, inH
    filename2 = askopenfilename(parent=window,
                               filetypes=(("그림파일", "*.raw;*.gif;*.jpg;*.png;*.tif;*.bmp"), ("모든파일", "*.*")))
    if filename2 == '' or filename2 == None:
        return
    photo = Image(filename=filename2)
    inImage2 = []
    fsize2 = os.path.getsize(filename2)
    inH2 = inW2 = inW
    outImage = []
    fp2 = open(filename2, 'rb')
    for i in range(inH):
        tmpList = []
        for k in range(inW) :
            tmpList.append(np.array([0, 0, 0]))
        inImage2.append(tmpList)
    blob = photo.make_blob(format='RGB')  
    for  i  in range(inH):
        for  k  in  range(inW):
            r, g, b = blob[(i * 3 * inH) + (k * 3) + 0], blob[(i * 3 * inH) + (k * 3) + 1], blob[
                (i * 3 * inH) + (k * 3) + 2]
            inImage2[i][k] = [r, g, b]
    inImage2 = np.array(inImage2)
    fp2.close()
    for i in range(outH):
        tmpList = []
        for j in range(outW):
            tmpList.append([0, 0, 0])
        outImage.append(tmpList)
    value = askinteger('합성 비율', '두 번째 영상 가중치', minvalue=1, maxvalue=100)
    w1 = (1 - value/100)
    w2 = value/100
    for i in range(outH):
        for j in range(outW):
            data = [int(inImage[i][j][0] * w1) + int(inImage2[i][j][1] * w2),
                    int(inImage[i][j][1] * w1) + int(inImage2[i][j][1] * w2),
                    int(inImage[i][j][2] * w1) + int(inImage2[i][j][2] * w2)]
            data = np.array(data, dtype=np.int32)
            data[data < 0] = 0
            data[data > 255] = 255
            outImage[i][j] = data
    display_gif()
    photo = None

def bigData01():
    '''
    폴더 안의 raw 파일들을 모두 DB로 저장
    '''
    global window, canvas, paper, filename, inImage, inW, inH
    saveFp = asksaveasfilename(parent=window)
    con = sqlite3.connect(saveFp)
    cur = con.cursor()
    dirName = askdirectory()
    file_list = []
    for i in range(6):
        file_list = glob.glob(os.path.join(dirName, "*.raw"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.png"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.jpg"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.bmp"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.tif"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.gif"))
    for input_file in file_list:
        if input_file[-3:] != "raw":
            global gif
            gif = True
            filename = input_file
            loadImage_gif(input_file)
        else:
            filename = input_file
            loadImage(input_file)
        fsize = os.path.getsize(input_file)  # raw파일 size
        fname = os.path.basename(filename).split(".")[0]
        try:
            sql = "DELETE FROM imageTable WHERE filename = '" + fname + "'"
            cur.execute(sql)
        except:
            pass
        try:
            sql = "CREATE TABLE imageTable(filename CHAR(20), resolution smallint" + \
                  ", row  smallint,  col  smallint, value CHAR(20))"
            cur.execute(sql)
            con.commit()
        except:
            pass
        for i in range(inW):
            for k in range(inH):
                if gif:
                    sql = "INSERT INTO imageTable VALUES('" + fname + "'," + str(inW) + \
                    "," + str(i) + "," + str(k) + "," + "'" + str(inImage[i][k]) + "'" + ")"
                else:
                    sql = "INSERT INTO imageTable VALUES('" + fname + "'," + str(inW) + \
                    "," + str(i) + "," + str(k) + "," + str(inImage[i][k]) + ")"
                cur.execute(sql)  # str은 ' ' 앞뒤로 중요 (query)
    con.commit()
    cur.close()
    con.close()  # 데이터베이스 연결 종료
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def bigData02():
    global window, canvas, paper, filename, inImage, inW, inH
    ip = askstring("ip주소", "192.168.226.131")
    userName = askstring("user name", "DB에서 생성된 유저")
    password = askstring("password", "password: 1234")
    db = askstring("DB name", "사용할 DB")
    con = pymysql.connect(host=ip, user=userName, password=password, db=db, charset='utf8')
    cur = con.cursor()
    dirName = askdirectory()
    file_list = []
    for i in range(6):
        file_list = glob.glob(os.path.join(dirName, "*.raw"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.png"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.jpg"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.bmp"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.tif"))
        if file_list != []: break
        file_list = glob.glob(os.path.join(dirName, "*.gif"))
    for input_file in file_list:
        if input_file[-3:] != "raw":
            global gif
            gif = True
            filename = input_file
            loadImage_gif(input_file)
        else:
            filename = input_file
            loadImage(input_file)
        fsize = os.path.getsize(input_file)  # raw파일 size
        fname = os.path.basename(filename).split(".")[0]
        try:
            sql = "DELETE FROM imageTable WHERE filename = '" + fname + "'"
            cur.execute(sql)
        except:
            pass
        try:
            sql = "CREATE TABLE imageTable(filename CHAR(20), resolution smallint" + \
                  ", row  smallint,  col  smallint, value CHAR(20))"
            cur.execute(sql)
            con.commit()
        except:
            pass
        for i in range(inW):
            for k in range(inH):
                if gif:
                    sql = "INSERT INTO imageTable VALUES('" + fname + "'," + str(inW) + \
                    "," + str(i) + "," + str(k) + "," + "'" + str(inImage[i][k]) + "'" + ")"
                else:
                    sql = "INSERT INTO imageTable VALUES('" + fname + "'," + str(inW) + \
                    "," + str(i) + "," + str(k) + "," + str(inImage[i][k]) + ")"
                cur.execute(sql)  # str은 ' ' 앞뒤로 중요 (query)
    con.commit()
    cur.close()
    con.close()  # 데이터베이스 연결 종료
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def drawSheet(cList) :
    global csvList, cellList, window
    if cellList == None or cellList == [] :
        pass
    else :
        for row in cellList:
            for col in row:
                col.destroy()
    rowNum = len(cList)
    colNum = len(cList[0])
    cellList = []
    # 빈 시트 만들기
    global status
    status.destroy()
    window.geometry(str(rowNum*60) + 'x' + str(colNum*60))
    for i in range(0, rowNum):
        tmpList = []
        for k in range(0, colNum):
            ent = Entry(window, text='')
            tmpList.append(ent)
            ent.grid(row=i, column=k)
        cellList.append(tmpList)
    # 시트에 리스트값 채우기. (= 각 엔트리에 값 넣기)
    for i in range(0, rowNum):
        for k in range(0, colNum):
            cellList[i][k].insert(0, cList[i][k])

def openCSV() :
    global  csvList, filename
    csvList = []
    filename = askopenfilename(parent=window,
                filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    filereader = open(filename, 'r', newline='')
    header = filereader.readline()
    header = header.strip()  # 앞뒤 공백제거
    header_list = header.split(',')
    csvList.append(header_list)
    for row in filereader:  # 모든행은 row에 넣고 돌리기.
        row = row.strip()
        row_list = row.split(',')
        csvList.append(row_list)
    drawSheet(csvList)
    filereader.close()

def  saveCSV() :
    global csvList, filename
    if csvList == [] :
        return
    saveFp = asksaveasfile(parent=window, mode='w', defaultextension='.csv',
               filetypes=(("CSV파일", "*.csv"), ("모든파일", "*.*")))
    filewriter = open(saveFp.name, 'w', newline='')
    csvWrite = csv.writer(filewriter)
    for  row_list  in  csvList :
        csvWrite.writerow(row_list)
    
def openJSON() :
    global  csvList, filename
    inImage = []
    filename = askopenfilename(parent=window,
                filetypes=(("JSON파일", "*.json"), ("모든파일", "*.*")))
    filereader = open(filename, 'r', newline='', encoding='utf-8')
    jsonDic = json.load(filereader)
    csvName = list(jsonDic.keys())
    jsonList = jsonDic[csvName[0]]
    # 헤더 추출
    header_list = list(jsonList[0].keys())
    csvList.append(header_list)
    # 행들 추출
    for tmpDic in jsonList:
        tmpList = []
        for header in header_list:
            data = tmpDic[header]
            tmpList.append(data)
        inImage.append(tmpList)
    drawSheet(csvList)
    filereader.close()
    
def  saveJSON() :
    global csvList, filename
    if csvList == [] :
        return
    saveFp = asksaveasfile(parent=window, mode='w', defaultextension='.json',
               filetypes=(("JSON파일", "*.json"), ("모든파일", "*.*")))
    filewriter = open(saveFp.name, 'w', newline='')
    # csvList --> jsonDic
    fname = os.path.basename(filename).split(".")[0]
    jsonDic = {}
    jsonList = []
    tmpDic = {}
    header_list = csvList[0]
    for i in range(1, len(csvList)) :
        rowList = csvList[i]
        tmpDic = {}
        for k in range(0, len(rowList)) :
            tmpDic[header_list[k]] = rowList[k]
        jsonList.append(tmpDic)
    jsonDic[fname] = jsonList
    json.dump(jsonDic, filewriter, indent=4)
    filewriter.close()

def openExcel() :
    global csvList, filename
    csvList = []
    filename = askopenfilename(parent=window,
      filetypes=(("엑셀파일", "*.xls;*.xlsx"), ("모든파일", "*.*")))
    workbook = xlrd.open_workbook(filename)
    sheetCount = workbook.nsheets  # 속성
    sheet1 = workbook.sheets()[0]
    sheetName = sheet1.name
    sRow = sheet1.nrows
    sCol = sheet1.ncols
    #print(sheetName, sRow, sCol)
    for i  in range(sRow) :
        tmpList = []
        for k in range(sCol) :
            value = sheet1.cell_value(i,k)
            tmpList.append(value)
        csvList.append(tmpList)
    drawSheet(csvList)

def saveExcel() :
    global csvList, filename
    if csvList == [] :
        return
    saveFp = asksaveasfile(parent=window, mode='w', defaultextension='.xls',
               filetypes=(("Excel파일", "*.xls"), ("모든파일", "*.*")))
    filename = saveFp.name
    outWorkbook = xlwt.Workbook()
    outSheet = outWorkbook.add_sheet('sheet1') # 이름을 추후에 지정하세요.
    for i in range(len(csvList)) :
        for k in range(len(csvList[i])) :
            outSheet.write(i,k, csvList[i][k])
    outWorkbook.save(filename)
    
def sqliteData01() :
    global csvList, filename
    saveFp = askopenfilename(parent=window)
    con = sqlite3.connect(saveFp)  # 데이터베이스 지정(또는 연결)
    cur = con.cursor()  # 연결 통로 생성 (쿼리문을 날릴 통로)
    csvList = []
    sql = "SELECT name FROM sqlite_master WHERE type='table'"
    cur.execute(sql)
    tableNameList = []
    while True :
        row = cur.fetchone()
        if row == None:
            break
        tableNameList.append(row[0]);
    def selectTable() :
        selectedIndex = listbox.curselection()[0]
        subWindow.destroy()
        # 테이블의 열 목록 뽑기
        # print(colNameList)
        #colNameList = ["userID", "userName", "userAge"]
        #csvList.append(colNameList)
        sql = "SELECT * FROM " + tableNameList[selectedIndex]
        cur.execute(sql)
        while True:
            row = cur.fetchone()
            if row == None:
                break
            row_list = []
            for ii in range(len(row)) :
                row_list.append(row[ii])
            csvList.append(row_list)
            drawSheet(csvList)
    subWindow = Toplevel(window)  # window의 하위로 지정
    listbox = Listbox(subWindow)
    button = Button(subWindow, text='선택', command=selectTable)
    listbox.pack(); button.pack()
    for  sName in tableNameList :
        listbox.insert(END, sName)
    subWindow.lift()

def sqliteData02() :
    global csvList, filename
    saveFp = asksaveasfilename(parent=window)
    con = sqlite3.connect(saveFp) # 데이터베이스 지정(또는 연결)
    cur = con.cursor()  # 연결 통로 생성 (쿼리문을 날릴 통로)
    # 열이름 리스트 만들기
    colList = []
    for data in csvList[0] :
        colList.append(data.replace(' ', ''))
    tableName = os.path.basename(filename).split(".")[0]
    try:
        sql = "CREATE TABLE " + tableName + "("
        for colName in colList :
            sql += colName + " CHAR(20),"
        sql = sql[:-1]
        sql += ")"
        cur.execute(sql)
    except:
        pass
    for i in range(1, len(csvList)) :
        rowList = csvList[i]
        sql = "INSERT INTO " +  tableName + " VALUES("
        for row in rowList:
            sql += "'" + row + "',"
        sql = sql[:-1]
        sql += ")"
        cur.execute(sql)
    con.commit()
    cur.close()
    con.close()  # 데이터베이스 연결 종료
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def mySqlData01() :
    global csvList, filename
    csvList = []
    ip = askstring("ip주소", "192.168.226.131")
    userName = askstring("user name", "DB에서 생성된 유저")
    password = askstring("password", "password: 1234")
    db = askstring("DB name", "사용할 DB")
    con = pymysql.connect(host=ip, user=userName, password=password, db=db, charset='utf8')
    cur = con.cursor()
    sql = "SELECT table_name FROM information_schema.tables WHERE table_schema = '" + db + "'"
    cur.execute(sql)
    tableNameList = []
    while True:
        row = cur.fetchone()
        if row == None:
            break
        tableNameList.append(row[0])
    def selectTable():
        index = listbox.curselection()[0]
        subWindow.destroy()
        sql = "SELECT * FROM "
        sql += tableNameList[index]
        cur.execute(sql)
        col = cur.description
        colNameList = []
        for cName in col:
            colNameList.append(cName[0])
        csvList.append(colNameList)
        while True:
            row = cur.fetchone()
            if row == None:
                break
            csvList.append(list(row))
        cur.close()
        con.close()
        drawSheet(csvList)
    subWindow = Toplevel(window) # window의 하위 지정
    listbox = Listbox(subWindow)
    button = Button(subWindow, text="선택", command=selectTable)
    listbox.pack()
    button.pack()
    for tName in tableNameList:
        listbox.insert(END, tName)
    subWindow.lift()
    
def mySqlData02():
    global csvList, filename
    ip = askstring("ip주소", "192.168.226.131")
    userName = askstring("user name", "DB에서 생성된 유저")
    password = askstring("password", "password: 1234")
    db = askstring("DB name", "사용할 DB")
    con = pymysql.connect(host=ip, user=userName, password=password, db=db, charset='utf8')
    cur = con.cursor()
    headerList = []
    tableName = os.path.basename(filename)
    tableName = tableName[:tableName.find(".")]
    for header in csvList[0]:
        headerList.append(header.replace(" ", ""))
    sql = "CREATE TABLE "
    sql += tableName + "("
    for header in headerList:
        sql += header + " CHAR(20), "
    sql = sql[:-2]
    sql += ");"
    cur.execute(sql)
    for i in range(1, len(csvList)):
        rowList = csvList[i]
        sql = "INSERT INTO " + tableName + " VALUES("
        for row in rowList:
            sql += "'" + row + "', "
        sql = sql[:-2]
        sql += ");"
        cur.execute(sql)
    con.commit()
    cur.close()
    con.close()
    subWindow = Toplevel(window)
    pLabel = Label(subWindow, text='Save OK')
    def choice():
        subWindow.destroy()
    button = Button(subWindow, text='확인', command=choice)
    pLabel.pack()
    button.pack()
    
def learnLinearR():
    global csvList, filename
    '''
    linear R 학습 & 결과
    '''
    filename = askopenfilename(parent=window,
      filetypes=(("데이터 파일", "*.xls;*.xlsx;*.csv;*.json;*.txt"), ("모든파일", "*.*")))
    dataframe = pd.read_csv(filename, header=False, sep=",")
    brt = askfloat('learning_rate', 'learning_rate', minvalue=0, maxvalue=2)
    return
    
def predictLinearR():
    global csvList, filename
    '''
    학습한 LR 예측
    '''
    return

def learnBinaryL():
    global csvList, filename
    '''
    이항분류 로지스틱 학습
    '''
    filename = askopenfilename(parent=window,
      filetypes=(("데이터 파일", "*.xls;*.xlsx;*.csv;*.json;*.jpg;*.raw;*.png;*.txt"), ("모든파일", "*.*")))
    return

def predictBinaryL():
    global csvList, filename
    '''
    이항분류 로지스틱 예측
    '''
    return

def learnMultiL():
    global csvList, filename
    '''
    다항분류 로지스틱 학습
    '''
    filename = askopenfilename(parent=window,
      filetypes=(("데이터 파일", "*.xls;*.xlsx;*.csv;*.json;*.jpg;*.raw;*.png;*.txt"), ("모든파일", "*.*")))
    return

def predictMultiL():
    global csvList, filename
    '''
    다항분류 로지스틱 예측
    '''
    return

def knn():
    global csvList, filename
    '''
    K-NN 분류
    '''
    filename = askopenfilename(parent=window,
      filetypes=(("데이터 파일", "*.xls;*.xlsx;*.csv;*.json;*.jpg;*.raw;*.png;*.txt"), ("모든파일", "*.*")))
    return

def learnSVM():
    global csvList, filename
    '''
    서포트벡터머신 학습
    '''
    filename = askopenfilename(parent=window,
      filetypes=(("데이터 파일", "*.xls;*.xlsx;*.csv;*.json;*.jpg;*.raw;*.png;*.txt"), ("모든파일", "*.*")))
    return

def predictSVM():
    global csvList, filename
    '''
    SVM 예측
    '''
    return
    
## 변수선언 init
window, canvas, paper, filename = [None] * 4
inImage, outImage = [], []
inW, inH, outW, outH = [0] * 4
photo, paper_copy, pLabel = None, None, None
panYN, panYN_gif= False, False
sx, sy, ex, ey = [0] * 4
gif = False
VIEW_X, VIEW_Y = 128, 128
cellList,csvList = [], []

## main
if __name__ == "__main__":
    window = Tk()
    window.geometry('500x500')
    window.title('cutz Analyzer v1.0')
    window.bind("<ButtonRelease-1>", mouseDrop)
    window.bind("<Button-1>", mouseClick)
    status = Label(window, text='이미지 정보: ', bd=1, relief=SUNKEN, anchor=W)
    status.pack(side=BOTTOM, fill=X)
    
    mainMenu = Menu(window)
    window.config(menu=mainMenu)
    fileMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='파일', menu=fileMenu)
    fileMenu.add_command(label='열기', command=openFile)
    fileMenu.add_command(label='저장', command=saveFile)
    fileMenu.add_separator()
    fileMenu.add_command(label='CSV열기', command=openCSV)
    fileMenu.add_command(label='CSV저장', command=saveCSV)
    fileMenu.add_separator()
    fileMenu.add_command(label='JSON열기', command=openJSON)
    fileMenu.add_command(label='JSON저장', command=saveJSON)
    fileMenu.add_separator()
    fileMenu.add_command(label='Excel열기', command=openExcel)
    fileMenu.add_command(label='Excel저장', command=saveExcel)
    fileMenu.add_separator()
    fileMenu.add_command(label='종료', command=exitFile)
    
    pixelMenu = Menu(mainMenu)
    mainMenu.add_cascade(labe='화소점처리', menu=pixelMenu)
    pixelMenu.add_command(label='밝게하기', command=lambda : addImage(1))
    pixelMenu.add_command(label="어둡게하기", command=lambda : addImage(2))
    pixelMenu.add_command(label='밝게하기(곱연산)', command=lambda: addImage(3))
    pixelMenu.add_command(label="어둡게하기(나눗셈)", command=lambda: addImage(4))
    pixelMenu.add_command(label='AND연산', command=lambda: addImage(5))
    pixelMenu.add_command(label="OR연산", command=lambda: addImage(6))
    pixelMenu.add_command(label='XOR연산', command=lambda: addImage(7))
    pixelMenu.add_command(label='반전', command=lambda: addImage(8))
    pixelMenu.add_command(label='감마', command=lambda: addImage(9))
    pixelMenu.add_command(label='파라볼라(Cap)', command=lambda: addImage(10))
    pixelMenu.add_command(label='파라볼라(Cup)', command=lambda: addImage(11))
    pixelMenu.add_command(label='이진화', command=lambda: addImage(12))
    pixelMenu.add_command(label='범위강조', command=lambda: addImage(13))
    
    geoMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='기하학처리', menu=geoMenu)
    geoMenu.add_command(label='상하반전', command=lambda: direct(1))
    geoMenu.add_command(label='좌우반전', command=lambda: direct(2))
    geoMenu.add_command(label='화면이동', command=panImage)
    geoMenu.add_command(label='줌아웃', command=zoomOut)
    geoMenu.add_command(label='줌인(forward)', command=lambda: zoomIn(1))
    geoMenu.add_command(label='줌인(backward)', command=lambda: zoomIn(2))
    geoMenu.add_command(label='회전(rotate)', command=rotate)
    
    areaMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='화소영역처리', menu=areaMenu)
    areaMenu.add_command(label='엠보싱', command=embossing)
    areaMenu.add_command(label='블러링', command=lambda: blurring(0))
    areaMenu.add_command(label='가우시안블러링', command=gausian_blurring)
    areaMenu.add_command(label='샤프닝', command=lambda: sharpening(1))
    areaMenu.add_command(label='고주파샤프닝', command=lambda: sharpening(2))
    areaMenu.add_command(label='언샤프마스크', command=lambda: blurring(1))
    
    analyzeMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='데이터분석', menu=analyzeMenu)
    analyzeMenu.add_command(label='평균값', command=lambda: a_average(1))
    analyzeMenu.add_command(label='절사평균', command=lambda: a_average(2))    
    analyzeMenu.add_command(label='최댓값&최솟값', command=a_minmax)
    analyzeMenu.add_command(label='히스토그램', command=a_histogram_plt)
    analyzeMenu.add_command(label='스트레칭', command=stretch)
    analyzeMenu.add_command(label='엔드인', command=endin)
    analyzeMenu.add_command(label='평활화', command=equalize)
    
    compareMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='비교', menu=compareMenu)
    compareMenu.add_command(label='원본사진비교', command=display_copy)
    compareMenu.add_command(label='원본되돌리기', command=rollback)
    compareMenu.add_command(label='출력비율변경', command=display_geniune)
    compareMenu.add_command(label='사진합성하기', command=morphing)
    
    otherMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='다른 포맷 처리', menu=otherMenu)
    otherMenu.add_command(label='CSV 내보내기', command=saveRGBCSV)
    otherMenu.add_command(label='CSV 불러오기', command=openCSV)
    otherMenu.add_separator()
    otherMenu.add_command(label='SQLite 내보내기', command=saveSQLite)
    otherMenu.add_command(label='SQLite 목록 불러오기', command=loadSQLite)
    otherMenu.add_separator()
    otherMenu.add_command(label='mySQL 내보내기', command=savemySql)
    otherMenu.add_command(label='mySQL 목록 불러오기', command=loadmySql)
    otherMenu.add_separator()
    otherMenu.add_command(label='Excel 내보내기(화소)', command=sqlExcel1)
    otherMenu.add_command(label='Excel 불러오기(화소)', command=sqlExcel3)
    otherMenu.add_command(label='Excel 내보내기(음영)', command=sqlExcel2)
    
    bigDataMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='빅데이터 처리', menu=bigDataMenu)
    bigDataMenu.add_command(label='SQLite 대용량 보내기', command=bigData01)
    bigDataMenu.add_command(label='mySQL 대용량 보내기', command=bigData02)
    
    mlMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='머신러닝 처리', menu=mlMenu)
    mlMenu.add_command(label='학습: Linear Regression', command=learnLinearR)
    mlMenu.add_command(label='예측: Linear Regression', command=predictLinearR)
    mlMenu.add_separator()
    mlMenu.add_command(label='학습: Binary Logistic', command=learnBinaryL)
    mlMenu.add_command(label='예측: Binary Logistic', command=predictBinaryL)
    mlMenu.add_separator()
    mlMenu.add_command(label='학습: Multi Logistic', command=learnMultiL)
    mlMenu.add_command(label='예측: Multi Logistic', command=predictMultiL)
    mlMenu.add_separator()
    mlMenu.add_command(label='K-NN', command=knn)
    mlMenu.add_separator()
    mlMenu.add_command(label='학습: SVM', command=learnSVM)
    mlMenu.add_command(label='예측: SVM', command=predictSVM)
    
    excelMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='Excel DB처리', menu=excelMenu)
    excelMenu.add_command(label='SQLite 엑셀 읽기', command=sqliteData01)
    excelMenu.add_command(label='SQLite 엑셀 쓰기', command=sqliteData02)
    excelMenu.add_command(label='mySql 엑셀 읽기', command=mySqlData01)
    excelMenu.add_command(label='mySql 엑셀 쓰기', command=mySqlData02)

    
    window.mainloop()