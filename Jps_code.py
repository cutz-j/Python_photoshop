from tkinter import *
import os.path
import math
from tkinter.filedialog import *
from tkinter.simpledialog import *
import operator
import numpy as np
import struct
import threading

## 함수 선언
def loadImage(fname):
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    fsize = os.path.getsize(fname)
    inH = inW = int(math.sqrt(fsize))
    inImage = np.zeros([inH, inW], dtype=np.int32)
    fp = open(fname, 'rb')
    for  i  in range(inH) :
        for  k  in  range(inW) :
            inImage[i][k] =  int(ord(fp.read(1)))
    fp.close()
    
def openFile():
    global window, canvas, paper, filename,inImage, outImage,inW, inH, outW, outH, photo, gif
    filename = askopenfilename(parent=window, filetypes=(("그림파일", "*.raw; *.gif"), ("모든파일", "*.*")))
    if filename[-3:] == "gif":
        gif = True
        loadImage_gif(filename)
        equal_gif()
        return
    else: gif = False
    loadImage(filename) # 파일 -> 입력메모리
    equal() # 입력메모리 -> 출력메모리

def display():
    global window, canvas, PLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if gif == True:
        display_gif()
        return
    if  canvas != None:
        canvas.destroy()
    if pLabel != None:
        pLabel.destroy()
    window.geometry(str(outH) + 'x' + str(outW))
    canvas = Canvas(window, width=outW, height=outH)
    paper = PhotoImage(width=outW, height=outH)
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH) :
            for k in range(0, outW) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data, data, data), (k,i))
    threading.Thread(target=putPixel).start()
    canvas.pack()

    
def display_first():
    if gif == True:
        display_first_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if  canvas != None :
        canvas.destroy()
    window.geometry(str(outH) + 'x' + str(outW))
    canvas = Canvas(window, width=outW, height=outH)
    paper = PhotoImage(width=outW, height=outH)
    paper_copy = paper.copy()
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')           
    def putPixel() :
        for i in range(0, outH) :
            for k in range(0, outW) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data, data, data), (k,i))
                paper_copy.put('#%02x%02x%02x' % (data, data, data), (k,i))
    threading.Thread(target=putPixel).start()        
    canvas.pack()

    
def display_copy():
    if gif == True:
        display_copy_gif()
        return
    global window, canvas, pLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if  canvas != None :
        canvas.destroy()
    window.geometry(str(outH*2) + 'x' + str(outW))
    canvas = Canvas(window, width=outW, height=outH)
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')
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
    if gif == True:
        saveFile_gif()
        return
    global window, canvas, paper, filename,inImage, outImage,inW, inH, outW, outH
    saveFp = asksaveasfile(parent=window, mode='wb',
                               defaultextension="*.raw", filetypes=(("RAW파일", "*.raw"), ("모든파일", "*.*")))
    for i in range(outW):
        for k in range(outH):
            saveFp.write( struct.pack('B',outImage[i][k]))
    saveFp.close()

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
    
def merge():
    if gif:
        merge_gif()
        return
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW, outH = inW, inH
    oldImage = inImage[:]
    new_file = askopenfilename(parent=window, filetypes=(("그림파일", "*.raw"), ("모든파일", "*.*")))
    loadImage(new_file)
    newImage = inImage[:]
    inImage = (np.array(oldImage) + np.array(newImage)) / 2
    outImage = np.array(inImage[:], dtype=np.int32)
    display()
    
''' ######################### GIF 처리 공간 ######################### '''
def loadImage_gif(fname) :
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    global inImageR, inImageG, inImageB, outImageR, outImageG, outImgageB
    photo = PhotoImage(file=filename)
    inW = photo.width()
    inH = photo.height()
    inImage = []
    tmpList = []
    for i in range(inH):
        tmpList = []
        for k in range(inW) :
            tmpList.append(np.array([0, 0, 0]))
        inImage.append(tmpList)
    for  i  in range(inH):
        for  k  in  range(inW):
            r, g, b = photo.get(k, i)
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
    if  canvas != None :
        canvas.destroy()
    if pLabel != None:
        pLabel.destroy()    
    window.geometry(str(outH) + 'x' + str(outW))
    canvas = Canvas(window, width=outW, height=outH)
    paper = PhotoImage(width=outW, height=outH)
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH) :
            for k in range(0, outW) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data[0], data[1], data[2]), (k,i))
    threading.Thread(target=putPixel).start()
    canvas.pack()

def display_first_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if  canvas != None :
        canvas.destroy()
    window.geometry(str(outH) + 'x' + str(outW))
    canvas = Canvas(window, width=outW, height=outH)
    paper = PhotoImage(width=outW, height=outH)
    paper_copy = paper.copy()
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')
    def putPixel() :
        for i in range(0, outH) :
            for k in range(0, outW) :
                data = outImage[i][k]
                paper.put('#%02x%02x%02x' % (data[0], data[1], data[2]), (k,i))
                paper_copy.put('#%02x%02x%02x' % (data[0], data[1], data[2]), (k,i))
    threading.Thread(target=putPixel).start()
    canvas.pack()
        
def display_copy_gif():
    global window, canvas, pLabel, paper, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
    if  canvas != None :
        canvas.destroy()
    window.geometry(str(outH*2) + 'x' + str(outW))
    canvas = Canvas(window, width=outW, height=outH)
    canvas.create_image((outW/2, outH/2), image=paper, state='normal')
    canvas.pack(side=RIGHT)
    photo = PhotoImage()
    pLabel = Label(window, image=photo)
    pLabel.pack(side=LEFT)
    pLabel.configure(image=paper_copy)
    
def rollback_gif():
    global window, canvas, paper, PLabel, filename, inImage, outImage, inW, inH, outW, outH, photo, paper_copy
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
    outW, outH = inW, outH
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
    outW, outH = inW, outH
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
    outW, outH = inW, outH
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
    outW, outH = inW, outH
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
    
def merge_gif():
    global window, canvas, paper, filename, inImage, outImage, inW, inH, outW, outH, photo
    outW, outH = inW, inH
    oldImage = np.copy(inImage)
    new_file = askopenfilename(parent=window, filetypes=(("그림파일", "*.gif"), ("모든파일", "*.*")))
    photo = PhotoImage(file=new_file)
    inW = photo.width()
    inH = photo.height()
    newImage = []
    tmpList = []
    for i in range(inH):
        tmpList = []
        for k in range(inW) :
            tmpList.append(np.array([0, 0, 0]))
        newImage.append(tmpList)
    for  i  in range(inH):
        for  k  in  range(inW):
            r, g, b = photo.get(k, i)
            newImage[i][k] = [r, g, b]
    newImage = np.array(newImage)
    photo = None
    inImage = (np.array(oldImage) + np.array(newImage)) / 2
    outImage = np.array(inImage[:], dtype=np.int32)
    display_gif()
    
## 변수선언 init
window, canvas, paper, filename = [None] * 4
inImage, outImage = [], []
inW, inH, outW, outH = [0] * 4
photo, paper_copy, pLabel = None, None, None
panYN, panYN_gif= False, False
sx, sy, ex, ey = [0] * 4
gif = False

## main
if __name__ == "__main__":
    window = Tk()
    window.geometry('500x500')
    window.title('J Photo 1.0')
    window.bind("<ButtonRelease-1>", mouseDrop)
    window.bind("<Button-1>", mouseClick)
    
    mainMenu = Menu(window)
    window.config(menu=mainMenu)
    fileMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='파일', menu=fileMenu)
    fileMenu.add_command(label='열기', command=openFile)
    fileMenu.add_command(label='저장', command=saveFile)
    fileMenu.add_separator()
    fileMenu.add_command(label='종료', command=exitFile)
    
    pixelMenu = Menu(mainMenu)
    mainMenu.add_cascade(labe='화소점처리', menu=pixelMenu)
    pixelMenu.add_command(label='동일영상', command=equal)
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
    
    compareMenu = Menu(mainMenu)
    mainMenu.add_cascade(label='비교', menu=compareMenu)
    compareMenu.add_command(label='원본사진비교', command=display_copy)
    compareMenu.add_command(label='원본되돌리기', command=rollback)
    compareMenu.add_command(label='사진합성하기', command=merge)
    
    window.mainloop()