import os
import numpy as np
import cv2
import tkinter.filedialog as fd
arrowlist = ["300","400","450","500"]
txt = []
C_area = 0
numberH = 0
A = 0
L = 0
C = 0
arrow = 0
letter = 0
id_of_image = []
circlexist = 1
arrowexist = 1
lineexist = 1
line_frame = fd.askdirectory()
f = open(line_frame+"/"+"line_frame.txt","r")
for i in f:
    id_of_image.append(i.strip("\n"))
Directory = fd.askdirectory()
image = os.listdir(Directory)
for name in image:
    img = cv2.imread(Directory+"/"+name)
    liste = name.split(".")        
    imgcopy = np.copy(img)
    graycopy = cv2.cvtColor(imgcopy,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(graycopy,60,255,cv2.THRESH_BINARY_INV)
    thresh = cv2.dilate(thresh,(11,11),iterations=3)
    thresh=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,(5,5))
    cnts, ret = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    maxcnt = None
    maxarea = 0
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 1600:
            continue
        if area > maxarea:
            maxcnt = cnt
            maxarea = area
    if liste[0] in id_of_image:
        center,radius = cv2.minEnclosingCircle(maxcnt)
        if center[0] < img.shape[1]//2:
            txt.append(liste[0]+"_"+"0")
        else:
            txt.append(liste[0]+"_"+"1")
        continue
    try:
        approx = cv2.approxPolyDP(maxcnt,0.02*cv2.arcLength(maxcnt,True),True)
        if len(approx) == 8:
            C +=1
            if C %10 == 0:
                if circlexist ==1 and C_area < cv2.contourArea(maxcnt):
                    (xdir,ydir),radius = cv2.minEnclosingCircle(maxcnt)
                    [x,y,w,h] = cv2.boundingRect(maxcnt)
                    C_area =  cv2.contourArea(maxcnt)//2
                    ofset = (C_area**(1/2))//2
                    # test
                    #cv2.imshow("win_2",img[int(y+ofset):int(y+h-ofset),int(x+ofset):int(x+w-ofset)])
                    #cv2.waitKey(0)
                    letterimg = img[int(y+ofset):int(y+h-ofset),int(x+ofset):int(x+w-ofset)]
                    lettergray = cv2.cvtColor(letterimg,cv2.COLOR_BGR2GRAY)
                    ret,thresh = cv2.threshold(lettergray,60,255,cv2.THRESH_BINARY_INV)
                    epsilon = 0.024*cv2.arcLength(cnt,True)
                    approx = cv2.approxPolyDP(cnt,epsilon,True)
                    if len(approx) == 8:
                        txt.append(liste[0] + "_" + str(xdir)+"_"+str(ydir) + "_" + "T")
                    elif len(approx) == 12 and  numberH == 0:
                        txt.append(liste[0] + "_" + str(xdir)+"_"+str(ydir) + "_" + "H")
                        numberH +=1
                    elif len(approx) == 12:
                        txt.append(liste[0] + "_" + str(xdir)+"_"+str(ydir) + "_" + "X")
                    elif len(approx) == 6:
                        txt.append(liste[0] + "_" + str(xdir)+"_"+str(ydir) + "_" + "L")
                    lineexist = 1
                    circlexist = 0
                    letter +=1
                    arrowexist = 1
        elif len(approx) == 7:
            A +=1
            if A%15 == 0:
                if arrowexist ==1:
                    arrow +=1
                    if arrow>4:
                        continue
                    center, radius = cv2.minEnclosingCircle(maxcnt)
                    [x,y,w,h] = cv2.boundingRect(maxcnt)
                    rect = cv2.minAreaRect(cnt)
                    txt.append(liste[0]+"_"+str(rect[2])+"_"+str(arrowlist[arrow-1]))
                    circlexist = 1
                    arrowexist = 0
        elif len(approx) == 6 and lineexist ==1:
            L += 1
            if L%30 ==0:
                if lineexist ==1:
                    lineexist = 0
                    circlexist = 1
                    arrowexist = 1
        cv2.imshow("win",img)
    except:
        arrowexist = 1
    if cv2.waitKey(2) == ord("q"):
        break

f = open("output.txt","a")
for i in txt:
    f.write(i)
    f.write("\n")


