import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
# we are using FaceMeshModule insted of FaceDetectionModule to get more accurate results because it have more nodes
from cvzone.PlotModule import LivePlot
# for ploting live graph
from datetime import datetime

#cap = cv2.VideoCapture("training-2.mp4")
cap = cv2.VideoCapture(0)
#for webcam

detector = FaceMeshDetector(maxFaces=1)
# if we want to track multiple faces then increase maxFaces

plotY = LivePlot(640,360,[20,40],invert=True)

idList = [22,23,24,110,157,158,159,160,161,130,243]
# points near eye which are used to find other points 22,23,24 in face mesh
# 22,23,24, 110,157,158,159,160,161, 130,243 are the points of the left eye
# 159 & 23 are mid points top and bottom

ratioList = []
labels = []
ravg = []

blink = 0
count = 0
time = []
time.append(datetime.now())
d_t = []
m = 0

color = (255,0,255)
# run the video
while True:
    # to make video play countinously without stopping
    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        cap.set(cv2.CAP_PROP_POS_FRAMES,0)

    success,img = cap.read()
    # read image or video
 
    #img, faces = detector.findFaceMesh(img)
    # this will find face and add mesh and return our video with mesh

    img, faces = detector.findFaceMesh(img,draw=False)
    # this will not draw the face mesh but it will add mesh to the image. Used to check selected points clearly

    if len(ravg) == 100:
        break
    # to outline the points we are selecting
    elif faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img,face[id],1,(0,255,0),thickness=-1)

        # distance between points on eye
        # if we just consider distance then we get inaccurate result because if person moves front and back the dist is also varied so we consider the ratio for vertical and horizontal to normalize the dist
        leftUp = face[159]
        leftDown = face[23]
        # 159,23 are points on end of the eye vertically

        leftLeft = face[130]
        leftRight = face[243]
        # 130,243 are points on end of the eye horizontally

        lenghtVer,_ = detector.findDistance(leftUp,leftDown)
        lenghtHor,_ = detector.findDistance(leftLeft,leftRight)
        # ,_ is not useful for us but it is imp to mention because detector.findDistance will also give some extra info

        cv2.line(img,leftUp,leftDown,(0,0,255),3)
        cv2.line(img,leftLeft,leftRight,(0,0,255),3)

        ratio = ((lenghtVer/lenghtHor)*100)
        #float values are more smooth for ploting

        # to make graph more smooth
        ratioList.append(ratio)
        if len(ratioList)>3:
            ratioList.pop(0)
        
        ratioAvg = sum(ratioList)/len(ratioList)

        #print(int(ratioAvg))

        #to count no of blink
        if ratioAvg<=27 and count == 0:
            blink +=1
            #by this we can count blinks but it will also count from multiple frames at once so we have to stop counting for specific frames after we register one count so we add count value to the method
            count = 1
            ravg.append(ratioAvg)
            color = (0,255,0)

            time.append(datetime.now())
            # capturing the time at which the blink happened so we can find duration of blink

            # calculating the time difference b/w recent blink and blink before that so we can come to a conclusion wheather he is drowsy or now
            diff_time = time[blink] - time[blink-1]
            d_t.append(diff_time.total_seconds())
            if diff_time.total_seconds() >= 3:
                labels.append(1)
                print(1)
            else:
                labels.append(0)
                print(0)
                
        
        # to find when eye lid distance comes closer and move far so we can count a blink accurately
        if count != 0:
            count +=1
            if ratio >= 32:
                count = 0
                color=(255,0,255)
        
        cvzone.putTextRect(img,f"blick count = {blink}",(50,100),colorR=color)


        imgPlot = plotY.update(ratioAvg,color)

        img = cv2.resize(img,(640,360))
        #img = cv2.resize(img,(1280,720))

        imgStack = cvzone.stackImages([img,imgPlot],2,1)
        # combining our vide and graph
        #imgStack = cvzone.stackImages([img,imgPlot],1,1) for ploting img on top of each other

    else:
        # if no face found in video then we just present image with not found message
        img = cv2.resize(img,(640,360))
        imgNotVisible = cvzone.putTextRect(img,f"Face not Visible",(100,200),colorR=(0,0,255))
        imgNotVisible = cv2.resize(img,(640,360))
        imgStack = imgNotVisible
    
    cv2.imshow("image",imgStack)
    cv2.waitKey(30)
print(type(d_t[0]))

import csv

list1 = ravg
list2 = d_t
list3 = labels

# Combine lists into tuples (assuming both lists have the same length)
data = zip(list1, list2, list3)

# Define the file name
file_name = 'svc_data4.csv'

# Writing data to CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write column headers if needed
    writer.writerow(['Ratio_Avg', 'diff_time', 'Label'])
    # Write data row by row
    writer.writerows(data)

print(f"Data has been saved to {file_name}")
