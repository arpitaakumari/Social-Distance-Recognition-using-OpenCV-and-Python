#importing the libraries
import cv2
import numpy as np
from math import sqrt,pow
import time
#set the angle factor between 0 and 1. As the camera becomes vertical shift to 1.
calib_angle = 0.5

#Calibarating the camera angle 

#to claculate the distance between the 2 points
def dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5

#calibarating the horizontal direction of the camera
def T2S(T):
    S = abs(T/((1+T**2)**0.5))
    return S

#calibarating the vertical direction of the camera
def T2C(T):
    C = abs(1/((1+T**2)**0.5))
    return C

#to check the proximity of the peron with another person.
def isclose(p1,p2):
    c_d = dist(p1[2], p2[2])
    if(p1[1]<p2[1]):
        a_w = p1[0]
        a_h = p1[1]
    else:
        a_w = p2[0]
        a_h = p2[1]
    T = 0
    try:
        T=(p2[2][1]-p1[2][1])/(p2[2][0]-p1[2][0])
    except ZeroDivisionError:
        T = 1.633123935319537e+16
    S = T2S(T)
    C = T2C(T)
    d_hor = C*c_d
    d_ver = S*c_d
    vc_calib_hor = a_w*1.3
    vc_calib_ver = a_h*0.4*calib_angle
    c_calib_hor = a_w *1.7
    c_calib_ver = a_h*0.2*calib_angle
    if (0<d_hor<vc_calib_hor and 0<d_ver<vc_calib_ver):    #red alert
        return 1
    elif 0<d_hor<c_calib_hor and 0<d_ver<c_calib_ver:      #medium alert
        return 2
    else:                                                  #safe
        return 0


#load YOLO
print(" LOADING THE MODEL ")
net = cv2.dnn.readNet("C:/Users/ARPITA KUMARI/objectdetection/yolov3.weights","C:/Users/ARPITA KUMARI/objectdetection/yolov3.cfg")
with open("C:/Users/ARPITA KUMARI/objectdetection/model_data/coco_classes.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

#loading video
print(" LOADING THE VIDEO FILE ")
cap=cv2.VideoCapture("C:/Users/ARPITA KUMARI/objectdetection/social_distance_video_1.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX
starting_time= time.time()
frame_id = 0
writer=None
(W,H)=(None,None)

while True:
    grabbed,frame= cap.read() # 
    frame_id+=1
    if not grabbed:
        break;
        
    if W is None or H is None:
        (H, W) = frame.shape[:2]
        FW=W
        if(W<1075):
            FW = 1075
        FR = np.zeros((H+210,FW,3), np.uint8)

        col = (255,255,255)
        FH = H + 210
    FR[:] = col
    
    blob = cv2.dnn.blobFromImage(frame,0.00392,(416,416),(0,0,0),True,crop=False) #reduce 416 to 320           
    net.setInput(blob)
    outs = net.forward(outputlayers)
    class_ids=[]
    confidences=[]
    boxes=[]
    location=[]
    for out in outs:
        for detection in out:
            #start = timer()
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if classes[class_id] == 'person':
                if confidence > 0.4:
                #object detected
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x=int(centerX - width/2)
                    y=int(centerY - height/2)
                    boxes.append([x,y,int(width),int(height)]) #put all rectangle areas
                    confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                    class_ids.append(class_id) #name of the object that was detected
    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)
    if len(indexes)>0:
        status = []
        idf = indexes.flatten()
        close_pair = []
        s_close_pair = []
        center = []
        co_info = []
        for i in idf:
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cen = [int(x + w / 2), int(y + h / 2)]
            center.append(cen)
            cv2.circle(frame, tuple(cen),1,(0,0,0),1)
            co_info.append([w, h, cen])
            status.append(0)         
        for i in range(len(center)):
            for j in range(len(center)):
                g = isclose(co_info[i],co_info[j])
                if g == 1:
                    close_pair.append([center[i], center[j]])
                    status[i] = 1
                    status[j] = 1
                elif g == 2:
                    s_close_pair.append([center[i], center[j]])
                    if status[i] != 1:
                        status[i] = 2
                    if status[j] != 1:
                        status[j] = 2
        total_p = len(center)
        low_risk_p = status.count(2)
        high_risk_p = status.count(1)
        safe_p = status.count(0)
        kk = 0
        
        for i in idf:
            #cv2.line(FR,(0,H+1),(FW,H+1),2)
            tot_str = "TOTAL COUNT: " + str(total_p)
            high_str = "HIGH RISK COUNT: " + str(high_risk_p)
            low_str = "MEDIUM RISK COUNT: " + str(low_risk_p)
            safe_str = "SAFE COUNT: " + str(safe_p)

            cv2.putText(frame, tot_str, (10, 25),font, 1, (255, 255, 255), 4)
            cv2.putText(frame, safe_str, (10, 60),font, 1, (0, 255, 0), 4)
            cv2.putText(frame, low_str, (10, 95),font, 1, (0, 120, 255), 4)
            cv2.putText(frame, high_str, (10, 130),font, 1, (0, 0, 255), 4)

            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if status[kk] == 1:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)

            elif status[kk] == 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 120, 255), 2)

            kk += 1
        for h in close_pair:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
        for b in s_close_pair:
            cv2.line(frame, tuple(b[0]), tuple(b[1]), (0, 255, 255), 2)
        cv2.imshow('Social distancing analyser', frame)
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter("output_social.mp4", fourcc, 30,
                                 (frame.shape[1], frame.shape[0]), True)

    writer.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break;

print("Cleaning Up...")
cap.release()
writer.release()
cv2.destroyAllWindows()         