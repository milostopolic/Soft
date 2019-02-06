import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import math
import keras

from recognizednumber import RecognizedNumber

def scale_to_range(image): 

    return image/255

def prepare_for_ann(img_bin, recognizedNumber):
    
    extraPixels = 7
    contour = img_bin[recognizedNumber.get_top_left()[1] - extraPixels : recognizedNumber.get_bottom_left()[1] + extraPixels, recognizedNumber.get_top_left()[0] - extraPixels : recognizedNumber.get_top_right()[0] + extraPixels]
    
    resizedContour = cv2.resize(contour, (28, 28), interpolation = cv2.INTER_NEAREST)    
    mVector = scale_to_range(resizedContour).flatten()
    mColumn = np.reshape(mVector, (1, 784))
    #return mColumn
    return np.array(mColumn, dtype=np.float32)

def crossing_lines(img_bin, recognizedNumbers, longestBlue, longestGreen):

    x1b, y1b, x2b, y2b = longestBlue[0]
    x1g, y1g, x2g, y2g = longestGreen[0]

    #print(x1b, y1b, x2b, y2b)

    for recognizedNumber in recognizedNumbers:
        if(recognizedNumber.get_considered()):
            dist_from_blue = np.linalg.norm(np.cross(np.array([x2b,y2b])-np.array([x1b,y1b]), np.array([x1b,y1b])-np.array(recognizedNumber.get_bottom_right())))/np.linalg.norm(np.array([x2b,y2b])-np.array([x1b,y1b]))
            #dist_from_blue = np.linalg.norm(np.cross(np.array([x2b,y2b])-np.array([x1b,y1b]), np.array([x1b,y1b])-np.array(recognizedNumber.get_center())))/np.linalg.norm(np.array([x2b,y2b])-np.array([x1b,y1b]))
            # if(dist_from_blue <= 5 and not recognizedNumber.get_bluePassed()):
            #     recognizedNumber.set_bluePassed(True)

            dist_from_green = np.linalg.norm(np.cross(np.array([x2g,y2g])-np.array([x1g,y1g]), np.array([x1g,y1g])-np.array(recognizedNumber.get_bottom_right())))/np.linalg.norm(np.array([x2g,y2g])-np.array([x1g,y1g]))
            #dist_from_green = np.linalg.norm(np.cross(np.array([x2g,y2g])-np.array([x1g,y1g]), np.array([x1g,y1g])-np.array(recognizedNumber.get_center())))/np.linalg.norm(np.array([x2g,y2g])-np.array([x1g,y1g]))        
            # if(dist_from_green <= 5 and not recognizedNumber.get_greenPassed()):
            #     recognizedNumber.set_greenPassed(True)

            if((x1b<=recognizedNumber.get_bottom_right()[0] and recognizedNumber.get_bottom_right()[0]<=x2b and y2b<=recognizedNumber.get_bottom_right()[1] and recognizedNumber.get_bottom_right()[1]<=y1b) and dist_from_blue <= 8 and not recognizedNumber.get_bluePassed()):
                recognizedNumber.set_bluePassed(True)
                
                # imgNN = prepare_for_ann(img_bin, recognizedNumber)             
                # predicted = neuralN.predict(imgNN)

                global result
                result += recognizedNumber.get_prediction()
                print('+' + str(recognizedNumber.get_prediction()))

            if((x1g<=recognizedNumber.get_bottom_right()[0] and recognizedNumber.get_bottom_right()[0]<=x2g and y2g<=recognizedNumber.get_bottom_right()[1] and recognizedNumber.get_bottom_right()[1]<=y1g) and dist_from_green <= 8 and not recognizedNumber.get_greenPassed()):
                recognizedNumber.set_greenPassed(True)
                
                # imgNN = prepare_for_ann(img_bin, recognizedNumber)            
                # predicted = neuralN.predict(imgNN)

                #global result
                result -= recognizedNumber.get_prediction()
                print('-' + str(recognizedNumber.get_prediction()))

def removePastNumbers(number):

    return not (number.get_bottom_right()[0] > 625 or number.get_bottom_right()[1] > 465)

def alreadyRecognized(number):
    
    closeNumbers = []

    for recognizedNumber in recognizedNumbers:
        #dist = distance.euclidean(number.get_bottom_right(), recognizedNumber.get_bottom_right())
        distance = math.sqrt( (recognizedNumber.get_bottom_right()[0] - number.get_bottom_right()[0])**2 + (recognizedNumber.get_bottom_right()[1] - number.get_bottom_right()[1])**2 )
        if distance < 20:
            closeNumbers.append([distance, recognizedNumber])

    closeNumbers = sorted(closeNumbers, key=lambda num: num[0])   

    if len(closeNumbers) > 0:
        return closeNumbers[0][1]
    else:
        return None


# def draw_recognizedNumbers(frame, recognizedNumbers):
    
#     for number in recognizedNumbers:
#         cv2.rectangle(frame, number.get_top_left(), number.get_bottom_right(), (255, 255, 255), 1)
        
def search_for_numbers(frame, img_bin, recognizedNumbers):
    
    contours = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

    for contour in contours:
        coords = cv2.boundingRect(contour)        
        
        if coords[3] > 10:
            number = RecognizedNumber(coords)
            #recognizedNumbers.append(number)
            sameNumber = alreadyRecognized(number)
            if sameNumber is None:
                
                recognizedNumbers.append(number)
                # imgNN = prepare_for_ann(img_bin, number)             
                # number.set_prediction(np.argmax(neuralN.predict(imgNN)))      
            else:
                sameNumber.update_coordinates(coords)

    # for recognizedNumber in recognizedNumbers:
    #     if(recognizedNumber.get_predicted()==False and recognizedNumber.get_top_left()[0] > 30 and recognizedNumber.get_top_left()[1] > 30):            
    #         imgNN = prepare_for_ann(img_bin, recognizedNumber)             
    #         recognizedNumber.set_prediction(np.argmax(neuralN.predict(imgNN)))
    #         recognizedNumber.set_predicted = True

    recognizedNumbers = list(filter(removePastNumbers, recognizedNumbers))

    for number in recognizedNumbers:
        if(number.get_predicted()==False and number.get_top_left()[0] > 7 and number.get_top_left()[1] > 7):            
            imgNN = prepare_for_ann(img_bin, number)             
            number.set_prediction(np.argmax(neuralN.predict(imgNN)))
            number.set_predicted = True


        for number2 in recognizedNumbers:
            if(overlap(number,number2)):
                number.set_prediction(number.get_prediction() + number2.get_prediction())
                number2.set_considered(False)



        if(number.get_bluePassed() and not number.get_greenPassed()):
            cv2.rectangle(frame, number.get_top_left(), number.get_bottom_right(), (255, 0, 0), 1)
        elif(not number.get_bluePassed() and number.get_greenPassed()):
            cv2.rectangle(frame, number.get_top_left(), number.get_bottom_right(), (0, 255, 0), 1)
        elif(number.get_bluePassed() and number.get_greenPassed()):
            cv2.rectangle(frame, number.get_top_left(), number.get_bottom_right(), (0, 0, 255), 1)
        else:
            cv2.rectangle(frame, number.get_top_left(), number.get_bottom_right(), (255, 255, 255), 1)

def overlap(r1,r2):
    
    hoverlaps = True
    voverlaps = True
    if (r1.get_x() > r2.get_top_right()[0]) or (r1.get_top_right()[0] < r2.get_x()):
        hoverlaps = False
    if (r1.get_y() < r2.get_bottom_left()[1]) or (r1.get_bottom_left()[1] > r2.get_y()):
        voverlaps = False
    return hoverlaps and voverlaps

# def select_roi(image_orig, image_bin):
#     '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
#         Za svaki region napraviti posebnu sliku dimenzija 28 x 28. 
#         Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
#         Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
#         i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
#     '''
#     img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#     sorted_regions = [] # lista sortiranih regiona po x osi (sa leva na desno)
#     regions_array = []
#     for contour in contours: 
#         x,y,w,h = cv2.boundingRect(contour) #koordinate i velicina granicnog pravougaonika
#         area = cv2.contourArea(contour)
#         if area > 100 and h < 100 and h > 15 and w > 20:
#             # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
#             # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
#             region = image_bin[y:y+h+1,x:x+w+1]
#             regions_array.append([resize_region(region), (x,y,w,h)])       
#             cv2.rectangle(image_orig,(x,y),(x+w,y+h),(0,255,0),2)
#     regions_array = sorted(regions_array, key=lambda item: item[1][0])
#     sorted_regions = sorted_regions = [region[0] for region in regions_array]
    
#     # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
#     return image_orig, sorted_regions

def image_bin(image_gs):
    
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    ret,image_bin = cv2.threshold(image_gs, 160, 255, cv2.THRESH_BINARY)
    return image_bin

def getLongestLine(lines):

    maxLength = 0
    longestLine = lines[0]
    for line in lines:
        x1, y1, x2, y2 = line[0]        
        distance = math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 )
        if(distance > maxLength):
            maxLength = distance
            longestLine = line
    return longestLine


def houghProb(frame): 

    edges = cv2.Canny(frame,50,150)
    blurred = cv2.GaussianBlur(edges,(7,7),1)
    minLineLength = 100
    maxLineGap = 50
    lines = cv2.HoughLinesP(blurred,1,np.pi/180,100,minLineLength,maxLineGap)
    #for x1,y1,x2,y2 in lines[0]:
    #    cv2.line(frame,(x1,y1),(x2,y2),(0,0,255,),2)
    #cv2.imshow('frameX', frame)
    return lines

def findLines(frame):
    b,g,r = cv2.split(frame)
    b_blur = cv2.GaussianBlur(b,(5,5),1)
    b_hough = houghProb(b_blur)
    g_hough = houghProb(g)
    return getLongestLine(b_hough), getLongestLine(g_hough)


neuralN = keras.models.load_model("keras_mnist1.h5")
result = 0
recognizedNumbers = []
cap = cv2.VideoCapture("videos/video-9.avi")
ret, frame = cap.read()

longestBlue, longestGreen = findLines(frame)
# b,g,r = cv2.split(frame)
# b_blur = cv2.GaussianBlur(b,(5,5),1)
# b_hough = houghProb(b_blur)
# g_hough = houghProb(g)
# longestBlue = getLongestLine(bLines)
# longestGreen = getLongestLine(gLines)


while(cap.isOpened()):
   ret, frame = cap.read()
   if not ret:
    print('Final result: ' + str(result))
    break
   #time.sleep(1/10)
#    gs = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#    img_bin = image_bin(gs)
   #frame,x = select_roi(frame, img_bin)

        

   mask = cv2.inRange(frame, np.array([160, 160, 160], dtype="uint8"), np.array([255, 255, 255], dtype="uint8"))
   whiteImage = cv2.bitwise_and(frame, frame, mask = mask)
   whiteImage = cv2.cvtColor(whiteImage, cv2.COLOR_BGR2GRAY)
   img_bin = cv2.threshold(whiteImage, 1, 255, cv2.THRESH_BINARY)[1]

   search_for_numbers(frame, img_bin, recognizedNumbers)
   #recognizedNumbers = list(filter(removePastNumbers, recognizedNumbers))
   #draw_recognizedNumbers(frame, recognizedNumbers)

   crossing_lines(img_bin, recognizedNumbers, longestBlue, longestGreen)
   #print(result)
   for x1,y1,x2,y2 in longestBlue:
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255,),2)
   for x1,y1,x2,y2 in longestGreen:
        cv2.line(frame,(x1,y1),(x2,y2),(0,0,255,),2)
   cv2.imshow('frame',frame)
   if cv2.waitKey(1) & 0xFF == ord('q') or ret==False :
       cap.release()
       cv2.destroyAllWindows()
       break
   