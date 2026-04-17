import cv2

def detect_writers(image):

    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray,0,255,
            cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(25,5))
    dilated = cv2.dilate(thresh,kernel,iterations=2)

    contours,_ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    regions=[]

    for c in contours:

        x,y,w,h = cv2.boundingRect(c)

        if w>150 and h>60:

            roi=image[y:y+h,x:x+w]

            regions.append((roi,(x,y,w,h)))

    return regions