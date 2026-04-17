import cv2

def highlight_stress(image):

    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    edges=cv2.Canny(gray,50,150)

    contours,_=cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for c in contours:

        area=cv2.contourArea(c)

        if area>300:

            x,y,w,h=cv2.boundingRect(c)

            cv2.rectangle(
                image,
                (x,y),
                (x+w,y+h),
                (0,0,255),
                2
            )

    return image