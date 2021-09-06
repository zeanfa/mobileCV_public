import numpy as np
import cv2

print('Press 4 to Quit the Application\n')

#Open Default Camera
cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    #Take each Frame
    ret, frame = cap.read()
    
    #Flip Video vertically (180 Degrees)
    frame = cv2.flip(frame, 180)

    invert = ~frame

    # Show video
    cv2.imshow('Cam', frame)
    cv2.imshow('Inverted', invert)

    # Exit if "4" is pressed
    k = cv2.waitKey(1) & 0xFF
    if k == 52 : #ord 4
        #Quit
        print ('Good Bye!')
        break

#Release the Cap and Video   
cap.release()
cv2.destroyAllWindows()
