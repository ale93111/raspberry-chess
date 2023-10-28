import cv2
import time
import numpy as np
from skimage.transform import resize

from picamera2 import Picamera2

#image_test = cv2.imread('/home/alessandro/Documents/code/test.png')
image_ref  = cv2.imread('./chess_board_reference.png')
height_ref, width_ref= image_ref.shape[:2]
image_ref = (resize(image_ref, (480,640), anti_aliasing=True)*255).astype(np.uint8)

image_ref_gray = cv2.cvtColor(image_ref, cv2.COLOR_BGR2GRAY)

ret, corners_ref = cv2.findChessboardCornersSB(image_ref_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
   

print(image_ref.shape)

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

font = cv2.FONT_HERSHEY_SIMPLEX 
prev_frame_time = 0
new_frame_time = 0

while True:
    new_frame_time = time.time() 
    img = picam2.capture_array()
    image_test = img[:,:,:3]

    image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)

    ret_test, corners_test = cv2.findChessboardCornersSB(image_test_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
    
    if ret_test:
        homography, mask = cv2.findHomography(corners_test, corners_ref, cv2.RANSAC)
        # homography, mask = cv2.findHomography(corners_ref, corners_test, cv2.RANSAC)

        height, width = image_test.shape[:2]
  
        transformed_img = cv2.warpPerspective(image_test, homography, (width, height))

    #transformed_img = transformed_img[:height_ref, :width_ref]

    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = str(int(fps))
    
    cv2.putText(transformed_img, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    
    cv2.imshow('img',transformed_img)
    if cv2.waitKey(1) == ord('q'):
        break
 
cv2.destroyAllWindows()

