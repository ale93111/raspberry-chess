import cv2
import time
import numpy as np
# from skimage.transform import resize
import torch
from PIL import Image

from picamera2 import Picamera2

from myutils import bbox_yolo_to_pascal_voc

perfect_corners = []
for i in range(1,8):
    for j in range(1,8):
        bbox = [60*i, 60*j]
        perfect_corners.append(bbox)
        
corners_ref = np.array(perfect_corners)

shape_ref = [480, 480] 

model_path = 'models/yolo5n_chess_pieces_rg.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)  # local model

picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
picam2.start()

font = cv2.FONT_HERSHEY_SIMPLEX 
prev_frame_time = 0
new_frame_time = 0

# out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1.0, (480,480))

while True:
    new_frame_time = time.time() 
    img = picam2.capture_array()
    image_test = img[:,:,:3]

    image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    image_test_rgb = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)

    ret_test, corners_test = cv2.findChessboardCornersSB(image_test_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
    
    if ret_test:
        homography, mask = cv2.findHomography(corners_test, corners_ref, cv2.RANSAC)
        # homography, mask = cv2.findHomography(corners_ref, corners_test, cv2.RANSAC)

        height, width = image_test.shape[:2]
  
        transformed_img = cv2.warpPerspective(image_test_rgb, homography, (width, height))
        
        transformed_img = transformed_img[:shape_ref[0],:shape_ref[1]]

        yolo_image = Image.fromarray(transformed_img)
        results = model(yolo_image)
        
        output_image = np.array(results.ims[0]).astype(np.uint8)

        yolo_bboxes = results.xywhn[0].numpy()
        predictions_bboxes = [bbox_yolo_to_pascal_voc(yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3], yolo_bbox[5], 480, 480) for yolo_bbox in yolo_bboxes if yolo_bbox[4] > 0.75]

        for bbox in predictions_bboxes:
            if bbox[4] == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
        
            output_image = cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    else:
        output_image = image_test
        output_image = cv2.resize(output_image, (480, 480), interpolation = cv2.INTER_LINEAR)
 

    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = str(int(fps))
    
    cv2.putText(output_image, fps, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA) 
    
    cv2.imshow('img',output_image)
    # out.write(output_image)
    
    if cv2.waitKey(1) == ord('q'):
        break
 
# out.release()
cv2.destroyAllWindows()

