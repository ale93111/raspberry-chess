import io
import cv2
import time
import torch
import numpy as np
from PIL import Image

import chess
import chess.svg
import cairosvg

from picamera2 import Picamera2

from myutils import get_reference_corners, calibrate_image, predict_yolo

board = chess.Board()
x_chess_board = 'abcdefgh'

corners_ref = get_reference_corners()

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

old_centers = None

while True:
    new_frame_time = time.time() 
    img = picam2.capture_array()
    image_test = img[:,:,:3]

    image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    image_test_rgb = cv2.cvtColor(image_test, cv2.COLOR_BGR2RGB)
    height, width = image_test.shape[:2]

    ret_test, corners_test = cv2.findChessboardCornersSB(image_test_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
    
    ret, output_image = calibrate_image(image_test_rgb, corners_ref, shape_ref, height, width)
    
    if ret:
        predictions_bboxes, new_centers = predict_yolo(output_image, model, shape_ref)  
        
        if old_centers is None:
            old_centers = new_centers 
        
        new_pos = [center for center in new_centers if center not in old_centers]
        
        move = ''
        if new_pos is not None:
            old_pos = [center for center in old_centers if center not in new_centers]

            for i in range(len(new_pos)):
                move += x_chess_board[int(old_pos[i][0]) - 1] + str(int(old_pos[i][1]))
                move += x_chess_board[int(new_pos[i][0]) - 1] + str(int(new_pos[i][1]))
        
            if chess.Move.from_uci(move) in board.legal_moves:
                board.push_uci(move)
            else:
                print(move)
                
        lastmove = chess.Move.from_uci(move) if move else move
        output_image = Image.open(io.BytesIO(cairosvg.svg2png(chess.svg.board(board, lastmove=lastmove, size=480))))
                

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

