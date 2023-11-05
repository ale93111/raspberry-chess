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

chessboard_image = Image.open(io.BytesIO(cairosvg.svg2png(chess.svg.board(board, lastmove='', size=480))))

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

out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 1.0, (480*2,480))

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
        if len(new_pos) > 0:
            old_pos = [center for center in old_centers if center not in new_centers]

            for i in range(len(new_pos)):
                move += x_chess_board[int(old_pos[i][0]) - 1] + str(int(old_pos[i][1]))
                move += x_chess_board[int(new_pos[i][0]) - 1] + str(int(new_pos[i][1]))
        
            if chess.Move.from_uci(move) in board.legal_moves:
                board.push_uci(move)
            else:
                print(move)
                
            old_centers = new_centers 
                
        lastmove = chess.Move.from_uci(move) if move else move
        chessboard_image = Image.open(io.BytesIO(cairosvg.svg2png(chess.svg.board(board, lastmove=lastmove, size=480))))
        chessboard_image = np.ascontiguousarray(np.array(chessboard_image)[:,:,:3].astype(np.uint8))
        
        for bbox in predictions_bboxes:
            if bbox[4] == 0:
                color = (255, 0, 0)
            else:
                color = (0, 255, 0)
            output_image = cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
    else:
        output_image = image_test_rgb
        output_image = cv2.resize(output_image, shape_ref, interpolation = cv2.INTER_LINEAR)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
    fps = 'fps:'+str(round(fps,2))
    
    cv2.putText(output_image, fps, (7, 70), font, 1, (100, 255, 0), 1, cv2.LINE_AA) 
    
    cv2.imshow('img',np.hstack([output_image,chessboard_image]))
    out.write(np.hstack([output_image,chessboard_image]))
    
    if cv2.waitKey(1) == ord('q'):
        break
 
out.release()
cv2.destroyAllWindows()
