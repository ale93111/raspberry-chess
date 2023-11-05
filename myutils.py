import cv2
import numpy as np
from PIL import Image

def get_reference_corners():
    perfect_corners = []
    for i in range(1,8):
        for j in range(1,8):
            bbox = [60*i, 60*j]
            perfect_corners.append(bbox)
    
    corners_ref = np.array(perfect_corners)

    return corners_ref


def calibrate_image(image_test_rgb, corners_ref, shape_ref, height, width):
    image_test_gray = cv2.cvtColor(image_test_rgb, cv2.COLOR_RGB2GRAY)
    
    ret_test, corners_test = cv2.findChessboardCornersSB(image_test_gray, (7, 7), cv2.CALIB_CB_EXHAUSTIVE)
    
    if ret_test:
        homography, mask = cv2.findHomography(corners_test, corners_ref, cv2.RANSAC)
        # homography, mask = cv2.findHomography(corners_ref, corners_test, cv2.RANSAC)
  
        transformed_img = cv2.warpPerspective(image_test_rgb, homography, (width, height))
        
        output_image = transformed_img[:shape_ref[0],:shape_ref[1]]
        
    else:
        output_image = None
        # output_image = image_test_rgb
        # output_image = cv2.resize(output_image, shape_ref, interpolation = cv2.INTER_LINEAR)
        # output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
 
    return ret_test, output_image


def predict_yolo(image, model, shape_ref):
    yolo_image = Image.fromarray(image)
    results = model(yolo_image)

    yolo_bboxes = results.xywhn[0].numpy()
    predictions_bboxes = [bbox_yolo_to_pascal_voc(yolo_bbox[0], yolo_bbox[1], yolo_bbox[2], yolo_bbox[3], yolo_bbox[5], shape_ref[0], shape_ref[1]) for yolo_bbox in yolo_bboxes if yolo_bbox[4] > 0.75]
    centers = [[np.floor(yolo_bbox[0]*8) + 1, np.floor(yolo_bbox[1]*8) + 1] for yolo_bbox in yolo_bboxes if yolo_bbox[4] > 0.75]
    
    return predictions_bboxes, centers


def fen_to_board(fen):
    board = []
    for row in fen.split('/'):
        brow = []
        for c in row:
            if c == ' ':
                break
            elif c in '12345678':
                brow.extend( ['--'] * int(c) )
            elif c == 'p':
                brow.append( 'bp' )
            elif c == 'P':
                brow.append( 'wp' )
            elif c > 'Z':
                brow.append( 'b'+c.upper() )
            else:
                brow.append( 'w'+c )

        board.append( brow )
        
    return board


def bboxes_from_chess_board(chess_board_position):
    bboxes = []
    for i in range(8):
        for j in range(8):
            if chess_board_position[i][j].startswith('b'):
                bbox = [1, 60*j+5, 60*i+5, 60*(j+1)-5, 60*(i+1)-5]
                bboxes.append(bbox)
            elif chess_board_position[i][j].startswith('w'):
                bbox = [0, 60*j+5, 60*i+5, 60*(j+1)-5, 60*(i+1)-5]
                bboxes.append(bbox)
            else:
                continue
                # bbox = [0, 60*j+5, 60*i+5, 60*(j+1)-5, 60*(i+1)-5]
                # bboxes.append(bbox)
    return bboxes


def bbox_to_yolobbox(bbox, image_w, image_h):
    class_id, x1, y1, x2, y2 = bbox
    return [class_id, ((x2 + x1)/(2*image_w)), ((y2 + y1)/(2*image_h)), (x2 - x1)/image_w, (y2 - y1)/image_h]


def bbox_yolo_to_pascal_voc(x_center, y_center, w, h, class_id, image_w, image_h):
    w = w * image_w
    h = h * image_h
    x1 = ((2 * x_center * image_w) - w)/2
    y1 = ((2 * y_center * image_h) - h)/2
    x2 = x1 + w
    y2 = y1 + h
    return [int(x1), int(y1), int(x2), int(y2), int(class_id)]

