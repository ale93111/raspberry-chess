import os
import cv2

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


counter_image = 0

output_directory = './datasets/chess/'

directory_path = './archive/raw/1/'

subdirectories_path = sorted(os.listdir(directory_path))

for subdirectory_path in subdirectories_path:
    directory_files = os.listdir(directory_path+subdirectory_path)

    directory_images = [file for file in directory_files if file.endswith(('.png', '.jpg', '.jpeg'))]

    with open(os.path.join(directory_path+subdirectory_path, "board.fen")) as f:
        fen_position = f.read()

    chess_board_position = fen_to_board(fen_position)
    bboxes = bboxes_from_chess_board(chess_board_position)
    bboxes_yolo = [bbox_to_yolobbox(bbox, 480, 480) for bbox in bboxes]

    output_directory_labels = os.path.join(output_directory, 'labels/all/')
    output_directory_images = os.path.join(output_directory, 'images/all/')

    for image_path in directory_images:
        image = cv2.imread(os.path.join(directory_path+subdirectory_path, image_path))
        
        cv2.imwrite(os.path.join(output_directory_images, str(counter_image)+'.png'), image)
        
        labels = [f"{bbox_yolo[0]} {bbox_yolo[1]} {bbox_yolo[2]} {bbox_yolo[3]} {bbox_yolo[4]}" for bbox_yolo in bboxes_yolo]
        labels = '\n'.join(labels)

        with open(os.path.join(output_directory_labels, str(counter_image)+".txt"),'w') as f:
            f.write(labels)
            
        counter_image += 1
