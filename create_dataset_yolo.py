import os
import cv2

from myutils import fen_to_board, bboxes_from_chess_board, bbox_to_yolobbox

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
