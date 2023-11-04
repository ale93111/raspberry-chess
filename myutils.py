


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

