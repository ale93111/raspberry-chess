# raspberry-chess
Chess &amp; Computer Vision for Raspberry Pi

![Alt Text](https://github.com/ale93111/raspberry-chess/blob/main/assets/visualization.gif)

## Introduction
This is my first project for Raspberry Pi and since I am also interested in chess I wanted to make a system able to track a chess game and show the evaluation and best moves in real time.

My equipment:
- Raspberry Pi 4 model B 8GB RAM
- Raspberry Pi Camera Module 3
- Mini portable chess set
- Red/Green acrylic paint

I decided to paint my chess pieces red/green to enhance their visibility and ease of detection by the computer vision system against the background.

## Computer Vision &amp; Machine Learning 
The chessboard is detected with the function cv2.findChessboardCornersSB(), then I compute the perspective transform with cv2.findHomography()

Chess pieces are detected with a YOLOv5n model with runs at 0.6 FPS on the Raspberry Pi. The model is trained on this publicly available dataset: [Raspberry Turk Chess Dataset](https://www.kaggle.com/datasets/joeymeyer/raspberryturk)

Model weights are provided.
