import os
import cv2

from sklearn.model_selection import train_test_split

directory_path = './datasets/chess/'

output_directory_images = os.path.join(directory_path, 'images/')
output_directory_labels = os.path.join(directory_path, 'labels/')

output_directory_images_all = os.path.join(output_directory_images, 'all')
output_directory_labels_all = os.path.join(output_directory_labels, 'all')

images_path = os.listdir(output_directory_images_all)

images_train, images_valtest = train_test_split(images_path, test_size=0.20, random_state=42)
images_val,   images_test    = train_test_split(images_valtest, test_size=0.25, random_state=42)

for image_path in images_train:
    image = cv2.imread(os.path.join(output_directory_images_all, image_path))
    cv2.imwrite(os.path.join(output_directory_images, 'train/'+image_path), image)
    
    with open(os.path.join(output_directory_labels_all, image_path.replace('.png','.txt'))) as f:
        labels = f.read()
    with open(os.path.join(output_directory_labels, 'train/'+image_path.replace('.png','.txt')), 'w') as f:
        f.write(labels)

for image_path in images_val:
    image = cv2.imread(os.path.join(output_directory_images_all, image_path))
    cv2.imwrite(os.path.join(output_directory_images, 'val/'+image_path), image)
    
    with open(os.path.join(output_directory_labels_all, image_path.replace('.png','.txt'))) as f:
        labels = f.read()  
    with open(os.path.join(output_directory_labels, 'val/'+image_path.replace('.png','.txt')), 'w') as f:
        f.write(labels)
    
for image_path in images_test:
    image = cv2.imread(os.path.join(output_directory_images_all, image_path))
    cv2.imwrite(os.path.join(output_directory_images, 'test/'+image_path), image)
    
    with open(os.path.join(output_directory_labels_all, image_path.replace('.png','.txt'))) as f:
        labels = f.read()
    with open(os.path.join(output_directory_labels, 'test/'+image_path.replace('.png','.txt')), 'w') as f:
        f.write(labels)
