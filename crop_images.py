import cv2
import os

DATA_DIR = "./data"
OUTPUT_DIR = "./data1"

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

for dirName in os.listdir(DATA_DIR):
    for imgPath in os.listdir(os.path.join(DATA_DIR, dirName)):
        #print(os.path.join(OUTPUT_DIR, dirName, imgPath))
        img = cv2.imread(os.path.join(DATA_DIR, dirName, imgPath))
        crop_img = img[0:9200, 0:1500]

        os.makedirs(os.path.join(OUTPUT_DIR, dirName), exist_ok=True)  # create the directory

        output_path = os.path.join(OUTPUT_DIR, dirName, imgPath)
        cv2.imwrite(output_path, crop_img)

cv2.destroyAllWindows()
