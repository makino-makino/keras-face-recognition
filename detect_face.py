import glob
import os

import cv2
from icecream import ic


def main():
    names = ['murai']
    out_dir = "./data/out/"
    os.makedirs(out_dir, exist_ok=True)
    for i in range(len(names)):

        in_dir = "./data/in/"+names[i]+"/*.jpg"
        in_jpg_files = glob.glob(in_dir)
        os.makedirs(out_dir + names[i], exist_ok=True)
        ic(len(in_jpg_files))
        for num, jpg_file in enumerate(in_jpg_files):
            image = cv2.imread(jpg_file)
            if image is None:
                ic("Could not open:", num)
                continue

            image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cascade = cv2.CascadeClassifier(
                "./haarcascade_frontalface_alt.xml")

            face_list = cascade.detectMultiScale(
                image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

            if len(face_list) < 1:
                print("no face")
                continue
            
            for rect in face_list:
                x, y, width, height = rect
                image = image[rect[1]:rect[1] +
                                rect[3], rect[0]:rect[0]+rect[2]]
                if image.shape[0] < 64:
                    continue
                image = cv2.resize(image, (64, 64))

                fileName = os.path.join(
                    out_dir + names[i], str(num)+".jpg")
                cv2.imwrite(str(fileName), image)
                print(str(num)+".jpgを保存しました.")

            print(image.shape)


if __name__ == "__main__":
    main()
