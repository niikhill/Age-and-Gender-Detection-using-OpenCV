#author: Nikhil_Chauhan
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pandas as pd
import argparse
import calendar

parser = argparse.ArgumentParser()
parser.add_argument('--image')
img_path = input("Enter image path -> ")
print()
args = parser.parse_args()
sno_list = []
gender_list = []
age_list = []
date_list = []
time_list = []
now = datetime.now()
date = now.strftime("%d/%m/%Y")
month = calendar.month_name[int(now.strftime("%m"))]
time = now.strftime("%I:%M %p")

opencv_home = cv2.__file__
folders = opencv_home.split(os.path.sep)[0:-1]

path = folders[0]
for folder in folders[1:]:
    path = path + "/" + folder

face_detector_path = path+"/data/haarcascade_frontalface_default.xml"

#print("haar cascade configuration found here: ",face_detector_path)

if os.path.isfile(face_detector_path) != True:
    raise ValueError("Open Cv is not Installed, Please Check ->  Expected path ",
                     face_detector_path, " violated.")

haar_detector = cv2.CascadeClassifier(face_detector_path)


def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar_detector.detectMultiScale(gray, 1.3, 5)
    return faces


# Loading Models
age_model = cv2.dnn.readNetFromCaffe(
    "./datasets/age.prototxt", "./datasets/age.caffemodel")
gender_model = cv2.dnn.readNetFromCaffe(
    "./datasets/gender.prototxt", "./datasets/gender.caffemodel")
output_indexes = np.array([i for i in range(0, 101)])


def analysis(img_path):
    img = cv2.imread(img_path)
    print()
    print("Analyzing please wait")
    #plt.imshow(img[:, :, ::-1]); plt.axis('off'); plt.show()

    # detect face

    faces = detect_faces(img)

    for i, face in enumerate(faces):
        x, y, w, h = face
        detected_face = img[int(y):int(y+h), int(x):int(x+w)]
        detected_face = cv2.resize(detected_face, (224, 224))
        img_blob = cv2.dnn.blobFromImage(detected_face)
        # caffe model expects (1, 3, 224, 224) shape input
        # ---------------------------
        age_model.setInput(img_blob)
        age_dist = age_model.forward()[0]
        apparent_predictions = round(np.sum(age_dist * output_indexes), 2)
        #print("Pridicted age: ",apparent_predictions)
        # --------------------------
        gender_model.setInput(img_blob)
        gender_class = gender_model.forward()[0]
        gender = 'Woman ' if np.argmax(gender_class) == 0 else 'Man'
        print("Face No - " + str(i))
        print("Pridicted age: ", apparent_predictions)
        print("Gender: ", gender)
        print(f'Date: {date}')
        print(f'Time: {time}')
        sno_list.append(i)
        gender_list.append(gender)
        age_list.append(int(apparent_predictions))
        date_list.append(date)
        time_list.append(time)
        print()
        # ---------------------------
        cv2.imshow("face", detected_face)
        plt.imshow(detected_face[:, :, ::-1])
        plt.axis('off')
        # plt.savefig('./graphs/analy.png')
        # plt.savefig('./graphs/analy2.png')
        # plt.savefig('./graphs/analy3.png')
        # plt.savefig('./graphs/analy4.png')
        # plt.savefig('./graphs/analy5.png')
        plt.show()


# analysis(args.image)
analysis(img_path)
#file = open('ag.csv', 'w+', newline ='')
dict = {'Gender': gender_list, 'Age': age_list,
        'Date': date_list, 'Time': time_list}
df = pd.DataFrame(dict)

with open('./csv_data/%s.csv' % month, "a") as file:
    df.to_csv('./csv_data/%s.csv' % month, mode='a',
              header=file.tell() == 0, index=False)
    print("=============DATA SAVED IN CSV FILE=============")
# df.to_csv('ag.csv',mode='a',header=f.tell()==0)
# with file:
    # identifying header
    #header = ['Gender', 'Age']
    #writer = csv.DictWriter(file, fieldnames = header)
    # writing data row-wise into the csv file
    # writer.writeheader()
    # writer.writerow({'Gender' : gender_list,
    # 'Age': age_list })
