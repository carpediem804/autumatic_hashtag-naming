import os
import argparse
from torchvision import models
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils import *
import torch.utils.data as data
import torchvision.datasets as dsets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import PIL.Image as pilimg
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import requests
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import openpyxl

def image_loader(image_name):
    """load image, returns cuda tensor"""
    
    loader = transforms.Compose([            
            transforms.Resize(256),
            transforms.CenterCrop(227),
            transforms.ToTensor()
            ])
    
    image = pilimg.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image  #assumes that you're using GPU


def data_preprocessing(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    size = (128,128)
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    faces = face_cascade.detectMultiScale(image, 1.3, 2)
    
    for (x,y,w,h) in faces:
        cropped = image[y - int(h / 4):y + h + int(h / 4), x - int(w / 4):x + w + int(w / 4)]
        cv2.imwrite('out01.jpg',cropped)
        
    im = Image.open('out01.jpg')
    im.thumbnail(size)
    im.save('resize01.jpg')
    return 'resize01.jpg'

def creat_title(age, output_emotion, gender, face):
    # read data
    data = pd.read_excel('xxx.xlsx')
    name_data = pd.DataFrame(data)
    name_data = name_data.fillna(0)
    X_train = name_data.iloc[:,1:]
    Y_train = name_data.iloc[:,0]
    neigh = KNeighborsClassifier(n_neighbors=1)
    neigh.fit(X_train,Y_train)
    
    face_info_onehot = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    if face == 'dog':
        face_info_onehot[0] = 1
    elif face == 'cat':
        face_info_onehot[1] = 1
    elif face == 'fox':
        face_info_onehot[2] = 1
    elif face == 'frog':
        face_info_onehot[3] = 1
    elif face == 'bird':
        face_info_onehot[4] = 1
    elif face == 'dino':
        face_info_onehot[5] = 1
    elif face == 'hores':
        face_info_onehot[6] = 1
        
    if output_emotion == 'anger':
        face_info_onehot[7] = 1
    elif output_emotion == 'contempt':
        face_info_onehot[8] = 1
    elif output_emotion == 'disgust':
        face_info_onehot[9] = 1
    elif output_emotion == 'fear': 
        face_info_onehot[10] = 1
    elif output_emotion == 'happiness': 
        face_info_onehot[11] = 1
    elif output_emotion ==  'neutral':
        face_info_onehot[12] = 1
    elif output_emotion == 'sadness': 
        face_info_onehot[13] = 1
    elif output_emotion == 'surprise':
        face_info_onehot[14] = 1
        
    if age < 10:
        face_info_onehot[15] = 1
    elif 10 <= age < 20:
        face_info_onehot[16] = 1
    elif 10 <= age < 20:
        face_info_onehot[17] = 1
    elif 20 <= age < 25:
        face_info_onehot[18] = 1
    elif 25 <= age < 30:
        face_info_onehot[19] = 1    
    elif 30 <= age < 50:
        face_info_onehot[20] = 1    
    elif 50 <= age:
        face_info_onehot[21] = 1       
        
    if gender == 'male': 
        face_info_onehot[22] = 1    
    elif gender == 'female': 
        face_info_onehot[23] = 1    
        
    face_info_df = pd.DataFrame([face_info_onehot],columns = X_train.columns)
    title = neigh.predict(face_info_df)
    
    return title[0]
    


def main():
    print("Hello Taehong")
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_name", type=str, default="")
    args = parser.parse_args()
    
    # Creat Net
    net_face = models.vgg11_bn(pretrained=True)
    net_first = models.vgg11_bn(pretrained=True)
    
    old_classifier1 = list(net_face.classifier.children()) # get the classifier part alone
    old_classifier1.pop() # remove the last layer
    old_classifier1.append(nn.Linear(4096,7)) # add a new layer
    net_face.classifier = nn.Sequential(*old_classifier1) # attach it to the original vgg mode
    
    old_classifier2 = list(net_first.classifier.children()) # get the classifier part alone
    old_classifier2.pop() # remove the last layer
    old_classifier2.append(nn.Linear(4096,6)) # add a new layer
    net_first.classifier = nn.Sequential(*old_classifier2) # attach it to the original vgg mode  
    
    # Load Net 
    net_face.load_state_dict(torch.load('faceClssifcation_99.pth'))
    net_first.load_state_dict(torch.load('firstFeelingClssifcation_88.pth'))

    # Creast class
    face_classes = ('bird', 'cat', 'dino', 'dog',
           'fox', 'frog', 'horse')
    first_class = ('감정이 과한', '과묵한', '불만 많은', '사연있는',
           '호감형', '화끈한')
    
    # Data preprocessing
    resize_data = data_preprocessing(args.image_name)
    
    # Read image
    image = image_loader(resize_data)
    
    # Output
    output_face = net_face(image)
    output_first = net_first(image)
    
    _, face_preds = torch.max(output_face.data, 1)
    _, first_preds = torch.max(output_first.data, 1)
    
    face = face_classes[np.asarray(face_preds)[0]] 
    first = first_class[np.asarray(first_preds)[0]] 
    
    # microsoft lens
    face_api_url = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0/detect'


    headers = {
      'ocp-apim-subscription-key': 'e02defb79e7a49529cf41bba11713928',
      'Content-Type': "application/octet-stream",
      'cache-control': "no-cache",
    }

    params = {
        'returnFaceAttributes': 'gender,age,emotion',
    }

    data = open(args.image_name, 'rb').read()
    response = requests.post(face_api_url, params=params, headers=headers, data=data)
    faces = response.json()

    dic1 = dict(faces[0])
    dic2 = dic1['faceAttributes']
    
    age = dic2['age']
    gender =dic2['gender']
    
    dic3 = dict(dic2['emotion'])
    inverse = [(value, key) for key, value in dic3.items()]
    emotion = max(inverse)
    output_emotion = emotion[1]
    
    title = creat_title(age, output_emotion, gender, face)
    
    print('**************************Tae Hong*************************')
    print(age)
    print(output_emotion)
    print(gender)
    print(face)
    print(first)
    print(title)

if __name__ == "__main__":
    main()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    