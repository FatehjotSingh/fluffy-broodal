import cv2
import numpy as np
import pandas as pd
#import seaborn as sbr
#import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml as fml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score as acc
from PIL import Image as ig
import PIL.ImageOps as ops
import os, ssl, time

if not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context',None) :
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('csvs/image.npz')['arr_0']
Y = pd.read_csv('csvs/33.csv')['labels']
print(pd.Series(Y).value_counts())


classes = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
n_classes = len(classes)

Xtrain,Xtest,Ytrain,Ytest = tts(X, Y , random_state = 9, train_size = 7500, test_size = 2500)

Xtrain_scaled = Xtrain/255.0
Xtest_scaled = Xtest/255.0


clf = lr(solver = 'saga', multi_class = 'multinomial').fit(Xtrain_scaled, Ytrain)


Y_pred = clf.predict(Xtest_scaled)
accuracy = acc(Ytest, Y_pred)
print(accuracy)


capture = cv2.VideoCapture(0)

while(True):
    try:
        ret,frame = capture.read(0)

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        height,width = gray.shape

        upper_left = (int(width/2 -56),int(height/2-56))
        bottom_right = (int(width/2 +56),int(height/2+56))

        cv2.rectangle(gray, upper_left, bottom_right, (0,255,0), 2)
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        im_pil = ig.fromarray(roi)
        img_bw = im_pil.convert('L')
        img_bw_resized = img_bw.resize((22,30),ig.ANTIALIAS)

        img_bw_resized_inverted = ops.invert(img_bw_resized)
        pixel_filter = 20

        min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted - min_pixel,0 ,255)

        max_pixel = np.max(img_bw_resized_inverted)
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel

        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1,784)
        test_pred = clf.predict(test_sample)

        print(test_pred)
        cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('esc'):
            break    

    except Exception as e:
        pass

capture.release()
cv2.destroyAllWindows()