import os
import random
#import cv2
import shutil
from sklearn.model_selection import train_test_split

scr= "/Users/currentwire/Documents/sign-language-alphabet-recognizer-master/dataset"
dest="/Users/currentwire/Documents"
for i in os.listdir(scr):
    #random.shuffle(scr+"/"+str(i))
    if not i.startswith("."):

        X_train,X_test=train_test_split(os.listdir(scr+"/"+str(i)),test_size=0.33,random_state=42)
        for j in X_train:
             if not os.path.exists(dest+"/train_set/"+str(i)):
                dis=os.makedirs(dest+"/train_set/"+str(i))

                dis=dest+"/train_set/"+str(i)
             else:
                dis=dest+"/train_set/"+str(i)
             shutil.copy(scr+"/"+str(i)+"/"+str(j),dis+"/"+str(j))

        for k in X_test:
            if  not os.path.exists(dest + "/test_set/" + str(i)):

                dis2 = os.makedirs(dest + "/test_set/" + str(i))


                dis2 = dest + "/test_set/" + str(i)
            else:

                dis2 = dest + "/test_set/" + str(i)
             #shutil.copy(scr+"/"+str(i)+"/"+str(j),dis+"/"+str(j))
            shutil.copy(scr+"/"+str(i)+"/"+str(k),dis2+"/"+str(k))
