# -*- coding: utf-8 -*-
"""

@author: Yassine
"""
###########################################LIBRAIRIES####################################################
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import ImageTk, Image
import numpy
from keras.models import load_model
import pickle
import cv2
import mahotas

##########################################LOADING MODELS################################################
#Model NN
ModelNN = load_model('modNN.h5')

#Model ML
ModelML = pickle.load(open('ModelML.sav', 'rb'))

##########################################UNPICKLING CLASSES###########################################
#Function to unpickle data:
def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
#Unpickling classes
classes = unpickle("cifar-10-batches-py/batches.meta")
classes = classes[b'label_names']

##########################################DESCRIPTORS#################################################
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# feature-descriptor-2: Haralick Texture
def fd_haralick(image):
    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    # return the result
    return haralick

# feature-descriptor-3: Color Histogram
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    # return the histogram
    return hist.flatten()

#####################################INTERFACE#######################################################
#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Classification des images')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',12,'bold'))
label1=Label(top,background='#CDCDCD', font=('arial',12,'bold'))
label2=Label(top,background='#CDCDCD', font=('arial',12,'bold'))
label3=Label(top,background='#CDCDCD', font=('arial',12,'bold'))
label4=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
label5=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

#Fonction de classification NN
def classifyNN(file_path):
    global label_packed
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    image = numpy.array(image)
    image = image.reshape(32, 32, 3) 
    image = numpy.expand_dims(image, axis=0)
    image = image.astype('float32')/255
    pred = ModelNN.predict_classes([image])[0]
    sign = classes[pred]
    proba = ModelNN.predict(image)[0]
    idx = numpy.argmax(proba)
    label.configure(foreground='#011638', text=sign) 
    label1.configure(foreground='#011638', text="Precision: {:.2f}%".format(proba[idx]*100))
    label4.configure(foreground='#011638', text="Prediction des reseaux de neuronnes:")
    

#Fonction de classification ML
def classifyML(file_path):
    global label_packed
    image = cv2.imread(file_path)
    image = cv2.resize(image, (32,32), interpolation = cv2.INTER_AREA)
    image = numpy.array(image)
    image = image.reshape(32, 32, 3) 
    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)    
    Stack = numpy.hstack([fv_histogram, fv_haralick, fv_hu_moments])
    Stack = numpy.array(Stack.reshape(1,-1))
    pred = ModelML.predict(Stack)
    sign = classes[int(pred)]
    proba = ModelML.predict_proba(Stack)[0]
    idx = numpy.argmax(proba)
    label2.configure(foreground='#011638', text=sign) 
    label3.configure(foreground='#011638', text="Precision: {:.2f}%".format(proba[idx]*100))
    label5.configure(foreground='#011638', text="Prediction du Stacking Classifier:")

#Fonction d'affichage du boutton classifier
def show_classify_button(file_path):
    classify_b=Button(top,text="Predire la classe",
   command=lambda: [classifyNN(file_path),classifyML(file_path)],
   padx=1,pady=5)
    classify_b.configure(background='#364156', foreground='white',
font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
    
#Fonction d'upload
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),
    (top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass
upload=Button(top,text="Choisir une image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
def close_window (): 
    top.destroy()
quitter = Button (top,text = "Quitter", command = close_window, padx=10,pady=5)
quitter.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
quitter.pack(side=BOTTOM,pady=15)
upload.pack(side=BOTTOM,pady=15)
sign_image.pack(side=BOTTOM,expand=True)
label3.pack(side=BOTTOM,expand=True)
label2.pack(side=BOTTOM,expand=True)
label5.pack(side=BOTTOM,expand=True)
label1.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
label4.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Classification des images",pady=15, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
top.mainloop()

