import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Reshape, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from attention_layer import Attention

IMG_SIZE = 128

def load_data(path):
    X=[]
    y=[]

    for label,name in enumerate(["real","fake"]):
        folder=os.path.join(path,name)

        for img in os.listdir(folder):
            img_path=os.path.join(folder,img)
            image=cv2.imread(img_path)

            if image is None:
                continue

            image=cv2.resize(image,(IMG_SIZE,IMG_SIZE))
            X.append(image)
            y.append(label)

    return np.array(X)/255.0, np.array(y)

# LOAD DATA
X,y = load_data("spectrograms")

print("Dataset size:", len(X))

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# ================= MODEL =================

input_layer=Input(shape=(128,128,3))

# 🔥 LIGHTWEIGHT CNN (FIXED)
x=Conv2D(16,(3,3),activation='relu')(input_layer)
x=MaxPooling2D(2,2)(x)

x=Conv2D(32,(3,3),activation='relu')(x)
x=MaxPooling2D(2,2)(x)

x=Flatten()(x)

# Convert to sequence
x=Reshape((1,-1))(x)

# 🔥 Smaller LSTM
x=LSTM(64,return_sequences=True)(x)

# Attention
x=Attention()(x)

x=Dense(32,activation='relu')(x)
x=Dropout(0.3)(x)

output=Dense(1,activation='sigmoid')(x)

model=Model(inputs=input_layer,outputs=output)

model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train,y_train,epochs=15,batch_size=8)

model.save("deepfake_model.keras")

print("✅ Model Training Complete.")
