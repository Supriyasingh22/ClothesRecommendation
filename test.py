import pickle
import tensorflow
import numpy as np
import sklearn
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors
# to display images we used open cv
import cv2

# loading feature list
feature_list = np.array(pickle.load(open('embedding.pkl','rb')))
# print(np.array(feature_list).shape)
# loading filename
filenames = pickle.load(open('filenames.pkl','rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

# Create an instance of the GlobalMaxPooling2D layer
global_max_pooling_layer = GlobalMaxPooling2D()

# Build the model using tensorflow.keras.Sequential
model = tensorflow.keras.Sequential([
    model,
    global_max_pooling_layer  # Add the instance of GlobalMaxPooling2D layer
])

img = image.load_img('sample/img2.jpg',target_size=(224,224),interpolation='bicubic')
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array,axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

normalized_result_2d = normalized_result.reshape(1, -1)

# to calculate diatnce between normalized_result and feature list

neighbors = NearestNeighbors(n_neighbors=6,algorithm='brute',metric='euclidean')
neighbors.fit(feature_list)

distances, indices = neighbors.kneighbors(normalized_result_2d)

print(indices)

for file in indices[0][1:6]:
    temp = cv2.imread(filenames[file])
    cv2.imshow('output',cv2.resize(temp,(512,512)))
    cv2.waitKey(0)

