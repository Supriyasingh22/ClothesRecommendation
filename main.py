import streamlit as st
import os
from PIL import Image
import numpy as np
import sklearn
import pickle
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm


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


#for title
st.title('Clothes Recommender system')

# function foe ulpoaded file to dabe in some directory
def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads',uploaded_file.name),'wb') as f: # we are opening this file in write binary mode
            f.write(uploaded_file.getbuffer())
        return 1
    except:
        return 0

# for feature extraction
def feature_extraction(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224), interpolation='bicubic')
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result

#for recommendation here is this function
def recommend(features,feature_list):
    closests = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    closests.fit(feature_list)

    distances, indices = closests.kneighbors([features])
    return indices

# file upload
uploaded_file = st.file_uploader("Choose an Image") #file_uploader is a funciton for uploading file in streamlit
if uploaded_file is not None:
   if save_uploaded_file(uploaded_file):
       # after ulpoading I need to display it
        display_image = Image.open(uploaded_file)
        st.image(display_image)
        features = feature_extraction(os.path.join("uploads",uploaded_file.name),model)
        #st.text(features)
       #for recommendation we will call it

        result = recommend(features,feature_list)
       # now result have all the closets images  time to display it
        pic1,pic2,pic3,pic4,pic5 = st.columns(5) #to get five columns
        with pic1:
            st.image(filenames[result[0][0]])
        with pic2:
            st.image(filenames[result[0][1]])
        with pic3:
            st.image(filenames[result[0][2]])
        with pic4:
            st.image(filenames[result[0][3]])
        with pic5:
            st.image(filenames[result[0][4]])

   else:
       st.header("some error occured in file upload ")

