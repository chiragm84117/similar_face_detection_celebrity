import streamlit as st
from PIL import Image
import os
import cv2
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import GlobalMaxPooling2D
import tensorflow.python.keras

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])



detector = MTCNN()
feature_list = np.array(pickle.load(open('embedding1.pkl','rb')))
filenames = pickle.load(open('filename.pkl','rb'))

def save_uploaded_file(uploaded_image):
    try:
        with open(os.path.join('uploads',uploaded_image.name),'wb') as f:
            f.write(uploaded_image.getbuffer())
        return True
    except:
        return False

def extract_feature(img_path,model,detector):
    img = cv2.imread(img_path)
    results = detector.detect_faces(img)
    x, y, width, height = results[0]['box']

    # for croping the image
    face = img[y:y + width, x:x + width]
    # cv2.imshow('output',face)
    # cv2.waitKey(0)
    # detection done

    # extraction from image start
    image = Image.fromarray(face)
    image = image.resize((224, 224))

    face_array = np.asarray(image)
    face_array = face_array.astype('float32')
    expended_image = np.expand_dims(face_array, axis=0)

    preprocessed_image = preprocess_input(expended_image)
    result = model.predict(preprocessed_image).flatten()
    return result

def recommend(feature_list,features):
    similarity = []

    for i in range(len(feature_list)):
        similarity.append(cosine_similarity(features.reshape(1, -1), feature_list[i].reshape(1, -1))[0][0])
    index_pos = sorted(list(enumerate(similarity)), reverse=True, key=lambda x: x[1])[0][0]
    return index_pos


st.title('which bollywood celebrity are you?')

uploaded_img = st.file_uploader('choose an image')

if uploaded_img is not None:
#     save in directory
    if save_uploaded_file(uploaded_img):
#         load
#           feature extract
#          recomed
        display_img = Image.open(uploaded_img)
        # st.image(display_img)

        features = extract_feature(os.path.join('uploads',uploaded_img.name),model,detector)
        # st.text(features)
        # st.text(features.shape)
        index_pos = recommend(feature_list,features)
        predicted_actor = " ".join(filenames[index_pos].split('\\')[1].split('_'))
        # st.text(index_pos)
        col1,col2 = st.columns(2)
        with col1:
            st.header('Your upload image')
            st.image(display_img,width = 300)
        with col2:

            st.header('seems like '+predicted_actor)
            st.image(filenames[index_pos],width = 300)

# display