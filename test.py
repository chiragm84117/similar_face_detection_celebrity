#load img -> detection and extract it feature
#find the cosine distance of currrent image with all the 8655 image
#recomended the image

# from tensorflow.keras.applications.vgg19 import VGG19
# from tensorflow.keras.applications.vgg19 import preprocess_input
import numpy as np
import tensorflow.python.keras
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.layers import GlobalMaxPooling2D
import cv2
from mtcnn import MTCNN
from PIL import Image

feature_list = np.array(pickle.load(open('embedding1.pkl','rb')))
filenames = pickle.load(open('filename.pkl','rb'))

# model = VGG19(input_shape=(224model = ResNet50(input_shape=(224,224,3) , weights='imagenet' , include_top = False , pooling = 'max' )
# model = ResNet50(input_shape=(224,224,3) , weights='imagenet' , include_top = False , pooling = 'avg' )

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])


detector = MTCNN()

sample_img = cv2.imread('Bollywood_celeb_face_localized/Katrina_Kaif/Katrina_Kaif.25.jpg')
# results = detector.detect_faces(sample_img)
#
# # this makethe box around the face
# x,y,width,height = results[0]['box']
#
# # for croping the image
# face = sample_img[y:y+width,x:x+width]
# # cv2.imshow('output',face)
# # cv2.waitKey(0)
# # detection done
#
# # extraction from image start
image = Image.fromarray(sample_img)
image = image.resize((224,224))

face_array = np.asarray(image)
face_array = face_array.astype('float32')
expended_image = np.expand_dims(face_array,axis=0)

preprocessed_image = preprocess_input(expended_image)
result = model.predict(preprocessed_image).flatten()
# print(result)
# print(result.shape)

similarity = []

for i in range(len(feature_list)):
    similarity.append(cosine_similarity(result.reshape(1,-1),feature_list[i].reshape(1,-1))[0][0])

index_pos = sorted(list(enumerate(similarity)),reverse=True, key = lambda x:x[1])[0][0]
temp_img = cv2.imread(filenames[index_pos])
cv2.imshow('output',temp_img)
cv2.waitKey(0)