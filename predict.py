import numpy as np 
import tensorflow as tf
import tensorflow_hub as hub
from utility_function import get_args, process_images
from PIL import Image
import json

args= get_args()

with open('label_map.json', 'r') as f:
    flower = json.load(f)

def predict(image,model,top_k):
   
    imageLoad = Image.open(image)
    imageNp = np.asarray(imageLoad)
    imageProcess = process_images(imageNp)
    imageFinal = np.expand_dims(imageProcess,axis=0)
    
    probs = model.predict(imageFinal)
    prob_predictions= probs[0].tolist()
    probs_final, classes = tf.math.top_k(prob_predictions, k=args.top_k)
    
    probs_list = probs_final.numpy().tolist()
    index_shifted = classes.numpy()+1
    index = index_shifted.tolist()
    
    return probs_list, index

model_path="./model_project.h5"
model=tf.keras.models.load_model(model_path, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)


def checking(image_path,model,top_k):
    probs, classes = predict(image_path, model,top_k)   
    print(probs)
    print(classes)
    flower_type = str(classes[0])
    print('The predicted flower class is:',flower[flower_type])
    
    return probs, classes
   

checking(args.image_path,model,args.top_k)


