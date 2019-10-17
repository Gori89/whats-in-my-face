from keras.models import model_from_json
import json
import src.constants as const
import numpy as np
import random


from keras import backend as K
import tensorflow as tf

with open(const.MODEL_ROOT+const.MODEL+'.json','r') as f:
    model_json = json.load(f)

model = model_from_json(model_json)
model.load_weights(const.MODEL_ROOT+const.MODEL+'.h5')

print('Model loaded')


tf_session= K.get_session()
#tf_session= tf.compat.v1.Session()
tf_graph = tf.get_default_graph()
#tf_graph.finalize()

def predict(presence,img):
    '''
    img -> numpy array with shape (218, 178, 3)
    return Array of floats from 0 to 1. Each number represents the probability of having an attribute.
           The order of the attributes is defined in the constant ATTRIBUTES
    '''

    global tf_graph,tf_session

    if presence:
        with tf_session.as_default():
            with tf_graph.as_default():
                predictions=model.predict(np.array([img]))
                #predictions=[random.random(),0.6,0.3,0.2,0.3,0.8,0.7,0.5,0.8,0.6,0.7,0.5,0.6,0.2,0.9,0.4,0.5,0.6,0.7,0.3,0.2,0.9]
    else:
        predictions=[[0]*len(const.ATTRIBUTES)]

    result={}
    #print(predictions)
    for i, pred in enumerate(predictions[0]):

        result[const.ATTRIBUTES[i]]=int(pred*100)
    print(result)    
    return result 


