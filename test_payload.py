from keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

if __name__ == '__main__':
    label_index = { 0:'payload: NO', 1:'payload: YES'}

    # Load the trained classifier
    model = load_model('./checkpoints_payload/step2_epoch-023_acc-0.999275.h5')

    # Process the input image
    path = './payload_dataset/0/000001-cropped.jpg'
    img = image.load_img(path, color_mode='rgb', target_size=(300, 300))
    img_array = image.img_to_array(img) # convert to a Numpy array

    # Insert a new dimension of 1 at 0-position (i.e., batch-size dimension)
    img_array = np.expand_dims(img_array, axis=0)

    # In[3: Make prediction
    scores = model.predict(img_array)
    out_score = np.max(scores)
    out_label = label_index[scores.argmax(axis=-1)[0]]
    print('--------------------------')
    print(path)
    print('Scores: {}'.format(scores))
    print('Label: {}. Score: {}'.format(out_label, out_score))