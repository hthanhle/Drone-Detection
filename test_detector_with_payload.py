from PIL import Image, ImageFont, ImageDraw
import colorsys
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input
import keras.backend as K
from yolo3.model import tiny_yolo_body, yolo_eval
from yolo3.utils import letterbox_image, get_classes, get_anchors
from tensorflow.keras.models import load_model
import config as cfg


def detect_image(trained_yolov3, image, model_image_size, input_image_shape, boxes, scores, classes, class_names, font,
                 payload_classifier):
    """
    Detect drone objects with/without payload
    @param trained_yolov3: trained YOLOv3 detector
    @param image: input image
    @param model_image_size: expected image shape
    @param input_image_shape: image size
    @param boxes: boxes
    @param scores: scores
    @param classes: classes
    @param class_names: class names
    @param font: font
    @param payload_classifier: trained payload classifier
    @return: output image
             output bounding boxes
             output scores
             output classes
    """
    label_index = {0: 'payload NO', 1: 'payload YES'}
    if model_image_size != (None, None):
        assert model_image_size[0] % 32 == 0, 'Multiples of 32 required'
        assert model_image_size[1] % 32 == 0, 'Multiples of 32 required'
        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
    else:
        new_image_size = (image.width - (image.width % 32),
                          image.height - (image.height % 32))
        boxed_image = letterbox_image(image, new_image_size)

    image_data = np.array(boxed_image, dtype='float32')

    # print(image_data.shape)
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

    sess = K.get_session()
    out_boxes, out_scores, out_classes = sess.run(
        [boxes, scores, classes],
        feed_dict={
            trained_yolov3.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })

    print('Found {} objects'.format(len(out_boxes)))

    # thickness = (image.size[0] + image.size[1]) // 300
    thickness = 3
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print('Object type: {}, Score: {:.2f}, Location: '.format(predicted_class, score), [left, top], [right, bottom])

        # Crop the object of interest
        drone_object = image.crop((left, top, right, bottom))
        drone_object = drone_object.resize((300, 300))  # resize to the expected input shape
        drone_object = np.asarray(drone_object)  # convert the cropped image to a Numpy array
        drone_object = np.expand_dims(drone_object,
                                      axis=0)  # insert a dimension of 1 at 0-th position (i.e. batch size)
        pred_probas = payload_classifier.predict(drone_object)  # make prediction
        # out_proba    = np.max(pred_probas)  # get the max probability
        out_label = label_index[pred_probas.argmax(axis=-1)[0]]  # get the corresponding label
        label = '{}:{:.2f}'.format(out_label,
                                   score)  # Note that we are outputing the score obtained by the detector. If we would like to ouput the softmax score obtained by the classifier, just replace "score" with "out_proba"
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        if out_label == 'payload NO':
            color = (255, 0, 0)
        else:
            color = (0, 0, 255)
        # Draw multiple rectangles to thicken the outline
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=color)

        # Draw an outside rectangle to cover the text
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)], fill=color)

        # Draw the text
        draw.text(text_origin, label, fill=(255, 255, 255), font=font)
        del draw
    return image, out_boxes, out_scores, out_classes


def test_script(trained_yolov3, anchors, input_shape, file_path, classes_path,
                out_path, font_path, payload_classifier):
    """
    Test script
    @param trained_yolov3: trained YOLOv3 detector
    @param anchors: anchors
    @param input_shape: input shape
    @param file_path: path of input image
    @param classes_path: path of class file
    @param out_path: path of output image
    @param font_path: path of font file
    @param payload_classifier: trained payload classifier
    """
    # Define the font, class names, and colors for visualization
    font = ImageFont.truetype(font=font_path, size=16)
    class_names = get_classes(classes_path)
    # Create a YOLO instance 
    score = 0.3
    iou = 0.4
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(trained_yolov3.output, anchors,
                                       len(class_names), input_image_shape,
                                       score_threshold=score,
                                       iou_threshold=iou)  # Note that "model.output" or "model.outputs" does not matter (need double-check?)

    image = Image.open(file_path)
    image = image.convert('RGB')

    # Visualize the dectected objects
    detected_image, _, _, _ = detect_image(trained_yolov3, image, input_shape,
                                           input_image_shape, boxes, scores,
                                           classes, class_names, font,
                                           payload_classifier)

    detected_image.save(out_path + os.path.sep + os.path.splitext(os.path.basename(file_path))[0] + '.jpg')


if __name__ == '__main__':
    # Load the configurations from file
    classes_path = cfg.classes_path
    anchors_path = cfg.anchors_path
    model_path = cfg.model_path
    out_path = cfg.out_path
    font_path = cfg.font_path
    in_folder = cfg.in_folder
    payload_path = cfg.payload_path

    # Load the trained payload classifier
    payload_classifier = load_model(payload_path)
    print('Loading the payload classifier done')

    # Get the image list
    images = [img for img in os.listdir(in_folder) if img.endswith('.jpg')]
    images = sorted(images, key=lambda s: int(s[6:-4]))

    # Get class names and pre-defined anchors
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # Create a new (tiny) YOLOv3 model, then load the trained weights
    trained_yolov3 = tiny_yolo_body(Input(shape=(None, None, 3)), len(anchors) // 2, num_classes)
    trained_yolov3.load_weights(model_path)
    print('Loading the YOLOv3 detector done')

    count = 0
    # Process the images in the list
    for image in images:
        count += 1
        fullpath = in_folder + '/' + image
        test_script(trained_yolov3, anchors, (832, 640), fullpath, classes_path,
                    out_path, font_path, payload_classifier)
        print('{}'.format(count))
