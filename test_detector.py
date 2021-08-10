"""
Reference: https://github.com/qqwweee/keras-yolo3
"""
from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
from keras.layers import Input
import keras.backend as K
from yolo3.model import tiny_yolo_body, yolo_eval
from yolo3.utils import letterbox_image, get_classes, get_class_colors, get_anchors
import config as cfg


def detect_image(model, image, model_image_size, input_image_shape, boxes,
                 scores, classes, class_names, font):
    """
    Detect drone objects in a single input image
    @param model: trained YOLOv3 model
    @param image: input image
    @param model_image_size: image size
    @param input_image_shape: expected input shape
    @param boxes: boxes
    @param scores: scores
    @param classes: classes
    @param class_names: class names
    @param font: font
    @return: image with detected objects
             output bounding boxes
             output scores
             output classes
    """
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
            model.input: image_data,
            input_image_shape: [image.size[1], image.size[0]],
            K.learning_phase(): 0
        })
    print('Found {} objects'.format(len(out_boxes)))

    colors = get_class_colors(class_names)
    # thickness = (image.size[0] + image.size[1]) // 300
    thickness = 3
    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{}:{:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print('Object type: {}, Score: {:.2f}, Location: '
              .format(predicted_class, score), [left, top], [right, bottom])

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Draw multiple rectangles to thicken the outline
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[c])

        # Draw an outside rectangle to cover the text
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[c])

        # Draw the text
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw
    return image, out_boxes, out_scores, out_classes


def test_script(model, anchors, input_shape, file_path, classes_path,
                out_fig_path, font_path):
    """
    Test script
    @param model: trained YOLOv3 model
    @param anchors: anchors
    @param input_shape: input shape
    @param file_path: path of input image
    @param classes_path: path of class file
    @param out_fig_path: path of output image
    @param font_path: path of font file
    """
    # Define the font, class names, and colors for visualization
    font = ImageFont.truetype(font=font_path, size=25)
    class_names = get_classes(classes_path)

    # Create a new YOLO model 
    score = 0.3
    iou = 0.4
    input_image_shape = K.placeholder(shape=(2,))
    boxes, scores, classes = yolo_eval(model.output, anchors, len(class_names), input_image_shape,
                                       score_threshold=score, iou_threshold=iou)

    # Load the input image
    image = Image.open(file_path)
    image = image.convert('RGB')

    # Visualize the detected objects
    out_image, _, _, _ = detect_image(model, image, input_shape, input_image_shape, boxes, scores, classes, class_names,
                                      font)

    out_image.save(out_fig_path + os.path.splitext(os.path.basename(file_path))[0] + '.jpg')
    print(file_path + ' done')


if __name__ == '__main__':
    # Load the configurations
    classes_path = cfg.classes_path
    anchors_path = cfg.anchors_path
    model_path = cfg.model_path
    out_path = cfg.out_path
    font_path = cfg.font_path
    in_folder = cfg.in_folder

    # Get the image list
    images = [img for img in os.listdir(in_folder) if img.endswith(".jpg")]
    # images = sorted(images, key=lambda s: int(s[6:-4]))

    # Get class names and pre-defined anchors
    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # Create a new YOLOv3 model, then load the trained weights
    trained_model = tiny_yolo_body(Input(shape=(None, None, 3)), len(anchors) // 2,
                                   num_classes)
    trained_model.load_weights(model_path)
    print('Loading the trained weights done')

    # Process the images in the list
    for image in images:
        fullpath = in_folder + '/' + image
        test_script(trained_model, anchors, (832, 640), fullpath, classes_path, out_path, font_path)
