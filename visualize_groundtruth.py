from PIL import Image, ImageDraw
import numpy as np


def visualize_groundtruth(annotation_string, font, class_names,
                          colors, thickness=1):
    info = annotation_string.split(' ')
    file_path = info[0]  # filepath is the first value
    image = Image.open(file_path)
    image = image.convert('RGB')
    draw = ImageDraw.Draw(image)

    for i in range(1, len(info)):  # get the object boxes from the second value
        # Get the bounding box and the object type
        box = info[i].split(',')
        left = int(box[0])
        top = int(box[1])
        right = int(box[2])
        bottom = int(box[3])
        obj = int(box[4])

        # Get the coordinate of the text
        label = '{}'.format(class_names[obj])
        label_size = draw.textsize(label, font)
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Draw multiple rectangles to thicken the outline
        for i in range(thickness):
            draw.rectangle(
                [left + i, top + i, right - i, bottom - i],
                outline=colors[obj])

        # Draw an outside rectangle to cover the text
        draw.rectangle(
            [tuple(text_origin), tuple(text_origin + label_size)],
            fill=colors[obj])

        # Draw the text
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
    return image
