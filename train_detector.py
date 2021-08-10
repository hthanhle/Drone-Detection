import numpy as np
from keras.optimizers import Adam
from yolo3.model import preprocess_true_boxes, create_model
from yolo3.utils import get_random_data, get_classes, get_anchors, data_generator
import config as cfg

if __name__ == '__main__':
    # Load the configurations from file
    input_shape = cfg.input_shape
    is_tiny_version = cfg.is_tiny_version
    load_pretrained = cfg.pretrained
    out_model = cfg.out_model_path
    classes_path = cfg.classes_path
    anchors_path = cfg.anchors_path
    train_annotation = cfg.train_annotation_path
    test_annotation = cfg.test_annotation_path

    if is_tiny_version:
        if load_pretrained:
            pretrained_weights = './model_data/yolov3-tiny.h5'
    else:
        if load_pretrained:
            pretrained_weights = './model_data/darknet53_weights.h5'

    class_names = get_classes(classes_path)
    num_classes = len(class_names)
    anchors = get_anchors(anchors_path)

    # Create a YOLOv3 model with the pre-trained weights
    model = create_model(input_shape, anchors, num_classes,
                         pretrained_weights,
                         is_tiny_version=is_tiny_version,
                         load_pretrained=load_pretrained)
    model.summary()

    # Load the training set
    with open(train_annotation) as f:
        train_lines = f.readlines()
    num_train = len(train_lines)

    with open(test_annotation) as f:
        test_lines = f.readlines()
    num_test = len(test_lines)

    # In[1.5]: TRAINING STAGE 1: Train the model with the frozen layers to get
    # a stable loss
    batch_size = 4
    model.compile(optimizer=Adam(lr=1e-3), loss={
        'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, test on {} samples, with batch size {}.'
          .format(num_train, num_test, batch_size))

    model.fit_generator(data_generator(train_lines[:num_train], batch_size,
                                       input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        epochs=50, initial_epoch=0)
    model.save_weights(out_model + 'trained_full_yolo_drone_stage_1.h5')

    # In[1.6]: TRAINING STAGE 2: Unfreeze and continue training to fine-tune 
    # the model. Training could be longer if the result is not good.
    batch_size = 4
    for i in range(len(model.layers)):
        model.layers[i].trainable = True  # unfreeze all layers

    print('Reduce the learning rate')
    model.compile(optimizer=Adam(lr=1e-4),  # train with a small learning rate
                  loss={'yolo_loss': lambda y_true, y_pred: y_pred})
    print('Train on {} samples, test on {} samples, with batch size {}.'
          .format(num_train, num_test, batch_size))

    model.fit_generator(data_generator(train_lines[:num_train], batch_size,
                                       input_shape, anchors, num_classes),
                        steps_per_epoch=max(1, num_train // batch_size),
                        epochs=200, initial_epoch=50)

    model.save_weights(out_model + 'trained_full_yolo_drone_finnished.h5')
