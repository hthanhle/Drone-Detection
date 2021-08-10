from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from read_image_folder import read_image_folder
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def create_model():
    base_model = InceptionV3(weights='imagenet', include_top=False)
    for layer in base_model.layers:
        layer.trainable = False

    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Add the fully-connected layers
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)  # 2 payload classes: 0 and 1
    model = Model(inputs=base_model.input, outputs=predictions)

    return model


if __name__ == '__main__':
    # Load the payload dataset, and convert the labels to one-hot vectors
    images, labels = read_image_folder('payload_dataset', image_size=(300, 300), color_mode='rgb')
    labels = np_utils.to_categorical(labels, num_classes=None, dtype='float32')

    # Create the model
    model = create_model()
    checkpoint = ModelCheckpoint('checkpoint_payload/epoch-{epoch:03d}_acc-{acc:03f}.h5',
                                 verbose=1, monitor='acc', save_best_only=True)
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(images, labels, epochs=100, batch_size=32, callbacks=[checkpoint])

    # Save the trained model
    model.save('trained_payload_net.h5')

    # Option: Evaluate the model
    # result = model.evaluate(images, labels)
    # print('Result: {}'.format(result))
