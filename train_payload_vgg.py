from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from read_image_folder import read_image_folder
from keras.utils import np_utils
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint


def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
    for layer in base_model.layers[:]:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))  # 2 payload classes: 0 and 1
    model.summary()
    return model


if __name__ == '__main__':
    images, labels = read_image_folder('payload_dataset', image_size=(300, 300), color_mode='rgb')
    labels = np_utils.to_categorical(labels, num_classes=None, dtype='float32')

    # Create a VGG16 model with the pretrained weights
    model = create_model()
    checkpoint = ModelCheckpoint('./checkpoints_payload/vgg16_epoch-{epoch:03d}_val-acc-{val_acc:03f}.h5',
                                 verbose=1, monitor='val_acc', save_best_only=True)
    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model on the payload dataset for a few epochs
    model.fit(images, labels, validation_data=(images, labels), epochs=100, batch_size=64, callbacks=[checkpoint])

    # Save the trained model
    model.save('trained_payload_net.h5')

    # Option: Evaluate the model
    # result = model.evaluate(images, labels)
    # print('Result: {}'.format(result))
