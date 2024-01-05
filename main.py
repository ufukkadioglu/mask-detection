import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.applications.vgg19 import VGG19
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

model_path = "mask_net.h5"
data_path = "data"
dataset_12k_path = os.path.join(data_path, "face-mask-12k-images-dataset")


def get_generators():
    train_dir = os.path.join(dataset_12k_path, "Train")
    test_dir = os.path.join(dataset_12k_path, "Test")
    val_dir = os.path.join(dataset_12k_path, "Validation")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
    train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
                                                        class_mode='categorical', batch_size=32)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = train_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128),
                                                      class_mode='categorical', batch_size=32)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = train_datagen.flow_from_directory(directory=test_dir, target_size=(128, 128),
                                                       class_mode='categorical', batch_size=32)

    return train_generator, val_generator, test_generator


def train_model():
    print("Training model")

    train_generator, val_generator, test_generator = get_generators()

    vgg19 = VGG19(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    for layer in vgg19.layers:
        layer.trainable = False

    model = Sequential()
    model.add(vgg19)
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator) // 32,
                        epochs=20,
                        validation_data=val_generator,
                        validation_steps=(len(val_generator) // 32) or 1)  # ufuk

    model.save(model_path)


def model_trained():
    return os.path.isfile(model_path)


def get_model():
    print("Loading saved model")
    return load_model(model_path)


def test_model(model):
    print("Testing model")

    face_model = cv2.CascadeClassifier(f'{data_path}/haarcascades/haarcascade_frontalface_default.xml')

    plt.figure(figsize=(10, 10))

    examples = [
        f'{data_path}/face-mask-detection/images/maksssksksss83.png',
        f'{data_path}/face-mask-detection/images/maksssksksss116.png',
        f'{data_path}/face-mask-detection/images/maksssksksss155.png',
        f'{data_path}/face-mask-detection/images/maksssksksss244.png',
    ]

    print("Evaluating model")
    train_generator, val_generator, test_generator = get_generators()

    plt.axis("off")
    plt.text(.01, .99, f"Evaluation on test data: {model.evaluate_generator(test_generator)}", ha='left', va='top',
             weight='bold')

    print("Predicting samples")

    for img_num in range(4):
        img = cv2.imread(examples[img_num])
        img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

        faces = face_model.detectMultiScale(img, scaleFactor=1.1, minNeighbors=4)  # returns a list of (x,y,w,h) tuples

        new_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # colored output image

        mask_label = {0: 'MASK', 1: 'NO MASK'}
        dist_label = {0: (0, 255, 0), 1: (255, 0, 0)}

        # plotting
        for i in range(len(faces)):
            (x, y, w, h) = faces[i]
            crop = new_img[y:y + h, x:x + w]
            crop = cv2.resize(crop, (128, 128))
            crop = np.reshape(crop, [1, 128, 128, 3]) / 255.0
            mask_result = model.predict(crop)

            cv2.putText(new_img,
                        mask_label[mask_result.argmax()],
                        (x, y + h + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        dist_label[mask_result.argmax()],
                        2)

            cv2.rectangle(new_img,
                          (x, y),
                          (x + w, y + h),
                          dist_label[mask_result.argmax()],
                          1)

        plt.subplot(2, 2, img_num + 1)
        plt.axis("off")
        plt.imshow(new_img)

    plt.show()


def run_app():
    if not model_trained():
        train_model()

    model = get_model()
    test_model(model)


if __name__ == "__main__":
    print("Script started")
    run_app()
    print("Script finished")
