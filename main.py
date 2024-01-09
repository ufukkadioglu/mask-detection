import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from datetime import datetime
import asyncio

from keras.applications.efficientnet import EfficientNetB2
from keras.applications.resnet_v2 import ResNet101V2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.densenet import DenseNet169
from keras.applications.nasnet import NASNetMobile

data_path = "data"
dataset_12k_path = os.path.join(data_path, "face-mask-12k-images-dataset")


def log(transfer_model_class, message):
    if isinstance(transfer_model_class, str):
        logger_name = transfer_model_class
    else:
        logger_name = transfer_model_class.__name__

    print(f"[{datetime.now().strftime('%d.%m.%Y %H:%M:%S')}] [{logger_name}] {message}")


def get_model_path(transfer_model_class):
    trained_models_dir = "trained_models"
    current_mode_file = f"mask_model_{transfer_model_class.__name__}.h5"
    return os.path.join(trained_models_dir, current_mode_file)


def get_generators():
    train_dir = os.path.join(dataset_12k_path, "Train")
    test_dir = os.path.join(dataset_12k_path, "Test")
    val_dir = os.path.join(dataset_12k_path, "Validation")

    train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True, zoom_range=0.2, shear_range=0.2)
    train_generator = train_datagen.flow_from_directory(directory=train_dir, target_size=(128, 128),
                                                        class_mode='categorical', batch_size=32)

    val_datagen = ImageDataGenerator(rescale=1.0 / 255)
    val_generator = val_datagen.flow_from_directory(directory=val_dir, target_size=(128, 128),
                                                    class_mode='categorical', batch_size=32)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_generator = test_datagen.flow_from_directory(directory=test_dir, target_size=(128, 128),
                                                      class_mode='categorical', batch_size=32)

    return train_generator, val_generator, test_generator


def train_model(transfer_model_class):
    log(transfer_model_class, "Training model with")

    train_generator, val_generator, test_generator = get_generators()

    transfer_model = transfer_model_class(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    for layer in transfer_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(transfer_model)
    model.add(Flatten())
    model.add(Dense(2, activation='sigmoid'))
    model.summary()

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics="accuracy")

    model.fit(train_generator,
              steps_per_epoch=len(train_generator) // 32,
              epochs=20,
              validation_data=val_generator,
              validation_steps=(len(val_generator) // 32) or 1)

    model.save(get_model_path(transfer_model_class))

    return model


def model_trained(transfer_model_class):
    return os.path.isfile(get_model_path(transfer_model_class))


def get_model(transfer_model_class):
    log(transfer_model_class, "Loading saved model")
    return load_model(get_model_path(transfer_model_class))


def test_model(model, transfer_model_class, show_samples=False):
    log(transfer_model_class, "Testing model")

    face_model = cv2.CascadeClassifier(f'{data_path}/haarcascades/haarcascade_frontalface_default.xml')

    examples = [
        f'{data_path}/face-mask-detection/images/maksssksksss83.png',
        f'{data_path}/face-mask-detection/images/maksssksksss116.png',
        f'{data_path}/face-mask-detection/images/maksssksksss155.png',
        f'{data_path}/face-mask-detection/images/maksssksksss244.png',
    ]

    log(transfer_model_class, "Evaluating model")
    train_generator, val_generator, test_generator = get_generators()

    evaluation = model.evaluate(test_generator)
    evaluation_message = f"Evaluation on test data: {evaluation}"
    log(transfer_model_class, evaluation_message)

    if show_samples:
        plt.figure(figsize=(10, 10))

        plt.axis("off")
        plt.text(.01, .99, evaluation_message, ha='left', va='top', weight='bold')

        log(transfer_model_class, "Predicting samples")

        for img_num in range(4):
            img = cv2.imread(examples[img_num])
            img = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)

            faces = face_model.detectMultiScale(img, scaleFactor=1.1,
                                                minNeighbors=4)  # returns a list of (x,y,w,h) tuples

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

    return f"{transfer_model_class.__name__}: {evaluation}"


def run_model(transfer_model_class, force_retrain, show_samples):
    if not force_retrain and model_trained(transfer_model_class):
        model = get_model(transfer_model_class)
    else:
        model = train_model(transfer_model_class)

    return test_model(model, transfer_model_class, show_samples)


def run_app(transfer_models_to_use, force_retrain, show_samples):
    loop = asyncio.get_event_loop()
    processes = []

    for transfer_model_class in transfer_models_to_use:
        model_process = loop.run_in_executor(None, run_model, transfer_model_class, force_retrain, show_samples)
        processes.append(model_process)

    finished_tasks, _ = loop.run_until_complete(asyncio.wait(processes))

    print("Evaluation results:")
    for task in finished_tasks:
        print(task.result())


if __name__ == "__main__":
    log("Main Runner", "Script started")

    # model selection:
    # 1. Sort by Top-5 accuracy
    # 2. Ignore models with 100+ ms per inference step on cpu (to get result in reasonable time in a personal computer)
    # 3. Take top 5 which does not share the same architecture (i.e. best 1 is EfficientNetB2 and best 2 is
    # EfficientNetB1, ignore 2 since it has the same architecture with 1)
    # Resulting models at 09.01.2024:
    # EfficientNetB2, ResNet101V2, InceptionV3, DenseNet169, NASNetMobile

    run_app([EfficientNetB2, ResNet101V2, InceptionV3, DenseNet169, NASNetMobile],
            force_retrain=False,
            show_samples=False)

    log("Main Runner", "Script finished")
