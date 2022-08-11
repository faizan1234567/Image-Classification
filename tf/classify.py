import argparse
import os
from turtle import forward

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def read_args():
    '''get commandline args'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help="path to data dir")
    parser.add_argument('--batch', type=int,
                        help="batch size for data loading")
    parser.add_argument('--img', type=int,
                        help="resize to this image resolution")
    parser.add_argument('--epochs', type=int, help="number of epochs")
    parser.add_argument('--augment', action="store_true",
                        help="apply data augmentation")
    return parser.parse_args()


def load_data(dir, img_size, batch, augment=False):
    """load the dataset from root data directory
    args:
    dir: data dir
    """
    train_ds = None
    val_ds = None
    test_ds = None

    # set train and test dataset path here
    train_dir = os.path.join(dir, "seg_train", "seg_train")
    test_dir = os.path.join(dir, "seg_test", "seg_test")
    # load dataset
    if augment:
        # please add more data augmentation options if you wish
        train_datagen = ImageDataGenerator(
            rescale = 1/255,
            # featurewise_center=True,
            # featurewise_std_normalization=True,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=0.2)

        test_datagen = ImageDataGenerator(rescale=1/255)

        train_ds = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch,
            class_mode='categorical',
            subset='training')

        val_ds = train_datagen.flow_from_directory(
            train_dir,
            target_size=(img_size, img_size),
            batch_size=batch,
            class_mode='categorical',
            subset='validation'
        )
        test_ds = test_datagen.flow_from_directory(
            test_dir,
            target_size=(img_size, img_size),
            batch_size=batch,
            class_mode='categorical')
        train_datagen.fit(train_ds)
        test_datagen.fit(test_ds)
     

    else:

        train_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch)
        val_ds = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch)
        test_ds = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            seed=123,
            image_size=(img_size, img_size),
            batch_size=batch)
        for image_batch, label_batch in train_ds:
            print('image_batch shape: {}'.format(image_batch.shape))
            print(f'lable batch shape: {label_batch.shape}')
            break

    return train_ds, val_ds, test_ds


# define a deep learning model here
class classifer(Model):
    def __init__(self):
        super(classifer, self).__init__()
        self.conv1 = Conv2D(16, 3, 2)
        self.conv2 = Conv2D(32, 3, 2)
        self.conv3 = Conv2D(64, 3, 2)
        self.pool = MaxPooling2D(2)
        self.flatten = Flatten()
        self.d1 = Dense(128, activation="relu")
        self.d2 = Dense(10, activation="softmax")

    def call(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv3(self.conv2(x)))
        x = self.flatten(x)
        x = self.d1(x)
        output = self.d2(x)
        return output


@tf.function
def train_step(images, labels, model, loss_object, optimizer, train_loss, train_accuracy):
    with tf.GradientTape() as tape:
        # training=True is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)


@tf.function
def test_step(images, labels, model, loss_object, test_loss, test_accuracy):
    # training=False is only needed if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


def main():
    """main training code goes here"""
    args = read_args()
    EPOCHS = args.epochs
    # load the dataset into batches
    if args.augment:
        train_ds, val_ds, test_ds = load_data(
            args.data_dir, args.img, args.batch, augment=True)
    else:
        train_ds, val_ds, test_ds = load_data(
            args.data_dir, args.img, args.batch)

    # loss and accuracy initialization
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='train_accuracy')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
        name='test_accuracy')
    model = classifer()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False)
    optimizer = tf.keras.optimizers.Adam()

    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
       # reset loss stats

        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            train_step(images, labels, model, loss_object,
                       optimizer, train_loss, train_accuracy)

        for test_images, test_labels in test_ds:
            test_step(test_images, test_labels, model,
                      loss_object, test_loss, test_accuracy)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}')


if __name__ == "__main__":
    main()
