from typing import Optional

import gin

from classifiers.base_classifier import BaseClassifier
from constants import LOG_DIR, MODEL_DIR
from utils import pr_rec_f1
import numpy as np
import tensorflow as tf


class CustomClassifier(BaseClassifier):
    def __init__(self, model_dir: Optional[str] = None, epochs_num: int = 200, patience: int = 10):
        super().__init__()
        self.model_dir = model_dir if model_dir is not None else MODEL_DIR
        self.patience = patience
        self.epochs_num = epochs_num

    @gin.configurable
    def _init_model(self, *args, **kwargs):
        """Initializes model"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(300, activation='tanh'),
            tf.keras.layers.Dropout(0.9),
            tf.keras.layers.Dense(500, activation='tanh'),
            tf.keras.layers.Dropout(0.9),
            tf.keras.layers.Dense(100, activation='tanh'),
            tf.keras.layers.Dropout(0.9),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy',
                               tf.keras.metrics.Recall(thresholds=0.5),
                               tf.keras.metrics.Precision(thresholds=0.5)])
        return model

    def classify(self, test_data: np.ndarray):
        model = tf.keras.models.load_model(self.model_dir) if self.model is None else self.model
        return model.predict(test_data)

    def _lr_sheduler(self, epoch, lr):
        return lr if epoch < self.patience else lr * tf.math.exp(-0.1)

    def train_model(self, train_x, train_y, test_x, test_y):
        """Trains model using self.model"""
        print(f"{train_x.shape=}, {test_x.shape=}")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)
        lr_sheduler_callback = tf.keras.callbacks.LearningRateScheduler(self._lr_sheduler)
        early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=self.patience + 10)

        self.model.fit(train_x,
                       train_y,
                       epochs=self.epochs_num,
                       validation_data=(test_x, test_y),
                       callbacks=[tensorboard_callback,
                                  lr_sheduler_callback,
                                  early_stop_callback])

        print("evaluate:")
        self.model.evaluate(test_x, test_y, verbose=2)

        tp, tn, fp, fn, recall, precision, f1 = pr_rec_f1(train_y, self.classify(train_x))
        test_tp, test_tn, test_fp, test_fn, test_recall, test_precision, test_f1 = pr_rec_f1(test_y,
                                                                                             self.classify(test_x))
        print(f"Metrics for train: {tp=}, {tn=}, {fp=}, {fn=}, {recall=}, {precision=}, {f1=}")
        print(f"Metrics for test: "
              f"{test_tp=}, {test_tn=}, {test_fp=}, {test_fn=}, {test_recall=}, {test_precision=}, {test_f1=}")

    def save(self):
        self.model.save(self.model_dir, save_format='h5')
