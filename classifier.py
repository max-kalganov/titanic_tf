from typing import Optional

from constants import LOG_DIR, MODEL_DIR, PASSENGER_ID
from utils import prepare_test_data, save_predictions, pr_rec_f1, split_train_test, prepare_train_data
import numpy as np
import tensorflow as tf


def train_model(train_x: np.array, train_y: np.array,
                test_x: np.array, test_y: np.array) -> tf.keras.models.Model:
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='sigmoid'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(30, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy',
                           tf.keras.metrics.Recall(thresholds=0.5),
                           tf.keras.metrics.Precision(thresholds=0.5)])
    model.fit(train_x,
              train_y,
              epochs=300,
              validation_data=(test_x, test_y),
              callbacks=[tensorboard_callback])

    print("evaluate:")
    model.evaluate(test_x, test_y, verbose=2)

    tp, tn, fp, fn, recall, precision, f1 = pr_rec_f1(train_y, classify(train_x, model))
    test_tp, test_tn, test_fp, test_fn, test_recall, test_precision, test_f1 = pr_rec_f1(test_y, classify(test_x, model))
    print(f"Metrics for train: {tp=}, {tn=}, {fp=}, {fn=}, {recall=}, {precision=}, {f1=}")
    print(f"Metrics for test: "
          f"{test_tp=}, {test_tn=}, {test_fp=}, {test_fn=}, {test_recall=}, {test_precision=}, {test_f1=}")

    return model


def train_and_save_model():
    print("Reading data..")
    df = prepare_train_data()
    train_x, train_y, test_x, test_y = split_train_test(df)
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]
    print(f"Shapes: {train_x.shape=}, {test_x.shape=}")

    print("Start classifying..")
    model = train_model(train_x, train_y, test_x, test_y)
    model.save(MODEL_DIR, save_format='h5')


def classify(test_data: np.ndarray, model: Optional = None):
    model = tf.keras.models.load_model(MODEL_DIR) if model is None else model
    return model.predict(test_data)


def classify_and_save():
    full_test_df, formated_test_df = prepare_test_data()
    pred = classify(formated_test_df.to_numpy())
    format_pred = (pred > 0.5).astype(int)
    save_predictions(full_test_df[PASSENGER_ID].to_numpy(), format_pred.reshape(-1))
