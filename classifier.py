from ct import LOG_DIR, MODEL_DIR
from utils import prepare_train_data, prepare_test_data, save_predictions
import numpy as np
import tensorflow as tf


def train_model(train_x: np.array, train_y: np.array,
                test_x: np.array, test_y: np.array) -> tf.keras.models.Model:
    num_params = train_x.shape[1]
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR, histogram_freq=1)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_params, activation='relu'),
        tf.keras.layers.Dense(num_params, activation='relu'),
        tf.keras.layers.Dense(num_params, activation='relu'),
        tf.keras.layers.Dense(num_params, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_params, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_x,
              train_y,
              epochs=500,
              callbacks=[tensorboard_callback])

    print("evaluate:")
    model.evaluate(test_x, test_y, verbose=2)
    return model


def train_and_save_model():
    print("reading data..")
    train_x, train_y, test_x, test_y = prepare_train_data()
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]

    print("start classifying..")
    model = train_model(train_x, train_y, test_x, test_y)
    model.save(MODEL_DIR, save_format='h5')


def classify():
    model = tf.keras.models.load_model(MODEL_DIR)
    test_data = prepare_test_data()
    pred = model.predict(test_data)
    format_pred = (pred > 0.5).astype(int)
    save_predictions(format_pred.reshape(-1))


if __name__ == '__main__':
    classify()
