from utils import prepare_data
import numpy as np
import tensorflow as tf


def classify(train_x: np.array, train_y: np.array,
             test_x: np.array, test_y: np.array):
    num_params = train_x.shape[1]
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(num_params*2/3, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_params/3, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_params/10, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=500)

    print("evaluate:")
    model.evaluate(test_x, test_y, verbose=2)


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = prepare_data()
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]

    classify(train_x, train_y, test_x, test_y)
