from typing import Tuple

from constants import PATH_TO_TRAIN, PATH_TO_TEST, SURVIVED, PATH_TO_PRED, PASSENGER_ID
import pandas as pd
import numpy as np

from parser import Parser


def read_train(path: str = PATH_TO_TRAIN) -> pd.DataFrame:
    return pd.read_csv(path)


def read_test(path: str = PATH_TO_TEST) -> pd.DataFrame:
    return pd.read_csv(path)


def split_train_test(df: pd.DataFrame, test_size: float = 0.2)\
        -> Tuple[np.array, np.array, np.array, np.array]:
    # TODO: make smart split (choose not randomly, but equally from each group)
    assert test_size < 1, f"too big test_size = {test_size}"
    test = df.sample(int(test_size * len(df)))
    train = df.drop(test.index)
    assert len(test) + len(train) == len(df), f"wrong size test + train != df"
    return train.drop(SURVIVED, axis=1).to_numpy(), train[SURVIVED].to_numpy(),\
           test.drop(SURVIVED, axis=1).to_numpy(), test[SURVIVED].to_numpy()


def prepare_train_data() \
        -> Tuple[np.array, np.array, np.array, np.array]:
    full_dataset = read_train()
    df = Parser().parse(full_dataset)
    train_x, train_y, test_x, test_y = split_train_test(df)
    return train_x, train_y, test_x, test_y


def prepare_test_data() -> Tuple[np.array, np.array]:
    full_dataset = read_test()
    df = Parser().parse(full_dataset, test=True)
    return full_dataset[PASSENGER_ID].to_numpy(), df.to_numpy()


def save_predictions(pass_id: np.array, pred: np.array, path: str = PATH_TO_PRED):
    res_df = pd.DataFrame({PASSENGER_ID: pass_id, SURVIVED: pred})
    res_df.to_csv(path, index=False)


def custom_metric(ground_truth: np.array, predictions: np.array):
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    tp = sum(ground_truth & predictions)
    tn = sum((1 - ground_truth) & (1 - predictions))
    fp = sum(np.logical_xor(ground_truth, predictions) * predictions)
    fn = sum(np.logical_xor(ground_truth, predictions) * (1 - predictions))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * recall * precision / (recall + precision)
    return tp, tn, fp, fn, recall, precision, f1
