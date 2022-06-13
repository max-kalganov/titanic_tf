from typing import Tuple

from constants import PATH_TO_TRAIN, PATH_TO_TEST, SURVIVED, PATH_TO_PRED, PASSENGER_ID
import pandas as pd
import numpy as np
import gin

from formatter import DatasetFormatter


def read_train(path: str = PATH_TO_TRAIN) -> pd.DataFrame:
    return pd.read_csv(path)


def read_test(path: str = PATH_TO_TEST) -> pd.DataFrame:
    return pd.read_csv(path)


def get_x_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    assert SURVIVED in df.columns, f"labels are not found in dataset columns. {df.columns=}"
    df_cp = df.copy()
    return df_cp.drop(SURVIVED, axis=1).to_numpy(), df_cp[SURVIVED].to_numpy()


def split_train_test(df: pd.DataFrame, train_test_split: float = 0.8) -> Tuple[np.array, np.array, np.array, np.array]:
    # TODO: make smart split (choose not randomly, but equally from each group)
    assert train_test_split < 1, f"too big train_test_split = {train_test_split}"
    df_rand = df.sample(len(df))
    train_size = int(train_test_split * len(df))
    train = df_rand[:train_size]
    test = df_rand[train_size:]
    assert len(test) + len(train) == len(df), f"wrong size test + train != df"
    train_x, train_y = get_x_y(train)
    test_x, test_y = get_x_y(test)
    return train_x, train_y, test_x, test_y


def prepare_train_data() -> pd.DataFrame:
    full_dataset = read_train()
    return DatasetFormatter().format(full_dataset)


def prepare_test_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = read_test()
    return df, DatasetFormatter().format(df, test=True)


def save_predictions(pass_ids: np.array, preds: np.array, path: str = PATH_TO_PRED):
    res_df = pd.DataFrame({PASSENGER_ID: pass_ids, SURVIVED: preds})
    res_df.to_csv(path, index=False)


@gin.configurable
def pr_rec_f1(ground_truth: np.array, predictions: np.array, threshold: float = 0.5):
    predictions = (predictions > threshold).astype(int).reshape(-1)
    tp = sum(ground_truth & predictions)
    tn = sum((1 - ground_truth) & (1 - predictions))
    fp = sum(np.logical_xor(ground_truth, predictions) * predictions)
    fn = sum(np.logical_xor(ground_truth, predictions) * (1 - predictions))
    recall = tp / (tp + fn)
    precision = tp / (tp + fp)
    f1 = 2 * recall * precision / (recall + precision)
    return tp, tn, fp, fn, recall, precision, f1
