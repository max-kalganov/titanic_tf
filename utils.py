from typing import Tuple

from ct import PATH_TO_TRAIN, SURVIVED
import pandas as pd
import numpy as np

from parser import Parser


def read_train(path: str = PATH_TO_TRAIN) -> pd.DataFrame:
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


def prepare_data() \
        -> Tuple[np.array, np.array, np.array, np.array]:
    full_dataset = read_train()
    df = Parser().parse(full_dataset)
    train_x, train_y, test_x, test_y = split_train_test(df)
    return train_x, train_y, test_x, test_y

