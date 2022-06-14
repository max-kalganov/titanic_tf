import gin
import pandas as pd

from constants import VIS_CONFIG_PATH
from utils import read_train, read_test
from visualization import visualize_dataset


@gin.configurable
def get_data(train: bool = True) -> pd.DataFrame:
    return read_train() if train else read_test()


if __name__ == '__main__':
    gin.parse_config_file(VIS_CONFIG_PATH)
    full_dataset = get_data()
    visualize_dataset(full_dataset)
