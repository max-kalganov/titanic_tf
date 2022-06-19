from abc import abstractmethod, ABC
import random
import gin

from constants import PASSENGER_ID, PATH_TO_PRED
from utils import prepare_train_data, split_train_test, prepare_test_data, save_predictions


class BaseClassifier(ABC):
    def __init__(self):
        self.model = self._init_model()

    def get_model(self, *args, **kwargs):
        """Returns model for processing"""
        return self.model if self.model is not None else self._init_model(*args, **kwargs)

    @gin.configurable
    @abstractmethod
    def _init_model(self, *args, **kwargs):
        """Initializes model"""

    @abstractmethod
    def train_model(self, train_x, train_y, test_x, test_y):
        """Trains model using self.model"""

    @abstractmethod
    def classify(self, data):
        """Predict data"""

    def train_and_save_model(self):
        print("Reading data..")
        df = prepare_train_data()
        train_x, train_y, test_x, test_y = split_train_test(df)
        assert train_x.shape[0] == train_y.shape[0]
        assert test_x.shape[0] == test_y.shape[0]
        print(f"Shapes: {train_x.shape=}, {test_x.shape=}")

        print("Start classifying..")
        self.train_model(train_x, train_y, test_x, test_y)


def train_and_save_model(classifier, do_classify: bool = False, output_path: str = PATH_TO_PRED):
    print("Reading data..")
    df = prepare_train_data()
    train_x, train_y, test_x, test_y = split_train_test(df)
    assert train_x.shape[0] == train_y.shape[0]
    assert test_x.shape[0] == test_y.shape[0]
    print(f"Shapes: {train_x.shape=}, {test_x.shape=}")

    print("Start classifying..")

    classifier.train_model(train_x, train_y, test_x, test_y)
    if 'save' in dir(classifier):
        classifier.save()

    if do_classify:
        print(f"Classifying !!! in output path = {output_path}")
        classify_and_save(classifier, output_path=output_path)


def classify_and_save(classifier, output_path: str = PATH_TO_PRED):
    random.seed(1)
    full_test_df, formated_test_df = prepare_test_data()
    pred = classifier.classify(formated_test_df.to_numpy())
    format_pred = (pred > 0.5).astype(int)
    save_predictions(full_test_df[PASSENGER_ID].to_numpy(), format_pred.reshape(-1), output_path)
