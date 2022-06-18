import gin
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from classifiers.base_classifier import BaseClassifier
from utils import pr_rec_f1


class CustomRandomForestClassifier(BaseClassifier):
    @gin.configurable
    def _init_model(self, *args, **kwargs):
        """Initializes model"""
        return RandomForestClassifier(n_estimators=1000,
                                      max_depth=10,
                                      min_samples_split=6,
                                      min_samples_leaf=2,
                                      random_state=0)

    def train_model(self, train_x, train_y, test_x, test_y):
        """Trains model using self.model"""
        self.model.fit(train_x, train_y)
        tp, tn, fp, fn, recall, precision, f1 = pr_rec_f1(train_y, self.model.predict(train_x))
        test_tp, test_tn, test_fp, test_fn, test_recall, test_precision, test_f1 = pr_rec_f1(test_y,
                                                                                             self.model.predict(test_x))
        print(f"Metrics for train: {tp=}, {tn=}, {fp=}, {fn=}, {recall=}, {precision=}, {f1=}")
        print(f"Metrics for test: "
              f"{test_tp=}, {test_tn=}, {test_fp=}, {test_fn=}, {test_recall=}, {test_precision=}, {test_f1=}")

    def classify(self, data):
        return self.model.predict(data)


def train_random_forest(train_x: np.array, train_y: np.array,
                        test_x: np.array, test_y: np.array):
    for i in range(1, 100):
        print(f"{i=}")

    return model
