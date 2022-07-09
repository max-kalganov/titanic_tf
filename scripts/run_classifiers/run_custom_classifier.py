import os

from classifiers.base_classifier import train_and_save_model
from classifiers.custom_model import CustomClassifier
from constants import PATH_TO_PRED_DIR

if __name__ == '__main__':
    classifier = CustomClassifier()
    train_and_save_model(classifier, do_classify=True,
                         output_path=os.path.join(PATH_TO_PRED_DIR, "custom_classifier_pred.csv"))
