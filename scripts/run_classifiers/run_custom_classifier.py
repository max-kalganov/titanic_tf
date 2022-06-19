from classifiers.base_classifier import train_and_save_model
from classifiers.custom_model import CustomClassifier

if __name__ == '__main__':
    classifier = CustomClassifier()
    train_and_save_model(classifier, do_classify=True, output_path="custom_classifier_pred.csv")
