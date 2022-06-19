from classifiers.base_classifier import train_and_save_model
from classifiers.random_forest import CustomRandomForestClassifier

if __name__ == '__main__':
    classifier = CustomRandomForestClassifier()
    train_and_save_model(classifier, do_classify=True, output_path="random_forest_classifier_pred.csv")
