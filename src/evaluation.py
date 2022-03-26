from sklearn.metrics import classification_report


def evaluate_results(y_pred, y_test):
    print(classification_report(y_test, y_pred))
