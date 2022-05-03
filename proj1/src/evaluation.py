from sklearn.metrics import classification_report


def evaluate_results(y_pred, y_test, clf, X_test):
    print(classification_report(y_test, y_pred))

    # plot_roc_curve(clf, X_test, y_test)
    # plt.show()
