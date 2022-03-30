from models import baseline
from other import load_dataset


def main():
    algorithms = ['naive_bayes', 'knn', 'logistic_regression', 'decision_tree', 'bagging', 'random_forest',
                  'neural_net']

    #for algorithm in algorithms:
        #df_adu, df_text = load_dataset()
        #print(algorithm)
        #baseline(df_adu, df_text, algorithm=algorithm)
        #print('---------------')

    df_adu, df_text = load_dataset()
    baseline(df_adu, df_text, algorithm='naive_bayes')

    # print(df_adu.head())
    # print(df_text.head())


if __name__ == "__main__":
    main()
