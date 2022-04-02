from models import baseline
from other import load_dataset


def main():
    df_adu, df_text = load_dataset()
    baseline_2(df_adu)
    # class_distribution(df_adu)
    # print("Baseline")
    # test_different_features_sizes(df_adu.copy(), df_text.copy())
    # baseline(df_adu.copy(), df_text.copy())
    print("----------")

    #for algorithm in algorithms:
        #df_adu, df_text = load_dataset()
        #print(algorithm)
        #baseline(df_adu, df_text, algorithm=algorithm)
        #print('---------------')

    df_adu, df_text = load_dataset()
    baseline(df_adu, df_text, algorithm='naive_bayes')

    # print(df_adu.head())
    # print(df_text.head())

    print("One Hot Encoding")
    test_1_hot_vector(df_adu.copy(), df_text.copy())
    print("----------")
    # for algorithm in algorithms:
    #  print(algorithm)
    # baseline(df_adu, df_text, algorithm=algorithm)
    # print('---------------')

# print(df_adu.head())
# print(df_text.head())
"""

if __name__ == "__main__":
    main()
