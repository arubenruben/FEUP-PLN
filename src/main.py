from src.exploratory_analyses import class_distribution
from src.models import baseline_2
from src.other import load_dataset


def main():
    df_adu, df_text = load_dataset()
    baseline_2(df_adu)
    # class_distribution(df_adu)
    # print("Baseline")
    # test_different_features_sizes(df_adu.copy(), df_text.copy())
    # baseline(df_adu.copy(), df_text.copy())
    print("----------")


"""
    print("TF-IDF")
    test_tf_idf(df_adu.copy(), df_text.copy())
    print("----------")

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
