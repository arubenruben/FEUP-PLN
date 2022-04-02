from src.exploratory_analyses import class_distribution, outlier_detection
from src.models import baseline_2
from src.other import load_dataset


def main():
    df_adu, df_text = load_dataset()
    # baseline_2(df_adu)

    outlier_detection(df_adu)

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

if __name__ == "__main__":
    main()
