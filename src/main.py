from models import *
from other import load_dataset


def main():
    df_adu, df_text = load_dataset()
    
    #baseline_with_normalization(df_adu, df_text)
    #baseline(df_adu, df_text, algorithm="decision_tree")
    baseline(df_adu, df_text)

    #test_1_hot_vector(df_adu, df_text)
    #test_tf_idf(df_adu, df_text)
    print("---------------")


if __name__ == "__main__":
    main()
