from src.models import baseline_deleting_outliers, baseline
from src.other import load_dataset


def main():
    df_adu, df_text = load_dataset()
    # baseline_2(df_adu)

    for _ in range(5):
        print("Without Touching")
        baseline(df_adu.copy(), df_text.copy())
        print("----------")
        print("With No duplicates")
        baseline_deleting_outliers(df_adu.copy(), 'delete')

        print("----------")

        print("With Majority")
        baseline_deleting_outliers(df_adu.copy(), 'majority')


if __name__ == "__main__":
    main()
