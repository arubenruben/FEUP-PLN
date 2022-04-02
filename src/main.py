from src.exploratory_analyses import outlier_detection, deal_with_outliers
from src.models import baseline, baseline_deleting_outliers
from src.other import load_dataset, write_df_to_file


def main():
    df_adu, df_text = load_dataset()
    # baseline_2(df_adu)

    baseline_deleting_outliers(df_adu, 'majority')

    print("----------")


if __name__ == "__main__":
    main()
