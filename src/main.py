from src.exploratory_analyses import outlier_detection, deal_with_outliers
from src.models import baseline
from src.other import load_dataset, write_df_to_file


def main():
    df_adu, df_text = load_dataset()
    # baseline_2(df_adu)

    dict_collisions = outlier_detection(df_adu)

    deal_with_outliers(df_adu, dict_collisions, 'delete')

    write_df_to_file(df_adu)

    baseline(df_adu, df_text)

    print("----------")


if __name__ == "__main__":
    main()
