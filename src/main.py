from src.models import *
from src.other import load_dataset


def main():
    df_adu, df_text = load_dataset()

    for _ in range(5):
        model_annotator_explicit(df_adu.copy())
        print("---------------")


if __name__ == "__main__":
    main()
