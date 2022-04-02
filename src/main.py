from src.models import *
from src.other import load_dataset


def main():
    #df_adu, df_text = load_dataset()
    model_for_each_annotator()
    print("---------------")


if __name__ == "__main__":
    main()
