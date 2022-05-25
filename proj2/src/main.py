import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer

from data_loading import load_dataset, split_train_test
from evaluate import evaluate


def task_1():
    df_adu, _ = load_dataset()

    dataset = split_train_test(df_adu, 1.0, 0.0)

    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=False)
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=5)

    y_pred = []
    y_test = []

    for index, elem in enumerate(dataset['test']):
        # print(f"Evaluating:{index + 1}/{len(dataset['test'])}")
        inputs = tokenizer(elem['tokens'], padding=True, truncation=True, return_tensors="pt")
        outputs = model(**inputs)

        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        y_pred.append(np.argmax(predictions.detach().numpy(), axis=-1))

        y_test.append(elem['label'])

        if index == 1000:
            break

    evaluate(y_test, y_pred)


def preprocess_function(tokenizer, sample):
    return tokenizer(sample["tokens"], padding=True, truncation=True, return_tensors='pt')


if __name__ == '__main__':
    task_1()
