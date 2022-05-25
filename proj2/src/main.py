import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, TrainingArguments, DataCollatorWithPadding, Trainer
from transformers import AutoTokenizer

from data_loading import load_dataset, split_train_test
from evaluate import evaluate, compute_metrics
from constants import NUM_LABELS


def task_1():
    model_name = 'neuralmind/bert-base-portuguese-cased'

    df_adu, _ = load_dataset()

    dataset = split_train_test(df_adu)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)

    tokenized_dataset = dataset.map(lambda x: preprocess_function(tokenizer, x), batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",  # run validation at the end of each epoch
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()
    trainer.evaluate()
    trainer.predict(test_dataset=tokenized_dataset["test"])


def preprocess_function(tokenizer, sample):
    return tokenizer(sample["tokens"], padding=True, truncation=True, return_tensors='pt')


if __name__ == '__main__':
    task_1()
