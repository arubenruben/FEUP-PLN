from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from constants import NUM_LABELS
from data_loading import load_dataset, normalize_dataset, split_train_test


def task_1():
    df_adu, _ = load_dataset()

    normalize_dataset(df_adu)

    train, test = split_train_test(df_adu)

    # model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)

    tokenized_dataset = {
        'train': [],
        'test': []
    }

    for index, row in train.iterrows():
        tokenized_dataset['train'].append(preprocess_function(tokenizer, row))

    for index, row in test.iterrows():
        tokenized_dataset['test'].append(preprocess_function(tokenizer, row))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=NUM_LABELS)

    training_args = TrainingArguments(
        output_dir="./results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    """
    """


def preprocess_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True)


if __name__ == '__main__':
    task_1()
