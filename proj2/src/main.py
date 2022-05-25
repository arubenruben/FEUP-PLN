from transformers import AutoTokenizer, AutoModelForPreTraining
from transformers import DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer

from data_loading import load_dataset, normalize_dataset


def task_1():
    df_adu, _ = load_dataset()

    normalize_dataset(df_adu)

    print(df_adu)

    """
    model = AutoModelForPreTraining.from_pretrained('neuralmind/bert-large-portuguese-cased')
    tokenizer = AutoTokenizer.from_pretrained('neuralmind/bert-large-portuguese-cased', do_lower_case=False)

    tokenized_dataset = []

    for index, row in df_adu.iterrows():
        print(preprocess_function(tokenizer, row['tokens']))


    tokenized_imdb = imdb.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

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
        train_dataset=tokenized_imdb["train"],
        eval_dataset=tokenized_imdb["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    """


def preprocess_function(tokenizer, examples):
    return tokenizer(examples["text"], truncation=True)


if __name__ == '__main__':
    task_1()
