from transformers import AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoConfig
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from datasets import load_metric
from transformers import Trainer
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import numpy as np
import math


chunk_size = 128

NUM_LABELS = 5

model_name = "neuralmind/bert-large-portuguese-cased"

#model_name = "turing-usp/FinBertPTBR"

#model_name = "unicamp-dl/mMiniLM-L6-v2-mmarco-v2"

tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(example["tokens"], truncation=True, padding=True)


def tokenize_function_2(examples):
    result = tokenizer(examples["tokens"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def compute_metrics(eval_preds):
    metric = load_metric("accuracy")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

def task_1(model=None):

    df_adu, _ = load_dataset()

    dataset = split_train_test(df_adu)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
   
    training_args = TrainingArguments(
            "test-trainer", 
            evaluation_strategy="epoch",
            num_train_epochs=10,
            #fp16=True,
    )
    
    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS, output_attentions=False, output_hidden_states=False, ignore_mismatched_sizes=True)        
    else:
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = NUM_LABELS
        model = AutoModelForSequenceClassification.from_config(config)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        
    )

    trainer.train()

    eval_results = trainer.evaluate()

    print(eval_results)
    

def task_2():
    df_adu, df_text=load_dataset()

    model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    dataset = load_data_for_masking(df_text)

    tokenized_datasets = dataset.map(
        tokenize_function_2, batched=True, remove_columns=["tokens"]
    )


    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)


    #batch_size = 64
    # Show the training loss with every epoch
    #logging_steps = len(lm_datasets["train"]) // batch_size

    training_args = TrainingArguments(
        output_dir=f"{model_name}-finetuned-imdb",        
        evaluation_strategy="epoch",
        num_train_epochs=5,
        #fp16=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["test"],
        data_collator=data_collator,
    )


    #eval_results = trainer.evaluate()
    #print(f">>> Perplexity Before: {math.exp(eval_results['eval_loss']):.2f}")

    trainer.train()

    #eval_results = trainer.evaluate()
    #print(f">>> Perplexity After: {math.exp(eval_results['eval_loss']):.2f}")

    task_1(model)


task_1()


if __name__ == '__main__':
    task_1()
