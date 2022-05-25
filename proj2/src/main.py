import torch
from torch.utils.data import Dataset
from transformers import AutoModelForMaskedLM
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers import AdamW
from constants import NUM_LABELS
from data_loading import load_dataset, split_train_test, load_data_for_masking
from evaluate import compute_metrics
from tqdm import tqdm  # for our progress bar

model_name = 'neuralmind/bert-base-portuguese-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=False)


class AduDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings.input_ids)


def task_1(model=None):
    df_adu, _ = load_dataset()

    dataset = split_train_test(df_adu)

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=NUM_LABELS)

    return train(model, tokenizer, tokenized_dataset)


def task_2():
    df_adu, df_text = load_dataset()

    dataset = load_data_for_masking(df_text)

    inputs = tokenizer(dataset, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    inputs['labels'] = inputs.input_ids.detach().clone()

    # create random array of floats with equal dimensions to input_ids tensor
    rand = torch.rand(inputs.input_ids.shape)
    # create mask array
    mask_arr = (rand < 0.15) * (inputs.input_ids != 101) * \
               (inputs.input_ids != 102) * (inputs.input_ids != 0)

    selection = []

    for i in range(inputs.input_ids.shape[0]):
        selection.append(
            torch.flatten(mask_arr[i].nonzero()).tolist()
        )

    for i in range(inputs.input_ids.shape[0]):
        inputs.input_ids[i, selection[i]] = 103

    dataset = AduDataset(inputs)
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # and move our model over to the selected device
    model.to(device)
    # activate training mode
    model.train()

    # initialize optimizer
    optim = AdamW(model.parameters(), lr=5e-5)

    epochs = 5

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # process
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
            # extract loss
            loss = outputs.loss
            # calculate loss for every parameter that needs grad update
            loss.backward()
            # update parameters
            optim.step()
            # print relevant info to progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    task_1(model)


def train(model, tokenizer, tokenized_dataset):
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
    trainer.save_model()


def preprocess_function(sample):
    return tokenizer(sample["tokens"], truncation=True)


if __name__ == '__main__':
    task_1()
