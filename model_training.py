import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict, load_metric
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
from sklearn.model_selection import train_test_split
import evaluate

def open_df():
    # Load the preprocessed dataset
    df = pd.read_csv('preprocessed_java_methods.csv')

    # Extract labels and prepare the data
    labels = df['identifier']
    data = df.drop('identifier', axis=1)

    # Splitting the dataset
    train_data, temp_test_data, train_labels, temp_test_labels = train_test_split(data, labels, test_size=10000,
                                                                                  random_state=42)
    test_data, val_data, test_labels, val_labels = train_test_split(temp_test_data, temp_test_labels, test_size=5000,
                                                                    random_state=42)

    # Creating Dataset objects
    train_dataset = Dataset.from_pandas(train_data.assign(labels=train_labels))
    test_dataset = Dataset.from_pandas(test_data.assign(labels=test_labels))
    val_dataset = Dataset.from_pandas(val_data.assign(labels=val_labels))

    # Creating a DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    return dataset_dict

dataset_dict = open_df()
# Tokenizer and model checkpoint
checkpoint = "t5-base"  # Replace with your model checkpoint
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# Setting the maximum input length for the tokenizer
max_input_length = 512

# List of columns to be removed from the dataset
columns_to_remove = ['formal_parameters', 'modifiers', 'block', 'type_identifier', 'file_path', 'full_code',
                     '__index_level_0__']


def preprocess_data(examples):
    # Tokenizing the full code sections of the examples
    inputs = examples["full_code"]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding='max_length')

    # Tokenizing the labels (method names)
    labels = tokenizer(examples["labels"], max_length=max_input_length, truncation=True, padding='max_length')

    # Setting the tokenized labels as the 'labels' field in the model inputs
    model_inputs["labels"] = labels["input_ids"]

    # The function returns the processed inputs suitable for the model
    return model_inputs


# Tokenizing datasets
tokenized_datasets = dataset_dict.map(preprocess_data, remove_columns=columns_to_remove)

# Selecting subsets for training, validation, and testing
train_dataset = tokenized_datasets["train"].select(range(100000))
validation_dataset = tokenized_datasets["validation"].select(range(4000))
test_dataset = tokenized_datasets["test"].select(range(4000))

# Model Name and Directory
model_name = "t5p-base-medium-function-name-generation"
model_dir = f"/models_run/{model_name}"

# Batch Size
TRAINER_BATCH_SIZE = 6

# Seq2SeqTrainingArguments Configuration
args = Seq2SeqTrainingArguments(
    model_dir,
    evaluation_strategy="steps",
    eval_steps=11000,  # Adjust based on how often you want to evaluate
    logging_strategy="steps",
    logging_steps=11000,  # Adjust based on how often you want to log
    save_strategy="steps",
    save_steps=11000,  # Saves less frequently
    learning_rate=4e-5,
    per_device_train_batch_size=TRAINER_BATCH_SIZE,  # Increase if your hardware allows
    per_device_eval_batch_size=TRAINER_BATCH_SIZE,  # Increase if your hardware allows
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=2,
    predict_with_generate=True,
    fp16=True,  # Enables mixed-precision training with CUDA-compatible devices
    load_best_model_at_end=True,
    metric_for_best_model="rouge1",
    report_to="tensorboard"
)


def compute_metrics(eval_preds):
    rouge_metric = evaluate.load("rouge")
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(pred) for pred in decoded_preds]
    decoded_labels = ["\n".join(label) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


# Function that returns an untrained model to be trained
def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to('cuda')
    return model


data_collator = DataCollatorForSeq2Seq(tokenizer)

# Initializing the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Execute the script
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("/model")
