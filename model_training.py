import argparse
import numpy as np
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)
import evaluate
import torch
from model_evaluation import load_tokenized_datasets

# Constants
CHECKPOINT = "Salesforce/codet5p-220m"
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
TRAIN_DATASET_SIZE = 4000
VALIDATION_DATASET_SIZE = 4000
MODEL_DIR = "/models_run/t5p-base-medium-function-name-generation"
DEFAULT_TRAINER_BATCH_SIZE = 6

# Argument parsing
parser = argparse.ArgumentParser(description='Train a Seq2Seq model.')
parser.add_argument('--batch_size', type=int, default=DEFAULT_TRAINER_BATCH_SIZE,
                    help='Batch size for training and evaluation (default: %(default)s)')
args = parser.parse_args()

TRAINING_ARGUMENTS = {
    'evaluation_strategy': "steps",
    'eval_steps': 11000,  # Adjust based on how often you want to evaluate
    'logging_strategy': "steps",
    'logging_steps': 11000,  # Adjust based on how often you want to log
    'save_strategy': "steps",
    'save_steps': 11000,  # Saves less frequently
    'learning_rate': 4e-5,
    'per_device_train_batch_size': args.batch_size,  # Increase if your hardware allows
    'per_device_eval_batch_size': args.batch_size,  # Increase if your hardware allows
    'weight_decay': 0.01,
    'save_total_limit': 3,
    'num_train_epochs': 2,
    'predict_with_generate': True,
    'fp16': torch.cuda.is_available(),  # Enables mixed-precision training with CUDA-compatible devices
    'load_best_model_at_end': True,
    'metric_for_best_model': "rouge1",
    'report_to': "tensorboard"
}

# Load tokenized datasets
tokenized_datasets = load_tokenized_datasets('tokenized_datasets.xz')
train_dataset = tokenized_datasets["train"].select(range(TRAIN_DATASET_SIZE))
validation_dataset = tokenized_datasets["validation"].select(range(VALIDATION_DATASET_SIZE))


def compute_metrics(eval_preds):
    rouge_metric = evaluate.load("rouge")
    preds, labels = eval_preds

    # decode preds and labels
    labels = np.where(labels != -100, labels, TOKENIZER.pad_token_id)
    decoded_preds = TOKENIZER.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = TOKENIZER.batch_decode(labels, skip_special_tokens=True)

    # rougeLSum expects newline after each sentence
    decoded_preds = ["\n".join(pred) for pred in decoded_preds]
    decoded_labels = ["\n".join(label) for label in decoded_labels]

    result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return result


# Function that returns an untrained model to be trained
def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(CHECKPOINT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model


# Initializing the Seq2SeqTrainer
trainer = Seq2SeqTrainer(
    model_init=model_init,
    args=Seq2SeqTrainingArguments(MODEL_DIR, **TRAINING_ARGUMENTS),
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorForSeq2Seq(TOKENIZER),
    tokenizer=TOKENIZER,
    compute_metrics=compute_metrics
)

# Execute the script
if __name__ == "__main__":
    trainer.train()
    trainer.save_model("/model")
