import argparse
import re
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import lzma
import pickle
from metrics import evaluation_results

# Constants
DEFAULT_MODEL_DIR = 'model'
DEFAULT_BATCH_SIZE = 16
TEST_DATASET_SIZE = 4000


def clean_generated_text(text):
    # Remove non-alphabetic characters and replace with spaces
    cleaned_text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Split cleaned text into words
    words = cleaned_text.split()
    # Return the longest word (in case of multiple words) or an empty string if no words are present
    return max(words, key=len) if words else ""


def generate_predictions(model, tokenizer, test_dataset, batch_size, device):
    predictions = []

    for i in tqdm(range(0, len(test_dataset), batch_size)):
        # Extract a batch of input data
        batch = test_dataset[i:i + batch_size]
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']

        # Ensure input tensors are on the correct device (CPU or GPU)
        input_ids = torch.tensor(input_ids).to(device)
        attention_mask = torch.tensor(attention_mask).to(device)

        with torch.no_grad():
            # Generate predictions using the model
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Clean and decode the generated outputs
        decoded_outputs = [clean_generated_text(tokenizer.decode(output, skip_special_tokens=True)) for output in
                           outputs]

        # Check for empty predictions and handle them
        decoded_outputs = [output if output.strip() != "" else "<no_prediction>" for output in decoded_outputs]

        # Extend the list of predictions
        predictions.extend(decoded_outputs)

    return predictions


def load_model_and_tokenizer(model_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device


def load_tokenized_datasets(filename):
    with lzma.open(filename, 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate predictions and evaluate a Seq2Seq model.')
    parser.add_argument('--model_dir', type=str, default=DEFAULT_MODEL_DIR,
                        help='Directory containing the model files (default: {})'.format(DEFAULT_MODEL_DIR))
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE,
                        help='Batch size for prediction (default: {})'.format(DEFAULT_BATCH_SIZE))
    args = parser.parse_args()

    model, tokenizer, device = load_model_and_tokenizer(args.model_dir)
    tokenized_datasets = load_tokenized_datasets('tokenized_datasets.xz')
    test_dataset = tokenized_datasets["test"].select(range(TEST_DATASET_SIZE))
    predictions = generate_predictions(model, tokenizer, test_dataset, args.batch_size, device)
    labels = [tokenizer.decode(label_ids, skip_special_tokens=True) for label_ids in test_dataset["labels"]]

    evaluation_results(labels, predictions)
