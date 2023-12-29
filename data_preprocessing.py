import pandas as pd
import lzma
import ast
import pickle
import re
import argparse
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

# Constants
CHECKPOINT = "Salesforce/codet5p-220m"
MAX_INPUT_LENGTH = 512
TOKENIZER = AutoTokenizer.from_pretrained(CHECKPOINT)
COLUMNS_TO_REMOVE = ["boolean_type", "floating_point_type", "generic_type", "scoped_type_identifier", "integral_type",
                     "void_type"]
COLUMNS_TO_REMOVE_TOK = ['formal_parameters', 'modifiers', 'block', 'type_identifier', 'file_path', 'full_code',
                         '__index_level_0__']
TEST_SIZE_SPLIT_1 = 10000
TEST_SIZE_SPLIT_2 = 5000
RANDOM_STATE = 42


def read_csv_check_index(file_path):
    # Read the first row to check the column names
    first_row = pd.read_csv(file_path, nrows=1)

    # Check if the first column is 'Unnamed: 0'
    if first_row.columns[0] == 'Unnamed: 0':
        # Read the entire file with the first column as index
        df = pd.read_csv(file_path, index_col=0)
    else:
        # Read the entire file normally
        df = pd.read_csv(file_path)

    return df


def tokenize_data(examples):
    # Tokenizing the full code sections of the examples
    inputs = examples["full_code"]
    model_inputs = TOKENIZER(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')

    # Tokenizing the labels (method names)
    labels = TOKENIZER(examples["labels"], max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')

    # Setting the tokenized labels as the 'labels' field in the model inputs
    model_inputs["labels"] = labels["input_ids"]

    # The function returns the processed inputs suitable for the model
    return model_inputs


def find_type(row):
    # Checking for non-null primitive types in the specified columns
    columns = ["boolean_type", "floating_point_type", "generic_type", "scoped_type_identifier", "integral_type",
               "void_type"]
    for col in columns:
        if pd.notna(row[col]):
            # Return the first non-null primitive type found
            return row[col]
    # Return None if no non-null primitive types are found
    return None


def check_not_empty(row):
    # Check if "block" is not empty '{}' and not null, unless the method is abstract
    # Abstract methods are allowed to have empty or null blocks
    if (row['block'] == '{}' or row['block'] == "{\n }" or pd.isna(row['block'])) and 'abstract' not in row[
        'modifiers']:
        return False

    # Ensure that "type_identifier" is not null or empty
    # This field is crucial except for constructors and lambda expressions
    if pd.isna(row['type_identifier']) or row['type_identifier'] == '':
        return False

    # Ensure that "identifier" (method name) is not null or empty
    # Every method should have a name, though rare exceptions might occur
    if pd.isna(row['identifier']) or row['identifier'] == '':
        return False

    # Ensure that "modifiers" is not an empty list or null
    # Modifiers are important for understanding the method's properties
    if row['modifiers'] == '[]' or row['modifiers'] == [] or pd.isna(row['modifiers']):
        return False

    # If all checks are passed, the row is valid
    return True


def full_code_feature(row):
    identifier = "<extra_id_0>"
    type_id = row['type_identifier']
    params = row['formal_parameters']

    # Converting the string representation of modifiers list back to a list data type
    modifiers_list = ast.literal_eval(row['modifiers'])
    decorators = [mod for mod in modifiers_list if mod.startswith('@')]
    other_mods = [mod for mod in modifiers_list if not mod.startswith('@')]

    # Formatting the decorators to appear on separate lines
    formatted_decorators = '\n'.join(decorators).replace("'", "")
    # Formatting other modifiers to appear on the same line
    formatted_other_mods = ' '.join(other_mods).replace("'", "").replace(",", "")

    # Cleaning the block of code from comments
    block = row['block']
    block = re.sub(r'/\*.*?\*/', '', str(block), flags=re.DOTALL)  # Removing block comments
    block = re.sub(r'//.*?\n', '', block)  # Removing line comments

    # Constructing the full method code with new lines for readability
    return f"{formatted_decorators}\n{formatted_other_mods} {type_id} {identifier}{params} {block}"


def preprocess_data(file_path):
    df = read_csv_check_index(file_path)

    # Update 'type_identifier' column
    df['type_identifier'] = df.apply(
        lambda row: find_type(row) if pd.isna(row['type_identifier']) else row['type_identifier'], axis=1)

    # Removing the specified columns from the DataFrame
    df.drop(columns=COLUMNS_TO_REMOVE, inplace=True)

    df = df[df.apply(check_not_empty, axis=1)]
    df['full_code'] = df.apply(full_code_feature, axis=1)

    labels = df['identifier']
    data = df.drop('identifier', axis=1)

    train_data, temp_test_data, train_labels, temp_test_labels = train_test_split(
        data, labels, test_size=TEST_SIZE_SPLIT_1, random_state=RANDOM_STATE)
    test_data, val_data, test_labels, val_labels = train_test_split(
        temp_test_data, temp_test_labels, test_size=TEST_SIZE_SPLIT_2, random_state=RANDOM_STATE)

    # Creating datasets
    dataset_dict = DatasetDict({
        'train': Dataset.from_pandas(train_data.assign(labels=train_labels)),
        'validation': Dataset.from_pandas(val_data.assign(labels=val_labels)),
        'test': Dataset.from_pandas(test_data.assign(labels=test_labels))
    })

    tokenized_datasets = dataset_dict.map(tokenize_data, remove_columns=COLUMNS_TO_REMOVE_TOK)

    # Saving the tokenized dataset using LZMA compression
    with lzma.open('tokenized_datasets.xz', 'wb') as f:
        pickle.dump(tokenized_datasets, f)


# Execute the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a data file.')
    parser.add_argument('--file', '-f', type=str, default='java_methods.csv',
                        help='File containing Java Methods data (default: java_methods.csv)')
    args = parser.parse_args()
    preprocess_data('java_methods.csv')
