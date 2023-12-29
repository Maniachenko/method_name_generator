# -*- coding: utf-8 -*-

# Method Name Generation with CodeT5+: A Case Study on IntelliJ Community

This project explores enhancing code generation by predicting method names using CodeT5+, focusing on a large open-source project, IntelliJ Community. The process starts with extracting all methods from the project, forming a comprehensive dataset. Initially, the project tests the unfinetuned CodeT5+ model on this data to assess its baseline performance in predicting method names.

Subsequently, the model undergoes fine-tuning with the specific dataset, aimed at improving its accuracy in method name prediction tailored to the Java code in IntelliJ Community. A comparative analysis follows, evaluating the model's performance before and after fine-tuning, with an emphasis on quality improvements.

The project concludes by summarizing the findings and providing insights into the effectiveness of method name prediction as a method for enhancing code generation, especially in large open-source projects. It showcases the practical application of CodeT5+ in software development and explores machine learning's potential in code generation.

## Istallation And Imports
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install torch
# %pip install git+https://github.com/huggingface/transformers
# %pip install datasets accelerate huggingface_hub
# %pip install tree-sitter
# %pip install GitPython
# %pip install rouge_score nltk
# %pip install tensorboard
# %pip install spacy
# %pip install Levenshtein
# %pip install Rouge
# %pip install evaluate

import ast
import lzma
import math
import os
import pickle
import re
import requests
from collections import Counter, defaultdict
from io import BytesIO
from statistics import mean
from zipfile import ZipFile

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, load_metric
from git import Repo
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForSeq2Seq,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments, T5ForConditionalGeneration)
from tree_sitter import Language, Parser

# External libraries for specific evaluations
import evaluate
import Levenshtein
from rouge import Rouge

# from google.colab import drive
# drive.mount('/content/drive')

# # Uncomment the above lines if you are using Google Colab.
# # These lines of code are used to mount your Google Drive,
# # allowing you to access files from your Drive in the Colab environment.
# # rewrite all directories then

"""## CodeT5+: description and first try

### Model description

CodeT5+ is an advanced open code language model with versatile modes (encoder-only, decoder-only, encoder-decoder) for various code-related tasks. Detailed in the paper "CodeT5+: Open Code Large Language Models for Code Understanding and Generation," it surpasses its predecessor, CodeT5, by incorporating multiple pretraining tasks like span denoising and contrastive learning. It leverages pre-existing models like CodeGen for efficient scaling (2B, 6B, 16B sizes) and features a "shallow encoder and deep decoder" design. The model is further refined through instruction-tuning for natural language alignment, as per the Code Alpaca methodology.

### Very First Try

The code snippet under involves selecting the "Salesforce/codet5p-220m" model checkpoint and initializing the `AutoTokenizer` from Hugging Face's Transformers library. `AutoTokenizer` is versatile, automatically identifying and loading appropriate tokenizers for various models like BERT and GPT. It ensures consistency with pre-trained models and manages special tokens and model-specific rules efficiently. The tokenization process involves creating token IDs and attention masks, which are crucial for the model's understanding and focus. The code then uses `T5ForConditionalGeneration` for tasks like text generation, differing from `transformers.T5Model`, which outputs raw hidden states. Finally, the model processes a Java function sample, demonstrating its code understanding and generation capabilities.
"""

# Ensure GPU is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "Salesforce/codet5p-220m"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = T5ForConditionalGeneration.from_pretrained(checkpoint)

# Move model to the device (GPU or CPU)
model.to(device)

java_function = "public static int sumTwoNumbers(int a, int b){"

# Encoding and moving the tensor to the same device as the model
encoding = tokenizer(java_function, return_tensors="pt").to(device)
encoding["decoder_input_ids"] = encoding["input_ids"].clone()

# Generate the output
outputs = model.generate(**encoding, max_length=750)

# Decode and print the output
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

"""Given the output generated by the model, which deviates significantly from the expected Java syntax and instead resembles a C-like language, it suggests a need for a more targeted approach, potentially using models with a larger number of parameters (770 million, 2 billion, etc., depending on computational resources). To address this, the next step involves collecting and preparing specific data that aligns more closely with our intended use case, which in this context is Java programming. This data preparation is crucial for fine-tuning the model to better suit our requirements for generating Java code.

## Data: Collection and Preprocessing

The first step is to clone the repository for the Java parser, 'tree-sitter-java', which is essential for analyzing Java source code.
"""

# Cloning the 'tree-sitter-java' repository, essential for parsing Java source code.
!git clone https://github.com/tree-sitter/tree-sitter-java.git

"""The next phase is to build the Java language parser. This is achieved by using the Language.build_library function to compile the parser, and then initializing it with Language and Parser objects from the Tree-sitter library. This setup is necessary for parsing Java files effectively."""

# Load Java language parser
Language.build_library('build/my-languages.so', ['tree-sitter-java'])
JAVA_LANGUAGE = Language('build/my-languages.so', 'java')
parser = Parser()
parser.set_language(JAVA_LANGUAGE)

"""Here's a concise explanation of the node types for the method parser:



*   **'identifier'**: The method's name, used for calling it in the code.
*   **'formal_parameters'**: Arguments accepted by the method, specified in parentheses next to the method name.
*   **'modifier'**: Keywords and annotations in Java that define method access levels and behaviors (e.g., public, private, @Override). Depending on the annotations present, this can be categorized as either 'modifiers' or 'modifier'.
*   **'block'**: The method's code, enclosed in curly braces {}.
*   **'type_identifier'**: The return type of the method, such as int, String, boolean.

For primitive return types (instead of type identifier):
*   **"boolean_type"**: Identifies boolean return types or parameters.
*   **"floating_point_type"**: For floating-point numbers like float or double.
*   **"generic_type"**: Used for methods involving generic types.
*   **"scoped_type_identifier"**: Types with a scope, like inner classes or enums.
*   **"integral_type"**: For integral types like int, long, byte.
*   **"void_type"**: Indicates a void return type, with no value returned.

The core of the code under involves two primary functions: extract_methods and parse_java_file.



*   **extract_methods Function**: It recursively traverses the syntax tree of Java code, extracting information about each method_declaration. This includes identifier, formal_parameters, modifiers, block, type_identifier, and various primitive types. The function processes each node, capturing details in a structured method_info dictionary.
*   **parse_java_file Function**: This function reads a Java file in different encodings, parses it to create a syntax tree, and uses extract_methods to extract method details. Each method's information includes the file path for reference.

**NOTE:** In the event of an error, such as one related to C++ build tools, installing the latest version of Microsoft C++ Build Tools is suggested. This is often required for compiling language parsers and tools with native extensions.
"""

# Function to recursively traverse the syntax tree
def extract_methods(node):
    methods = []
    if node.type == 'method_declaration':
        # Initializing a dictionary to store various method properties
        method_info = {
            'identifier': '',
            'formal_parameters': '',
            'modifiers': [],
            'block': '',
            'type_identifier': '',
            "boolean_type": '',
            "floating_point_type": '',
            "generic_type": '',
            "scoped_type_identifier": '',
            "integral_type": '',
            "void_type": ''
        }

        for child in node.children:
            # Checking and extracting modifiers
            if child.type == 'modifiers' or child.type == 'modifier':
                # Extracting individual modifiers
                for mod in child.children:
                    method_info['modifiers'].append(mod.text.decode('utf8'))
            elif child.type in ['block', 'formal_parameters', "generic_type", 'type_identifier',
                                "scoped_type_identifier",
                                'identifier', "boolean_type", "floating_point_type", "integral_type", "void_type"]:
                # Storing other method details
                method_info[child.type] = child.text.decode('utf8')

        # Adding the collected method information to the methods list
        methods.append(method_info)

    # Recursively traversing child nodes to extract methods
    for child in node.children:
        methods.extend(extract_methods(child))

    return methods


# Function to parse a Java file
def parse_java_file(file_path):
    # Trying different encodings to read the file
    encoding_to_try = ['utf-8', 'ISO-8859-1', 'windows-1252', 'utf-16']
    for encoding in encoding_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                source_code = file.read()
            # Parsing the Java source code to a syntax tree
            tree = parser.parse(bytes(source_code, "utf8"))
            root_node = tree.root_node
            # Extracting methods from the root node of the syntax tree
            methods = extract_methods(root_node)

            # Associating each extracted method with its source file path
            for method in methods:
                method['file_path'] = file_path

            return methods
        except UnicodeDecodeError:
            # Trying next encoding if current one fails
            continue
    else:
        # Raising an exception if all encoding attempts fail
        raise Exception("Failed to decode file with tried encodings")

"""In the following cell, I am cloning the IntelliJ Community repository from GitHub. This repository contains the source code of the IntelliJ IDEA, an integrated development environment written in Java. This repository is a large and comprehensive codebase that is ideal for my task.

"""

# Cloning the IntelliJ Community repository for parsing and analysis.
!git clone https://github.com/JetBrains/intellij-community.git

"""The script systematically traverses the IntelliJ Community project directory, parsing every Java file found. Data extracted from each file is compiled into a DataFrame, ultimately saved as `java_methods.csv`. This CSV file forms a structured dataset of Java method information from the IntelliJ Community codebase, primed for subsequent analysis or machine learning applications."""

# Collecting all .java files from the IntelliJ Community repository
java_files = []
for dir_path, dir_names, file_names in os.walk('/content/intellij-community'):
    # Adding paths of .java files to the list
    java_files.extend([os.path.join(dir_path, file_name) for file_name in file_names if file_name.endswith(".java")])

all_methods = []

# Processing each Java file
for file in tqdm(java_files, desc="Processing Java Files"):
    # Parsing each Java file to extract method information
    methods_list = parse_java_file(file)
    # Adding the extracted methods to the all_methods list
    all_methods.extend(methods_list)

# Create a DataFrame from the collected method information
# Columns correspond to various attributes of the methods
methods_df = pd.DataFrame(all_methods,
                          columns=['identifier', 'formal_parameters', 'modifiers', 'block', 'type_identifier',
                                   "boolean_type",
                                   "floating_point_type", "generic_type", "scoped_type_identifier", "integral_type",
                                   "void_type", 'file_path'])

# Save the DataFrame containing all method information to a CSV file
methods_df.to_csv('java_methods.csv')

import pandas as pd

# Loading the previously parsed Java methods dataset from a CSV file
# This step is efficient as it avoids reparsing the entire IntelliJ Community project every time (when parser part is commented)
df = pd.read_csv('java_methods.csv')

df.head()

"""Let's delete first column that represents indices."""

df = df.iloc[:, 1:]

"""For efficiency, we'll consolidate all non-null primitive type entries into the type_identifier column."""

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


# Updating the 'type_identifier' column in the DataFrame
# This step consolidates all non-null primitive types into the 'type_identifier' column for convenience
df['type_identifier'] = df.apply(
    lambda row: find_type(row) if pd.isna(row['type_identifier']) else row['type_identifier'], axis=1)

# Defining the columns to be removed from the DataFrame
columns_to_remove = ["boolean_type", "floating_point_type", "generic_type", "scoped_type_identifier", "integral_type",
                     "void_type"]

# Removing the specified columns from the DataFrame
df.drop(columns=columns_to_remove, inplace=True)

"""Some features in our dataset might have NaN values, each indicating different scenarios:

Method without Type Identifier:
*   **Constructor:** Constructors are unique methods without a return type. They initialize new objects, bear the same name as the class, and cannot have a return type.
*   **Lambda Expressions (Java 8+)**: Lambda expressions, introduced in Java 8, provide implementations for functional interfaces. They lack a traditional return type identifier, consisting only of parameters, an arrow token (->), and a body.

Method without Block of Code:

*   **Abstract Methods**: In abstract classes or interfaces, abstract methods are declared without an implementation, hence no code block.
*   **Native Methods**: These methods, marked with the native keyword, are implemented in a non-Java language (e.g., C or C++), and do not contain a Java code block.
*   **Default Methods in Interfaces (Java 8+)**: Default methods in interfaces may have an empty body, allowing optional overriding by implementing classes.

Let's retain abstract methods and attempt to predict their identifier using other features, excluding the block of code.

**NOTE**: In rare cases, some methods might appear without identifiers, possibly due to parser limitations. This requires further analysis and refined parsing techniques.
"""

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


# Applying the function to each row to filter the DataFrame
# Retains only the rows where check_not_empty returns True
df = df[df.apply(check_not_empty, axis=1)]

df.shape

"""Now, let's merge all parsed method components into complete method codes, replacing each method's name with ```<extra_id_0>```. `full_code` will serve as our input.

The token ```<extra_id_0>``` is used as a placeholder in the method code to replace the original method names. This is part of a common technique in sequence-to-sequence language models, especially in tasks like translation, summarization, and, in this case, code generation.
"""

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


# Applying the function to each row to generate the full method code
df['full_code'] = df.apply(full_code_feature, axis=1)

"""I am going to verify the correctness of the formatted code for the first 10 entries in the dataset."""

# Displaying the full method code for the first 10 instances in the DataFrame
for instance_index in range(10):
    print(df.iloc[instance_index, :].full_code)

"""The code appears to be correctly formatted. Next, I will focus on the following stages on Defining Metrics, Tokenizing Data, Evaluating the Unfinetuned Model.

## Unfinetuned Model Test: Metrics, Data Tokenization, Unfinetuned Model Evaluation

### Metrics

This task - generating of identifiers from method blocks, is a text-to-text generation challenge. It involves converting input text into a different output text form. For evaluation, several metrics are considered:
*   **Accuracy**: Assesses how accurately the generated text matches the expected output.
*   **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** Score: Often used in summarization, it emphasizes word recall, making it suitable for generating precise and meaningful method names.
*   **Levenshtein Distance (Edit Distance)**: This metric calculates the number of single-character edits needed to transform one word into another, helping to gauge the closeness of generated method names to the expected ones.
*   **Semantic Similarity**: This metric calculates the semantic similarity between the predicted method names and the actual labels.

These metrics provide a comprehensive framework for evaluating the effectiveness of the model in generating accurate and relevant method names.

#### Accuracy

We have functions to evaluate our model's prediction accuracy in generating method names, handling camelCase conventions:
*   `split_camel_case`: Splits camelCase names into lowercase subwords for comparison.
*   `calculate_matches`: Counts matching subwords between predictions and actual names, regardless of their order.
*   `calculate_accuracy`: Calculates overall accuracy based on the number of matching subwords.

These functions ensure accurate assessment even when predicted words are in a different order or case from the actual names (e.g., comparing 'methodName' with 'nameMethod').
"""

# Helper function to split camelCase into subwords
def split_camel_case(name):
    # Regular expression to identify subwords in camelCase
    matches = re.finditer('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)', name)
    # Returning the found subwords in lowercase
    return [m.group(0).lower() for m in matches]


def calculate_matches(labels, predictions):
    """Calculate the number of matching words."""
    total_matches = 0
    total_subwords = 0

    for label, prediction in zip(labels, predictions):
        # Splitting both label and prediction into subwords
        label_subwords = split_camel_case(label)
        prediction_subwords = split_camel_case(prediction)

        matches = 0
        for l_word in label_subwords:
            # Finding a matching word in the prediction
            matched_word = next((p_word for p_word in prediction_subwords if l_word == p_word), None)
            if matched_word:
                matches += 1
                prediction_subwords.remove(matched_word)  # Removing the matched word to prevent reuse

        # Accumulating total matches and subwords
        total_matches += matches
        total_subwords += len(label_subwords)

    return total_matches, total_subwords


def calculate_accuracy(labels, predictions):
    # Calculating matches and total subwords
    matches, total_subwords = calculate_matches(labels, predictions)
    # Calculating and returning accuracy
    accuracy = matches / total_subwords if total_subwords != 0 else 0
    return accuracy

"""#### ROUGE

ROUGE-N recall scor is essential in evaluating the performance of our model in generating method names. ROUGE-N measures the overlap of N-grams between generated and reference names. This metric is crucial for assessing how well the generated names align with the reference names in terms of content and structure.

The function calculate_rouge_n uses the already mentioned `split_camel_case` method to split camelCase names into subwords and create N-grams, which are then compared to calculate the recall.
"""

def calculate_rouge_n(generated_names, reference_names, n=1):
    # Function to generate n-grams from a list of words
    def ngrams(words, n):
        # Creating n-grams using list comprehension and zip
        return zip(*[words[i:] for i in range(n)])

    scores = []  # List to store individual recall scores for each name pair

    # Looping through each pair of generated and reference names
    for gen_name, ref_name in zip(generated_names, reference_names):
        # Splitting camelCase names into words and creating n-grams
        gen_ngrams = Counter(ngrams(split_camel_case(gen_name), n))
        ref_ngrams = Counter(ngrams(split_camel_case(ref_name), n))

        # Calculating overlap of n-grams between generated and reference names
        overlap = sum((gen_ngrams & ref_ngrams).values())
        total_ref_ngrams = sum(ref_ngrams.values())  # Total n-grams in reference name

        # Calculating recall as the ratio of overlap to total reference n-grams
        recall = overlap / total_ref_ngrams if total_ref_ngrams > 0 else 0
        scores.append(recall)  # Adding recall score to the list

    # Calculating the average recall score across all name pairs
    average_recall = np.mean(scores) if scores else 0
    return average_recall

"""#### Levenshtein Distance

Levenshtein distance measures the number of single-character edits (insertions, deletions, or substitutions) required to change one word into another. It calculates the average Levenshtein distance between lists of labels and predictions.
"""

# Levenshtein Distance
def calculate_levenshtein_distance(labels, predictions):
    # Calculating Levenshtein distance for each pair of label and prediction
    distances = [Levenshtein.distance(ref, pred) for ref, pred in zip(labels, predictions)]

    # Returning the average distance
    return np.mean(distances)

"""#### Semantic Similarity

By employing semantic analysis, this function provides a nuanced measure of how well the generated names capture the conceptual essence of the reference names, going beyond mere syntactic comparison.

Each method name is split from camelCase. We then use an NLP model (like spaCy) to calculate the semantic similarity between each pair of generated and reference names. This similarity score reflects how close the generated name is to the reference name in terms of their meaning. Finally, the function computes and returns the average similarity score across all name pairs.

**NOTE:** Before using spaCy for semantic similarity, ensure the relevant NLP model (e.g., en_core_web_md) is downloaded:
"""

!python -m spacy download en_core_web_md

def calculate_similarity(labels, predictions):
    # Load spaCy model
    nlp = spacy.load("en_core_web_md")
    # Check if labels and predictions are of the same length
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must be of the same length.")

    similarities = []  # List to store similarity scores

    # Iterate over each label-prediction pair
    for ref, pred in zip(labels, predictions):
        # Convert camelCase names to space-separated words
        ref = ' '.join(split_camel_case(ref))
        pred = ' '.join(split_camel_case(pred))

        # Calculate semantic similarity using a predefined NLP model
        # nlp object is assumed to be a preloaded NLP model like spaCy
        similarity = nlp(ref).similarity(nlp(pred))
        similarities.append(similarity)

    # Return the average similarity score, or 0 if no similarities were calculated
    return np.mean(similarities) if similarities else 0

"""#### All-in-One

This function `evaluation_results` aggregates various metrics to give a comprehensive evaluation of the model's performance in generating method names
"""

def evaluation_results(labels, predictions):
    # Calculate various evaluation metrics for the generated method names

    # Calculate and print accuracy
    accuracy = calculate_accuracy(predictions, labels)
    print(f"Accuracy: {accuracy:.2f}")

    # Calculate and print ROUGE score
    rouge = calculate_rouge_n(predictions, labels)
    print(f"ROUGE: {rouge:.2f}")

    # Calculate and print Levenshtein distance
    levenshtein_distance = calculate_levenshtein_distance(labels, predictions)
    print(f"Levenshtein Distance: {levenshtein_distance}")

    # Calculate and print semantic similarity
    semantic_similarity = calculate_similarity(labels, predictions)
    print(f"Semantic Similarity: {semantic_similarity:.2f}")

"""### Data Tokenization

As a common practice, we split the dataset into:

- **Training set**: Data used for training the model parameters.
- **Validation set**: Data used for hyperparameter tuning or early stopping to avoid overfitting.
- **Test set**: Data used for checking what performance we can expect on new data.

The split ratio depends on the specific requirements of your project, but common practices are:

- Training Set: 70-80%
- Validation Set: 10-15%
- Test Set: 10-15%

We will use 70:15:15.
"""

# Extract the 'identifier' column as labels
labels = df['identifier']

# Drop the 'identifier' column from the DataFrame to get input data
data = df.drop('identifier', axis=1)

# Splitting the data into training and a temporary test set
train_data, temp_test_data, train_labels, temp_test_labels = train_test_split(data, labels, test_size=10000, random_state=42)

# Splitting the temporary test set into actual test and validation sets
test_data, val_data, test_labels, val_labels = train_test_split(temp_test_data, temp_test_labels, test_size=5000, random_state=42)

mean_identifier_len = math.ceil(mean([len(split_camel_case(i)) for i in labels]))+2 # will be used in geenration

"""The model expects data in the form of **Dataset** type. We also create a **DatasetDict** to organize the datasets. This structure is beneficial when working with models that require specific data formats such as those found in the Hugging Face Transformers library."""

# Creating Dataset objects for each set
train_dataset = Dataset.from_pandas(train_data.assign(labels=train_labels))
test_dataset = Dataset.from_pandas(test_data.assign(labels=test_labels))
val_dataset = Dataset.from_pandas(val_data.assign(labels=val_labels))

# Creating a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

"""We initialize the tokenizer for the T5-base model, as previously set up in "CodeT5+: description and first try". The tokenizer transforms textual input into a sequence of token ids, which is essential for model processing.

A key parameter, max_input_length, is set to 512 to define the maximum length of the input sequences. The preprocessing function, `preprocess_data`, tokenizes both the inputs (full code) and the labels (method names) while ensuring they adhere to this length constraint. This tokenized data is then used to train and evaluate the model.
"""

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


# Applying the preprocessing function to the datasets
# This step tokenizes all datasets and removes unnecessary columns
tokenized_datasets = dataset_dict.map(preprocess_data, remove_columns=columns_to_remove)

# Saving the tokenized dataset using LZMA compression
with lzma.open('tokenized_datasets.xz', 'wb') as f:
    # Serializing the tokenized datasets with pickle
    pickle.dump(tokenized_datasets, f)

# This step saves the preprocessed and tokenized dataset to disk.
# It allows for quicker access and reduces processing time in subsequent uses of this notebook.

"""### Unfinetuned Model Evaluation"""

# Loading the previously saved tokenized dataset from disk
with lzma.open('tokenized_datasets.xz', 'rb') as f:
    # Deserializing the tokenized datasets using pickle
    tokenized_datasets = pickle.load(f)

# This step loads the preprocessed and tokenized dataset from disk.
# It allows for quick access to the dataset that was saved in a previous session.

"""Here is selecting subsets of the tokenized dataset for training, validation, and testing. The sizes of these subsets are chosen for efficient use of resources. At this stage only test_dataset will be used."""

train_dataset = tokenized_datasets["train"].select(range(100000))
validation_dataset = tokenized_datasets["validation"].select(range(4000))
test_dataset = tokenized_datasets["test"].select(range(4000))

"""This code snippet is responsible for generating predictions using a given language model and cleaning the results for better readability and consistency. Here's a breakdown of the key steps:

The `clean_generated_text` function takes a generated text as input and performs the following steps:
   - Removes non-alphabetic characters, keeping only letters.
   - Splits the cleaned text into words.
   - Removes duplicate words (model can generate long words with duplicates to extend the result up to max token number output).
   - Reconstructs the text in camelCase style, where the first word starts with a lowercase letter and the rest are capitalized. This ensures consistency in the generated method names.

The `generate_predictions` function iterates through the test dataset in batches and generates predictions for each batch using the model. The generated text is then cleaned using the `clean_generated_text` function. Empty predictions (those containing only special tokens) are replaced with
```
<no_prediction>
```

 to indicate that the model did not provide a valid method name.

"""

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

"""Now it's time to generate names."""

# Generate predictions for a test dataset using the specified model and tokenizer.
PRED_BATCH_SIZE = 8
predictions = generate_predictions(model, tokenizer, test_dataset, PRED_BATCH_SIZE, device)

# Saving predictions to a file for further analysis or evaluation.
with open('predictions.txt', 'w') as file:
    for prediction in predictions:
        file.write(prediction + '\n')

"""Finally, unfinetuned model evaluation."""

# Reading predictions from the file
with open('predictions.txt', 'r') as file:
    predictions = file.read().splitlines()

# Decoding labels with already defined tokenizer
labels = [tokenizer.decode(label_ids, skip_special_tokens=True) for label_ids in test_dataset["labels"]]

# Show metrics evaluation
evaluation_results(labels, predictions)

"""## Model Finetuning: Training, Evaluation

### Finetuning

The Seq2SeqTrainingArguments is a configuration object used in the initialization of the trainer for sequence-to-sequence (seq2seq) models. It specifies various training settings and hyperparameters for the training process. Here's a description of some of the key arguments used in this configuration:

* **model_dir**: The directory where model checkpoints and logs will be saved.

* **evaluation_strategy**: Specifies when evaluation should be performed during training. In this case, it's set to "steps," meaning evaluation will be performed at regular step intervals.

* **eval_steps**: The number of steps between each evaluation. For setting this and two next parameters the number of total iterations can be used. To calculate the total number of iterations, divide the length of the training dataset by the batch size, then multiply the result by the number of training epochs (specified in the `num_train_epochs` parameter).

* **logging_strategy**: Specifies when training logs should be generated. Also set to "steps."

* **logging_steps**: The number of steps between each logging of training progress.

* **save_strategy**: Determines when model checkpoints should be saved. Here, it's set to "steps."

* **save_steps**: The number of steps between each model checkpoint save.

* **learning_rate**: The learning rate used in training.

* **per_device_train_batch_size**: The batch size per device for training data. Larger values can lead to faster training but require more memory.

* **per_device_eval_batch_size**: The batch size per device for evaluation data.

* **weight_decay**: A regularization term to prevent overfitting.

* **save_total_limit**: Limits the total number of saved checkpoints.

* **num_train_epochs**: The number of training epochs.

* **predict_with_generate**: Indicates that generation should be used for predictions.

* **fp16**: Enables mixed-precision training, which is faster and uses less memory on GPUs. It's set to True, meaning it will be used when training on CUDA-compatible devices.

* **load_best_model_at_end**: Specifies whether the best model checkpoint should be loaded at the end of training.

* **metric_for_best_model**: The evaluation metric used to determine the best model checkpoint. Here, it's set to "rouge1," indicating ROUGE-1 score.

* **report_to**: Specifies where training metrics should be reported. In this case, it's set to "tensorboard," which means TensorBoard will be used for visualization and monitoring during training.

Overall, Seq2SeqTrainingArguments is a crucial component for configuring and customizing the training process for sequence-to-sequence models. It allows you to fine-tune various aspects of training and adapt them to your specific use case and hardware resources. I have chosen very standard parameters.
"""

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

"""Data collator is used to preprocess and batch the training data for the Seq2Seq trainer. It handles tasks such as padding, truncation, and tokenization required for sequence-to-sequence models. It ensures that the input and target sequences are appropriately formatted for training."""

data_collator = DataCollatorForSeq2Seq(tokenizer)

"""The compute_metrics function is defined to calculate evaluation metrics for generated text. It takes evaluation predictions (eval_preds) as input, which consists of predicted and true labels. The function first decodes the predictions and labels, considering special tokens. Then, it formats the text for ROUGE evaluation, ensuring proper newline separation between sentences.
Finally, the `rouge_metric.compute` method is used to calculate ROUGE scores by comparing the decoded predictions with the decoded labels. The `use_stemmer` parameter is set to True, indicating that stemming should be used in the ROUGE calculation.
Both the Rouge Metric and the `compute_metrics` function are necessary components for the Seq2Seq trainer to evaluate the model's performance during training.
"""

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

""" In this code block, I define a function `model_init()` that returns an untrained model to be used for training. I then initialize the `Seq2SeqTrainer` object (trainer) with various components and configurations. All the components and settings defined in the previous sections are now combined to set up the training process. The `model_init()` function initializes the model, and the args, `train_dataset`, `validation_dataset`, `data_collator`, `tokenizer`, and `compute_metrics` are all configured to facilitate the training process.







"""

# Function that returns an untrained model to be trained
def model_init():
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)
    model = model.to('cuda')
    return model


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

"""Finally, finetuning."""

trainer.train()

"""
In the end, saving the model is essential."""

trainer.save_model("/model")

"""### Evaluation

The entire process is identical to testing the unfine-tuned model, with the exception of using the fine-tuned model and tokenizer for initialization.
"""

# Load the fine-tuned model and tokenizer
model_dir = '/model'
fine_tokenizer = AutoTokenizer.from_pretrained(model_dir)
fine_model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)

# Move the fine-tuned model to the specified device
fine_model.to(device)

# Generate predictions using the fine-tuned model
finetuned_predictions = generate_predictions(fine_model, fine_tokenizer, test_dataset, PRED_BATCH_SIZE, device)

# Saving predictions to a file
with open('predictions_finetuned.txt', 'w') as file:
    for prediction in finetuned_predictions:
        file.write(prediction + '\n')

# Reading predictions from the file
with open('predictions_finetuned.txt', 'r') as file:
    finetuned_predictions = file.read().splitlines()

# Show metrics evaluation for finetuned_predictions
evaluation_results(labels, finetuned_predictions)

"""## Conclusion: Summary of the Work, Evaluation, Problems and Future Steps

### Summary of the Work

In this assignment, I worked on the method name prediction task, which involved the following key steps:

1. **Introduction of CodeT5+**

2. **Data Collection**:

I selected a large open-source project, IntelliJ Community, as the source of code data.
I extracted all the methods from this project to create a dataset for the task.

3. **Data Preprocessing**:

I performed data preprocessing, which included tasks like cleaning and formatting the extracted code to prepare it for input into the model.

4. **Metric Definition**:

To evaluate the model's performance, I defined appropriate metrics. These metrics were used to assess the quality of the generated method names compared to the ground truth.

5. **Testing of Unfinetuned Model**:

I initially tested the CodeT5+ model without fine-tuning to establish a baseline for its performance.
This allowed me to understand the model's initial capabilities and limitations.

6. **Fine-Tuning Process**:

To improve the model's performance on the method name prediction task, I fine-tuned it using the collected dataset.
Fine-tuning involved training the model on the specific data to adapt it to the task's requirements.

7. **Testing of Finetuned Model**:

After fine-tuning, I re-evaluated the model's performance to observe the changes in quality.
I compared the results with those obtained from the unfinetuned model to assess the impact of fine-tuning.
Throughout this assignment, I aimed to enhance the model's ability to predict method names accurately and efficiently, demonstrating the iterative nature of the model development process.

### Evaluation

To evaluate the performance of the unfinetuned codet5+ model and its finetuned version, we look at various metrics such as Accuracy, ROUGE, Levenshtein Distance, and Semantic Similarity. These metrics give us an understanding of how well the model is performing in terms of generating correct predictions (Accuracy), capturing the overlap with the reference text (ROUGE), the number of edits required to change the prediction into the reference (Levenshtein Distance), and the closeness in meaning between the prediction and the reference (Semantic Similarity).

The unfinetuned model has the following evaluation metrics:

* Accuracy: 0.51
* ROUGE: 0.47
* Levenshtein Distance: 10.1075
* Semantic Similarity: 0.68

After finetuning, the model shows these results:

* Accuracy: 0.70
* ROUGE: 0.67
* Levenshtein Distance: 6.18975
* Semantic Similarity: 0.82

From these results, we can conclude that finetuning has significantly improved the performance of the codet5+ model. The accuracy has increased from 0.51 to 0.70, indicating that the model makes correct predictions more frequently after finetuning. The ROUGE score has also increased, suggesting that the overlap between the model's predictions and the reference text is better after finetuning.

The decrease in the Levenshtein Distance from 10.1075 to 6.18975 shows that the predictions made by the finetuned model are closer to the reference, requiring fewer edits to match exactly. Finally, the increase in Semantic Similarity from 0.68 to 0.82 indicates that the meaning of the predictions is much closer to the meaning of the reference text after finetuning.

Overall, finetuning has had a positive effect on the model's ability to generate predictions that are more accurate, more aligned with the reference, and semantically similar to the expected output.

### Problems and Future Steps

#### Problems:

* **Resource Constraints**: Effective utilization of cloud resources necessitates strategic budgeting. Additional resources are required to support models with a higher number of parameters and larger datasets for training, evaluation, and testing, as well as for the collection of more data.

* **Model Imperfection**: The model's initial flaws, such as the duplication of outputs and inappropriate name generation for tasks, can adversely affect its performance.

* **Abstract Predictions**: The model has difficulty predicting names for abstract methods and constructors, a common issue in code generation tasks.

#### Further Steps:

* **Data Cleaning Model**: Developing a model specifically for data cleaning is advantageous. Care must be taken to ensure this model does not introduce its own biases or errors.

* **Enhanced Prediction Mechanisms**: Improving prediction accuracy by detecting constructors and their corresponding classes. A context-aware system that comprehends code structure, including recognizing annotations like 'override' and 'deprecated', could enhance the accuracy in predicting method names and functionalities.

* **Active Learning**: Implementing an active learning loop in production, where the model continuously learns from new data, particularly from instances it currently misinterprets, can be a cost-effective method for ongoing improvement.

* **Diverse Learning Strategies**: Categorizing methods based on various criteria can be beneficial. These criteria might include the length of the code (such as one-line, multi-line, or beyond a certain maximum length), the purpose of the methods (like test, interface, abstract, etc.), or the type of identifiers used in the methods.
"""

