import numpy as np
import re
import spacy
import Levenshtein
from collections import Counter


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


# Levenshtein Distance
def calculate_levenshtein_distance(labels, predictions):
    # Calculating Levenshtein distance for each pair of label and prediction
    distances = [Levenshtein.distance(ref, pred) for ref, pred in zip(labels, predictions)]

    # Returning the average distance
    return np.mean(distances)


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
