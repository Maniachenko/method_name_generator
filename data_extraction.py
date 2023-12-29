# Import statements
import os
import pandas as pd
from tqdm import tqdm
from tree_sitter import Language, Parser
import argparse

JAVA_EXTENSIONS = [".java"]
ENCODINGS = ['utf-8', 'ISO-8859-1', 'windows-1252', 'utf-16']


# Load Java language parser
def load_java_parser():
    """
    Builds and loads the Java language parser using tree-sitter.
    Returns: Parser object for Java language.
    """
    Language.build_library('build/my-languages.so', ['tree-sitter-java'])
    java_language = Language('build/my-languages.so', 'java')
    parser = Parser()
    parser.set_language(java_language)
    return parser


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


def read_file(file_path):
    """
    Tries to read the file with different encodings.
    Args:
        file_path (str): Path to the file.
    Returns:
        str: Contents of the file.
    Raises:
        RuntimeError: If all encodings fail.
    """
    for encoding in ENCODINGS:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Failed to decode file: {file_path}")


def parse_java_file(file_path, parser):
    """
    Parses a Java file and extracts method information.
    Args:
        file_path (str): Path to the Java file.
        parser (Parser): tree-sitter Parser for Java.
    Returns:
        list: List of dictionaries, each containing method information.
    """
    source_code = read_file(file_path)
    tree = parser.parse(bytes(source_code, "utf8"))
    root_node = tree.root_node
    methods = extract_methods(root_node)
    for method in methods:
        method['file_path'] = file_path
    return methods


# Main function to process all Java files in a directory
def process_java_files(directory):
    java_files = []
    for dir_path, dir_names, file_names in os.walk(directory):
        java_files.extend(
            [os.path.join(dir_path, file_name) for file_name in file_names if file_name.endswith(".java")])

    all_methods = []
    parser = load_java_parser()

    for file in tqdm(java_files, desc="Processing Java Files"):
        methods_list = parse_java_file(file, parser)
        all_methods.extend(methods_list)

    methods_df = pd.DataFrame(all_methods,
                              columns=['identifier', 'formal_parameters', 'modifiers', 'block', 'type_identifier',
                                       "boolean_type", "floating_point_type", "generic_type", "scoped_type_identifier",
                                       "integral_type", "void_type", 'file_path'])

    methods_df.to_csv('java_methods.csv')


# Execute the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a directory of Java files.')
    parser.add_argument('--dir', '-d', type=str, default='intellij-community',
                        help='Directory containing Java project files (default: intellij-community)')
    args = parser.parse_args()
    process_java_files(args.dir)
