# Import statements
import os
import pandas as pd
from tqdm import tqdm
from tree_sitter import Language, Parser


# Load Java language parser
def load_java_parser():
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


# Function to parse a Java file
def parse_java_file(file_path, parser):
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

# Main function to process all Java files in a directory
def process_java_files(directory):
    java_files = []
    for dir_path, dir_names, file_names in os.walk(directory):
        java_files.extend([os.path.join(dir_path, file_name) for file_name in file_names if file_name.endswith(".java")])

    all_methods = []
    parser = load_java_parser()

    for file in tqdm(java_files, desc="Processing Java Files"):
        methods_list = parse_java_file(file, parser)
        all_methods.extend(methods_list)

    methods_df = pd.DataFrame(all_methods, columns=['identifier', 'formal_parameters', 'modifiers', 'block', 'type_identifier',
                                                    "boolean_type", "floating_point_type", "generic_type", "scoped_type_identifier",
                                                    "integral_type", "void_type", 'file_path'])

    methods_df.to_csv('java_methods.csv')

# Execute the script
if __name__ == "__main__":
    process_java_files('/content/intellij-community')
