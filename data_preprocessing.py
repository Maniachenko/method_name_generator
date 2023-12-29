import pandas as pd
import ast
import re

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
    # Load the dataset
    df = pd.read_csv(file_path)

    # Update 'type_identifier' column
    df['type_identifier'] = df.apply(
        lambda row: find_type(row) if pd.isna(row['type_identifier']) else row['type_identifier'], axis=1)

    # Remove unnecessary columns
    columns_to_remove = ["boolean_type", "floating_point_type", "generic_type", "scoped_type_identifier", "integral_type",
                         "void_type"]
    df.drop(columns=columns_to_remove, inplace=True)

    # Filter the DataFrame
    df = df[df.apply(check_not_empty, axis=1)]

    # Generate the full method code
    df['full_code'] = df.apply(full_code_feature, axis=1)

    # Optionally, save the preprocessed DataFrame to a new CSV
    df.to_csv('preprocessed_java_methods.csv')

# Execute the script
if __name__ == "__main__":
    preprocess_data('java_methods.csv')
