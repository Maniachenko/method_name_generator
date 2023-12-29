# Method Name Generation with CodeT5+: A Case Study on IntelliJ Community

## Overview
This project explores enhancing code generation by predicting method names using [**CodeT5+**](https://huggingface.co/models?sort=downloads&search=codet5p), focusing on a large open-source project, [**IntelliJ Community**](https://github.com/JetBrains/intellij-community). The process starts with extracting all methods from the project, forming a comprehensive dataset. Initially, the project tests the unfinetuned CodeT5+ model on this data to assess its baseline performance in predicting method names. Subsequently, the model undergoes fine-tuning with the specific dataset, aimed at improving its accuracy in method name prediction tailored to the Java code in IntelliJ Community. A comparative analysis follows, evaluating the model's performance before and after fine-tuning, with an emphasis on quality improvements.

## Project Structure
1. **Data Collection**: Extracting methods from IntelliJ Community.
2. **Data Preprocessing**: Cleaning and formatting the extracted code.
3. **Model Testing**: Evaluating the unfinetuned CodeT5+ model on the dataset.
4. **Model Fine-Tuning**: Fine-tuning CodeT5+ on the dataset and comparing the pre- and post-tuning performances.
5. **Evaluation**: Assessing the model's performance using metrics such as Accuracy, ROUGE, Levenshtein Distance, and Semantic Similarity.
6. **Conclusion and Future Steps**: Summarizing findings and suggesting future improvements.

## Setup and Installation
1. **Clone the repository**:
```
git clone -b master https://github.com/Maniachenko/method_name_generator.git
```
2. **Install the required packages**:
```
pip install -r requirements.txt
```
3. **Clone Necessary Repositories and Install Additional Dependencies**:

These commands clone additional repositories and install a specific SpaCy language model required for your project. Run these commands in the same environment where you installed the packages from requirements.txt.
* Clone the tree-sitter-java repository:
```
git clone https://github.com/tree-sitter/tree-sitter-java.git
```
* Clone the intellij-community repository:
```
git clone https://github.com/JetBrains/intellij-community.git
```
* Install the en_core_web_md model for SpaCy:
```
python -m spacy download en_core_web_md
```
4. **Download the Model**:

The model required for the project is hosted on Google Drive due to size limitations on GitHub.

* Access the model: Visit the provided [Google Drive link](https://drive.google.com/drive/folders/11q4dsYC9TvM5wrebtqW9HndFXas7mlZ4?usp=sharing).
* Download the model: Download the entire model directory.
* Place the model in your repository: After downloading, move the model directory (/model) to your project's repository folder.

### Running the Complete Process
For a comprehensive, ready-to-run process, refer to the Jupyter notebook [method_name_generation.ipynb](https://github.com/Maniachenko/method_name_generator/blob/master/method_name_generation.ipynb) and .py analog [method_name_generation.py](https://github.com/Maniachenko/method_name_generator/blob/master/method_name_generation.py) included in the repository. Alternatively, the project can be executed through individual scripts as detailed above.

 ## Usage
**Data Collection** 
* Extract Methods from IntelliJ Community Project:
```
python data_extraction.py
```
* For convenience for each file I have some specific arguments. To extract data from a specific Java project directory, use the --dir option with the data_extraction.py script. For example, if you want to process files in a directory named "another-directory", run the command:
```
python data_extraction.py --dir another-directory
```
**Data preprocessing**:
* Default Preprocessing:
```
python data_preprocessing.py
```
* To specify a different file path:
```
python script.py --file path_to_your_file.csv
```
* Or using the short option:

```
python script.py -f path_to_your_file.csv
```

Your data should be in the following format:
| id | identifier  | formal_parameters                                      | modifiers                       | block                                                                                                                                                                                                                                                                                                                                                             | type_identifier  | boolean_type | floating_point_type | generic_type               | scoped_type_identifier | integral_type | void_type | file_path                                                                                                                       |
|----|-------------|--------------------------------------------------------|---------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------------|--------------|---------------------|-----------------------------|------------------------|---------------|-----------|--------------------------------------------------------------------------------------------------------------------------------|
| 0  | getArtifact | ()                                                     | ['@NotNull', 'public']          | {<br>    return myArtifact;<br>}                                                                                                                                                                                                                                                                                                                                   | Artifact         |              |                     |                           |                        |               |           | intellij-community\aether-dependency-resolver\src\org\jetbrains\idea\maven\aether\ArtifactDependencyNode.java                |
| 1  | getDependencies | ()                                                 | ['@NotNull', 'public']          | {<br>    return myDependencies;<br>}                                                                                                                                                                                                                                                                                                                              |                  |              |                     | List<ArtifactDependencyNode> |                        |               |           | intellij-community\aether-dependency-resolver\src\org\jetbrains\idea\maven\aether\ArtifactDependencyNode.java                |
| 2  | isRejected  | ()                                                     | ['public']                      | {<br>    return myRejected;<br>}                                                                                                                                                                                                                                                                                                                                  |                  | boolean      |                     |                           |                        |               |           | intellij-community\aether-dependency-resolver\src\org\jetbrains\idea\maven\aether\ArtifactDependencyNode.java                |
| 3  | getClassifier | ()                                                   | ['@NotNull', 'public']          | {<br>    return myClassifier;<br>}                                                                                                                                                                                                                                                                                                                                | String           |              |                     |                           |                        |               |           | intellij-community\aether-dependency-resolver\src\org\jetbrains\idea\maven\aether\ArtifactKind.java                          |
| 4  | getExtension | ()                                                    | ['@NotNull', 'public']          | {<br>    return myExtension;<br>}                                                                                                                                                                                                                                                                                                                                 | String           |              |                     |                           |                        |               |           | intellij-community\aether-dependency-resolver\src\org\jetbrains\idea\maven\aether\ArtifactKind.java                          |
| 5  | find        | (String classifier, String extension)                  | ['public', 'static']            | {<br>    for (ArtifactKind kind : ArtifactKind.values()) {<br>      if (kind.getClassifier().equals(classifier) && kind.getExtension().equals(extension)) {<br>        return kind;<br>      }<br>    }<br>    return null;<br>  }                                                                                                                                    | ArtifactKind     |              |                     |                           |                        |               |           | intellij-community\aether-dependency-resolver\src\org\jetbrains\idea\maven\aether\ArtifactKind.java                          |
| 6  | kindsOf     | (boolean sources, boolean javadoc, String... artifactPackaging) | ['public', 'static']            | {<br>    EnumSet<ArtifactKind> result = EnumSet.noneOf(ArtifactKind.class);<br>    if (sources) {                                                                                                                                                                                                                                                                |                  |              |                     |                           |                        |               |           | (continues...)                                                                                                                 |

 
**Model Training** (the previous Data Preprocessing step should be implemented):
* Run Model Training:
```
python model_training.py
```
* To specify a model directory and batch size:
```
python script.py --model_dir your_model_directory --batch_size 20
```

**Model Evaluation** (the previous Data Preprocessing step should be implemented):
* Run Model Evaluation:
```
python model_evaluation.py
```
* To specify a batch size:
```
python script_name.py --batch_size 8
```

## Methodology
### Data Collection
The IntelliJ Community project was chosen for its extensive and diverse Java codebase.
Methods were extracted to create a dataset tailored for the task.

### Data Preprocessing
The extracted code underwent thorough cleaning and formatting.

### Model Testing and Fine-Tuning
Initial testing was performed on the unfinetuned CodeT5+ model.
The model was then fine-tuned on the dataset, focusing on improving its accuracy in predicting method names.
### Evaluation Metrics
* **Accuracy**: From 0.51 to 0.70 post-fine-tuning.
* **ROUGE**: Improved from 0.47 to 0.67.
* **Levenshtein Distance**: Decreased from 10.1075 to 6.18975, indicating closer predictions to the reference.
* **Semantic Similarity**: Increased from 0.68 to 0.82, suggesting better alignment with the expected output.
### Challenges and Future Directions
* **Resource Constraints**: Need for strategic budgeting and additional resources.
* **Model Imperfection**: Addressing initial flaws like output duplication.
* **Abstract Predictions**: Enhancing the model's ability to predict names for abstract methods and constructors.
* Future steps include developing a data cleaning model, implementing diverse learning strategies, and active learning.
### Conclusion
The project demonstrates the effectiveness of fine-tuning CodeT5+ in improving method name prediction accuracy. It highlights the potential of machine learning in enhancing code generation, particularly in large open-source projects.

