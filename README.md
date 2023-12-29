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
git clone [repository-link]
```
2. **Install the required packages:**
```
pip install -r requirements.txt
```

### Running the Complete Process
For a comprehensive, ready-to-run process, refer to the Jupyter notebook [method_name_generation.ipynb](https://github.com/Maniachenko/method_name_generator/blob/master/method_name_generation.ipynb) included in the repository. Alternatively, the project can be executed through individual scripts as detailed above.

 ## Usage
To extract methods from the IntelliJ Community project:
```
python data_extraction.py
```
For data preprocessing:
```
python data_preprocessing.py
```
To train and evaluate the model:
```
python model_training.py
python model_evaluation.py
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

