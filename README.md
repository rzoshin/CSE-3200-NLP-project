# CSE-3200-NLP-project
# Genre Prediction Project README

## Introduction
This repository contains a genre prediction project focusing on Bengali book summaries. The project utilizes a dataset consisting of approximately 4.5K book summaries labeled into seven genre categories. The goal is to build a model that accurately predicts the genre of a given book summary.

## Dataset
The dataset includes the following columns:
- **Id**: Unique identifier for each book summary
- **Summary**: Bengali book summaries
- **Label**: Genre categories (Fiction, Thriller, Children's Book, Political, Science-Fiction, War, Motivational)

## Sections Overview
### 1. Exploratory Data Analysis (EDA)
- Initial exploration of the dataset to understand its structure and characteristics.
- Examination of a random summary to identify noise in the data.

### 2. Pre-Processing
#### Cleaning Process
- Various cleaning techniques employed to preprocess the text data.

#### EDA in Summary Length
- Analysis of summary lengths to gain insights into the distribution of text lengths.

#### Remove Stop Words and Punctuations
- Removal of stop words and punctuations to prepare the text for modeling.

### 3. Modeling
#### Data Division
- Splitting the dataset into training and testing sets.

#### Model Selection
- Selection of Bangla BERT, a pre-trained model, for genre classification.

#### Training the Model
- Training of the selected model using the preprocessed data.

### 4. Evaluation
#### Training and Validation Accuracy Curve
- Visualization of the training and validation accuracy over epochs.

#### Custom Test Function
- Development of a custom function to evaluate the model's performance on the test set.

#### Confusion Matrix
- Generation of a confusion matrix to analyze the model's predictions.

#### Classification Report
- Assessment of the model's performance through a classification report.

## Results
The trained model achieved an accuracy of 82% in predicting the genre of Bengali book summaries.

## Files Included
1. **train.csv**: Training set
2. **test.csv**: Test set
3. **train_oshin.csv**: Cleaned training set
4. **test_oshin.csv**: Cleaned test set

## Dependencies
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- tqdm
- joblib
- TensorFlow
- Transformers

## How to Use
1. Clone the repository.
2. Ensure all dependencies are installed.
3. Run the provided scripts in the respective order to replicate the project workflow.
4. Refer to the generated visualizations and evaluation metrics for insights into model performance.

## Contributors
- [Your Name]

## License
This project is licensed under the [MIT License](LICENSE).
