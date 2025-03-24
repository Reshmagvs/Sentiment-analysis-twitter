# Sentiment Analysis Twitter

## Project Overview
This project demonstrates how to create a Twitter sentiment analysis model using Python. Sentiment analysis is used to determine the emotions and opinions of people on various topics. In this project, we analyze people's sentiment towards Pfizer vaccines by utilizing machine learning techniques on Twitter data.

## Dataset
The dataset used is [![Kaggle Dataset](https://img.shields.io/badge/Kaggle-Pfizer%20Vaccine%20Tweets-blue?style=flat&logo=kaggle)](https://www.kaggle.com/datasets/gpreda/pfizer-vaccine-tweets)


## Tech Stack Used
- **Programming Language:** Python
- **Data Collection:** Tweepy, Kaggle dataset
- **Data Preprocessing:** Pandas, NLTK, Regular Expressions
- **Feature Extraction:** TF-IDF, Count Vectorizer
- **Machine Learning Models:**
  - Logistic Regression
  - Naïve Bayes
  - Random Forest
  - Support Vector Machine (SVM)
  - XGBoost
- **Model Evaluation:** Accuracy Score, F1 Score, Confusion Matrix
- **Visualization:** Matplotlib, Seaborn, WordCloud

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/Reshmagvs/Sentiment-analysis-twitter
   cd Sentiment-analysis-twitter
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Download the dataset from Kaggle and place it in the project directory.

## Usage
1. Run the preprocessing script:
   ```sh
   python preprocess.py
   ```
2. Train the model:
   ```sh
   python train.py
   ```
3. Evaluate the model:
   ```sh
   python evaluate.py
   ```

## Results
The project compares multiple classifiers to determine which model provides the best accuracy for sentiment classification. The performance is evaluated using metrics like accuracy, precision, recall, and F1-score.

## Future Enhancements
- Implement deep learning models like LSTMs and Transformers.
- Use real-time Twitter API to fetch live tweets.
- Deploy the model as a web app.

## Contributing
Feel free to fork this repository and contribute by submitting a pull request.

## License
This project is licensed under the MIT License.
