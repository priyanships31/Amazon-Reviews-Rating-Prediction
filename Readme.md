# Amazon Review Rating Prediction

This project compares different feature extraction techniques and machine learning models to predict Amazon review ratings based on textual reviews.

## Project Overview

The goal is to predict review ratings (1-5) from text reviews using:
- **Feature extraction techniques**: TF-IDF, Word2Vec, Doc2Vec, FastText, Universal Sentence Encoder (USE), and SentenceTransformer
- **Machine learning models**: Linear Regression, SVM, Random Forest, XGBoost, and Neural Networks
- **Evaluation metrics**: MSE, R², and MAE

## Dataset

- Source: Amazon review data (small subset)
- Format: Text reviews with star ratings (1-5)
- Split: 80% train, 20% test
- Sample size: 10,000 training, 2,000 test reviews (due to computational constraints)

## Feature Extraction Methods

1. **TF-IDF**: 
   - Max features: 5000
   - N-gram range: (1,2)

2. **Word2Vec**:
   - Vector size: 300
   - Window: 5
   - Min count: 3

3. **Doc2Vec**:
   - Vector size: 300
   - Min count: 3
   - Epochs: 10

4. **FastText**:
   - Vector size: 300
   - Window: 5
   - Min count: 3

5. **Universal Sentence Encoder (USE)**:
   - Pretrained model from TensorFlow Hub

6. **SentenceTransformer**:
   - Pretrained 'all-MiniLM-L6-v2' model

## Models Evaluated

### Traditional Machine Learning
1. Linear Regression (with MaxAbsScaler)
2. SVM (RBF kernel)
3. Random Forest (100 estimators)
4. XGBoost (200 estimators, max_depth=5, learning_rate=0.1)

### Neural Networks
1. Sequential Dense NN (64 hidden units, ReLU, Dropout=0.3)
2. Sequential Dense NN with Early Stopping (patience=5)

## Results Summary

### Best Performing Models

| Model Type          | Best Feature Extraction | Best Test R² | Best Test MAE | Best Test MSE |
|---------------------|-------------------------|--------------|---------------|---------------|
| XGBoost             | USE                     | 0.446862     | 0.824849      | 1.058877      |
| SVM                 | USE                     | 0.437328     | 0.827933      | 1.077128      |
| Dense NN (EarlyStop)| USE                     | 0.427904     | 0.838643      | 1.09517       |

### Key Findings

1. **Universal Sentence Encoder (USE)** embeddings consistently performed best across most models
2. **XGBoost** achieved the highest R² score (0.447) with USE embeddings
3. Neural networks showed competitive performance but required more computational resources

## How to Run

1. Install dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost gensim tensorflow tensorflow-hub sentence-transformers
