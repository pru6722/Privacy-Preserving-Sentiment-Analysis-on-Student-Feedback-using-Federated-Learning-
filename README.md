# Privacy-Preserving-Sentiment-Analysis-on-Student-Feedback-using-Federated-Learning

## Overview
This project presents a privacy-preserving sentiment analysis system for student feedback by integrating modern machine learning techniques with advanced privacy mechanisms. The system is designed to achieve high predictive performance while ensuring data confidentiality.

The project explores and compares multiple approaches, including transformer-based models, differential privacy, and federated learning, to analyze feedback while preserving user privacy.

---

## Objectives
- Perform sentiment analysis on student feedback  
- Preserve user privacy during model training  
- Compare centralized, privacy-preserving, and distributed learning approaches  

---

## Dataset
The project uses the Coursera Course Reviews dataset, which contains:
- Review text  
- Ratings (1 to 5)  

### Sentiment Mapping
- Ratings 4 and 5 are labeled as Positive  
- Rating 3 is labeled as Neutral  
- Ratings 1 and 2 are labeled as Negative  

Note: The dataset is not included in this repository due to GitHub size limitations.

Dataset link:  
https://www.kaggle.com/datasets/imuhammad/course-reviews-on-coursera  

---

## Models Implemented

### Baseline Models
Traditional machine learning models such as Logistic Regression were implemented as a baseline. These models achieved approximately 40% accuracy, highlighting limitations of small or less expressive models.

---

### Transformer Model
A transformer-based model using DistilBERT was implemented for sentiment classification. This model achieved approximately 95% accuracy on the dataset, demonstrating the effectiveness of pretrained language models for text analysis.

---

### Differential Privacy
Differential Privacy was implemented using the Opacus library. Noise is added to gradients during training to prevent the model from memorizing individual data points.

The model achieved a privacy budget of approximately epsilon equal to 0.94, indicating strong privacy guarantees.

---

### Federated Learning
Federated Learning was implemented by simulating multiple clients. Each client trains a local model on its subset of data, and model parameters are aggregated using the Federated Averaging algorithm.

The federated model achieved approximately 94% accuracy, showing that decentralized training can maintain high performance without sharing raw data.

---

## Privacy Techniques

### Differential Privacy
Ensures that individual data points cannot be inferred from the trained model by adding controlled noise during training.

### Federated Learning
Enables decentralized model training where raw data remains on local devices and only model updates are shared.

---

## Results Comparison

| Approach                     | Accuracy | Privacy |
|-----------------------------|----------|--------|
| Baseline Machine Learning   | ~40%     | No     |
| DistilBERT (Centralized)    | ~95%     | No     |
| Differential Privacy Model  | Lower    | Yes    |
| Federated Learning Model    | ~94%     | Yes    |

---

## Key Insights
- Dataset size and quality significantly influence model performance  
- There is a trade-off between accuracy and privacy  
- Federated learning can achieve near-centralized performance without sharing raw data  
- Differential privacy provides strong guarantees against data leakage  

---

## Technology Stack
- Python  
- PyTorch  
- Hugging Face Transformers  
- Scikit-learn  
- Opacus  

---

## Project Structure

privacy_sentimental_Analysis/

src/
- prepare_data.py        # Dataset preparation and sentiment mapping  
- preprocess.py         # Data preprocessing  
- train_baseline.py     # Baseline machine learning models  
- train_bert.py         # Initial BERT implementation  
- train_model.py        # Final centralized model (DistilBERT)  
- train_dp_model.py     # Differential privacy training  
- train_fl_model.py     # Federated learning simulation  
- utils.py              # Utility functions  

data/ (excluded)  
outputs/ (excluded)  
.gitignore  
README.md  

---

## How to Run

Prepare the dataset:
python src/prepare_data.py  

Train the centralized model:
python src/train_model.py  

Run the differential privacy model:
python src/train_dp_model.py  

Run the federated learning model:
python src/train_fl_model.py  

---

## Future Work
- Combine federated learning with differential privacy  
- Develop a web-based interface for real-time feedback analysis  
- Extend the system to support multilingual feedback  

---

## Author
Prudhvi Sai Lingineni

---

## Conclusion
This project demonstrates how machine learning systems can be designed to balance performance and privacy. By combining transformer models with privacy-preserving techniques such as differential privacy and federated learning, the system provides a practical solution for secure and efficient student feedback analysis.
