Medical Text Simplifier
A Natural Language Processing (NLP) project designed to simplify complex medical text into easy to understand language using a hybrid rule based + Transformer approach along with a readability classification model.  
This project also includes an interactive Streamlit web application for real time simplification and readability analysis.


Project Overview:
Medical literature is often highly technical and difficult for non-medical readers to understand.  
This project addresses the problem by automatically simplifying medical sentences while preserving their meaning and clinical accuracy.

The system uses:
- ✅ Rule based medical dictionary for term replacement  
- ✅ T5 Transformer model (t5 small) for context aware simplification  
- ✅ Gradient Boosting Classifier to classify readability (Easy / Medium / Difficult)  
- ✅ Readability metrics (Flesch Reading Ease, Flesch-Kincaid Grade Level)  
- ✅ Streamlit interface for interactive usage  


 Objectives:
- Simplify complex medical text while retaining core meaning  
- Use both rule based and machine learning methods for robust simplification  
- Classify the readability difficulty of medical text  
- Provide a user friendly interface for simplification and evaluation  
- Quantify readability improvement using real metrics  


Dataset:
Med-EASi Dataset (via HuggingFace)

Contains pairs of:
- Expert (complex medical sentences)
- Simple (expert simplified sentences)

A custom 30 sample dataset is also included for demonstration.



 Methodology & Models:

1️⃣ Text Simplification
  Hybrid approach:
- Rule Based Simplification  
  Replaces complex medical terms using an enhanced dictionary.
- T5 Transformer Model (t5 small)
  Performs context aware rewriting after rule based preprocessing.
- Optional Google Translate  
  Ensures English output if multilingual fragments appear.



2️⃣ Readability Classification
Model: GradientBoostingClassifier

Input features:
-  TF-IDF features (1–2 n-grams, top 100 terms)  
-  12 linguistic features:  
  - word count  
  - char count  
  - sentence count  
  - syllable count  
  - difficult words  
  - SMOG index  
  - avg words/sentence  
  - avg syllables/word  
  - long words count  
  - comma count  
  - etc.

Outputs:
- `Easy`
- `Medium`
- `Difficult`


3️⃣ Readability Metrics
Uses textstat to compute:
- Flesch Reading Ease (FRE)  
- Flesch-Kincaid Grade  
- SMOG Index  

Used to measure improvement after simplification.


System Architecture:

Training Pipeline
1. Load dataset  
2. Preprocess data  
3. Compute readability metrics  
4. Extract TF-IDF + linguistic features  
5. Scale features  
6. Train Gradient Boosting model  
7. Save model, vectorizer, scaler using joblib  

Simplification Pipeline
1. Receive user text  
2. Apply rule based simplification  
3. Apply T5 model simplification  
4. Predict readability  
5. Compute readability improvement  
6. Display results in Streamlit  


Results & Performance:
 ✔ Readability Improvement
- Average FRE Gain: +10.93  
- Average Grade Drop: −2.18  
- Success Rate: 57%  
- **Simplified text ≤ Grade 8:** 23%

 ✔ Classification Accuracy
Confusion matrix shows strong performance for:
- Difficult → Difficult  
- Medium → Medium  

Low misclassification across classes.


Streamlit Application Features:
- Upload or load sample dataset  
- Train the ML classifier with adjustable hyperparameters  
- Enter any medical text for simplification  
- View original & simplified text side-by-side  
- View readability metrics & model predictions  
- Visualizations:
  - Bar charts  
  - Scatter plots  
  - Confusion matrix  
  - Histogram (Readability improvement distribution)


Technologies Used
- **Python**
- **Streamlit**
- **HuggingFace Transformers (T5)**
- **scikit-learn**
- **pandas, numpy**
- **textstat**
- **matplotlib, seaborn**
- **joblib**
- **googletrans**

