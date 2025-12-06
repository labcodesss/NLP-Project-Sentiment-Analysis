📘 NLP Sentiment Analysis – Movie Review Classifier

An end-to-end Natural Language Processing (NLP) project that classifies movie reviews as Positive or Negative using classical machine learning techniques. This project includes text preprocessing, TF-IDF vectorization, model training, saving artifacts, and deploying an interactive interface using Streamlit.

This repository is structured and documented for showcasing in ML/DL portfolios, university applications, and interviews.

🧠 Project Overview

This project demonstrates a full NLP pipeline:

1️⃣ Preprocess text — cleaning, normalization, tokenization
2️⃣ Convert text to numerical features — TF-IDF vectorizer
3️⃣ Train a sentiment classifier — Logistic Regression
4️⃣ Evaluate & save model artifacts — .joblib files
5️⃣ Deploy an interactive UI using Streamlit
6️⃣ Predict sentiment in real-time

It is lightweight, fast, and deployable on Streamlit Cloud for free.

🧪 Model Workflow
1️⃣ Data Preprocessing

Lowercasing
Removing punctuation
Removing URLs
Normalizing whitespace
Tokenization & lemmatization (if enabled)

2️⃣ Feature Extraction

TF-IDF (Term Frequency–Inverse Document Frequency)
Uses unigrams and bigrams
Maximum of ~30,000 features

3️⃣ Model Training

Algorithm used:
👉 Logistic Regression (scikit-learn)
Easy to interpret
Fast
Performs well for bag-of-words NLP models
Artifacts saved:
tfidf_vectorizer.joblib
sentiment_model.joblib

4️⃣ Prediction

For any input text, the UI or CLI shows:
Predicted label (Positive or Negative)
Positive probability score
Negative probability score


🖥️ How to Run Locally
✔️ 1. Create environment & install dependencies
pip install -r requirements.txt

✔️ 2. (Optional) Train model
python models/train_standalone.py

This generates .joblib model files inside models/.

✔️ 3. Run the Streamlit app

If using the deployment file:

streamlit run streamlit_app.py

If using the offline local app:

streamlit run src/app_direct.py


Streamlit will open at:

http://localhost:8501

🌐 How to Deploy on Streamlit Cloud

Push your repository to GitHub
Go to: https://share.streamlit.io
Click New App
Select your repo: NLP-Sentiment-Analysis

Choose:
Branch: main
File: streamlit_app.py


Deploy!
After deployment, Streamlit provides a public link you can share.

🎯 Key Features

✔ End-to-end NLP pipeline
✔ Real-time sentiment prediction
✔ Lightweight model (fast to load)
✔ Clean Streamlit UI
✔ Perfect for portfolios and GitHub projects
✔ Easy to deploy
✔ Fully documented

📦 Tech Stack
Component	Technology
Language	Python 3.x
NLP Toolkit	NLTK
ML Model	scikit-learn
Vectorizer	TF-IDF
Deployment	Streamlit
Packaging	Joblib

🤝 Contributing

Pull requests welcome!
Feel free to open an issue for bugs or feature suggestions.

📄 License

MIT License.
Free to use and modify.
