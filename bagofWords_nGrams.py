# =========================================================
# BAG OF N-GRAMS + NEWS CLASSIFICATION USING NAIVE BAYES
# =========================================================
# This script demonstrates:
# 1. How Bag of Words and N-grams work
# 2. Text preprocessing using spaCy (remove stopwords + lemmatization)
# 3. News category classification using Multinomial Naive Bayes
# 4. Model evaluation and confusion matrix visualization
# =========================================================

# -------------------------------
# 1. Import Required Libraries
# -------------------------------

import pandas as pd
import numpy as np
import spacy
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 2. Demonstration of N-grams
# -------------------------------
# CountVectorizer converts text into numerical vectors
# using Bag of Words or N-grams.

v = CountVectorizer()

v.fit(["Thor Hathodawala is looking for a job"])

# Vocabulary contains word → index mapping
print("Unigram Vocabulary:")
print(v.vocabulary_)

# Use unigram + bigram
v = CountVectorizer(ngram_range=(1,2))
v.fit(["Thor Hathodawala is looking for a job"])

print("\nUnigram + Bigram Vocabulary:")
print(v.vocabulary_)

# Use unigram + bigram + trigram
v = CountVectorizer(ngram_range=(1,3))
v.fit(["Thor Hathodawala is looking for a job"])

print("\nUnigram + Bigram + Trigram Vocabulary:")
print(v.vocabulary_)


# -------------------------------
# 3. Text Preprocessing using spaCy
# -------------------------------
# We remove:
# - stop words
# - punctuation
# - apply lemmatization

nlp = spacy.load("en_core_web_sm")

def preprocess(text):

    doc = nlp(text)
    filtered_tokens = []

    for token in doc:

        # Skip stopwords and punctuation
        if token.is_stop or token.is_punct:
            continue

        # Use lemma (base form of word)
        filtered_tokens.append(token.lemma_)

    return " ".join(filtered_tokens)


# Example preprocessing
print("\nPreprocessing Example:")
print(preprocess("Thor ate pizza"))


# -------------------------------
# 4. Load News Dataset
# -------------------------------
# Dataset contains:
# text → news article
# category → BUSINESS, SPORTS, CRIME, SCIENCE

df = pd.read_json("news_dataset.json")

print("\nDataset Shape:", df.shape)
print(df.head())


# -------------------------------
# 5. Handle Class Imbalance
# -------------------------------
# SCIENCE class has fewer samples.
# We perform undersampling to balance the dataset.

min_samples = 1381

df_business = df[df.category=="BUSINESS"].sample(min_samples, random_state=2022)
df_sports = df[df.category=="SPORTS"].sample(min_samples, random_state=2022)
df_crime = df[df.category=="CRIME"].sample(min_samples, random_state=2022)
df_science = df[df.category=="SCIENCE"].sample(min_samples, random_state=2022)

df_balanced = pd.concat(
    [df_business, df_sports, df_crime, df_science],
    axis=0
)

print("\nBalanced Dataset:")
print(df_balanced.category.value_counts())


# -------------------------------
# 6. Convert Category to Numbers
# -------------------------------

df_balanced["category_num"] = df_balanced["category"].map({
    "BUSINESS":0,
    "SPORTS":1,
    "CRIME":2,
    "SCIENCE":3
})


# -------------------------------
# 7. Preprocess Text
# -------------------------------

df_balanced["preprocessed_txt"] = df_balanced["text"].apply(preprocess)


# -------------------------------
# 8. Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    df_balanced.preprocessed_txt,
    df_balanced.category_num,
    test_size=0.2,
    random_state=2022,
    stratify=df_balanced.category_num

)


# -------------------------------
# 9. Create ML Pipeline
# -------------------------------
# Pipeline automatically performs:
# Text → Vectorization → Model Training

clf = Pipeline([

    # Bag of N-grams (Unigram + Bigram)
    ('vectorizer', CountVectorizer(ngram_range=(1,2))),

    # Naive Bayes classifier
    ('nb', MultinomialNB())

])


# -------------------------------
# 10. Train Model
# -------------------------------

clf.fit(X_train, y_train)


# -------------------------------
# 11. Make Predictions
# -------------------------------

y_pred = clf.predict(X_test)


# -------------------------------
# 12. Evaluate Model
# -------------------------------

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


# -------------------------------
# 13. Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))

sn.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()


# =========================================================
# END OF PROGRAM
# =========================================================