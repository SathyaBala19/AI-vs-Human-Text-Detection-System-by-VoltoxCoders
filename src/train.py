import os
import sys
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_series

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(base_dir, "Dataset (parquet)")
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)


# step 1 - load the data
print("=" * 45)
print(" Loading dataset")
print("=" * 45)

train_files = [
    os.path.join(data_dir, "train-00000-of-00003.parquet"),
    os.path.join(data_dir, "train-00001-of-00003.parquet"),
    os.path.join(data_dir, "train-00002-of-00003.parquet"),
]

all_dfs = []
for f in train_files:
    print(" reading", os.path.basename(f))
    all_dfs.append(pd.read_parquet(f, columns=["text", "generated"]))

df_full = pd.concat(all_dfs, ignore_index=True)
print("\n total rows loaded :", len(df_full))
print(" human samples (0) :", (df_full["generated"] == 0).sum())
print(" AI samples    (1) :", (df_full["generated"] == 1).sum())

# the dataset has more human text than AI text so we balance it
# taking 30000 from each class = 60000 total for training
n_samples = 30000
human_df = resample(df_full[df_full["generated"] == 0], n_samples=n_samples, random_state=42, replace=False)
ai_df    = resample(df_full[df_full["generated"] == 1], n_samples=n_samples, random_state=42, replace=False)

df_train = pd.concat([human_df, ai_df]).sample(frac=1, random_state=42).reset_index(drop=True)
print("\n using", len(df_train), "rows for training")

df_test = pd.read_parquet(os.path.join(data_dir, "test-00000-of-00001.parquet"), columns=["text", "generated"])
print(" test set size:", len(df_test))


# step 2 - clean the text
print("\n" + "=" * 45)
print(" Cleaning text")
print("=" * 45)

t = time.time()
print(" cleaning training data...")
X_train_clean = preprocess_series(df_train["text"])
print(" done in", round(time.time() - t, 1), "s")

t = time.time()
print(" cleaning test data...")
X_test_clean = preprocess_series(df_test["text"])
print(" done in", round(time.time() - t, 1), "s")

y_train = df_train["generated"].values
y_test = df_test["generated"].values

# just printing one sample to make sure the cleaning looks right
print("\n sample check:")
print(" before:", df_train["text"].iloc[0][:90])
print(" after :", X_train_clean.iloc[0][:90])


# step 3 - tfidf vectorization
print("\n" + "=" * 45)
print(" TF-IDF Vectorization")
print("=" * 45)

# ngram_range (1,2) means we use single words AND pairs of words
# for example "climate change" as a bigram is more useful than just "climate" and "change" separately
tfidf = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=50000,
    sublinear_tf=True,
    min_df=3,
    strip_accents="unicode"
)

t = time.time()
X_train_vec = tfidf.fit_transform(X_train_clean)
print(" vectorizer fit done in", round(time.time() - t, 1), "s")

t = time.time()
X_test_vec = tfidf.transform(X_test_clean)
print(" test transform done in", round(time.time() - t, 1), "s")

print(" vocab size:", len(tfidf.vocabulary_))

joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.joblib"))
print(" vectorizer saved")


# step 4 - train both models
print("\n" + "=" * 45)
print(" Training Models")
print("=" * 45)

# logistic regression - saga solver is good for large sparse matrices
t = time.time()
print("\n [1/2] Logistic Regression...")
lr_model = LogisticRegression(solver="saga", C=1.0, max_iter=1000, n_jobs=-1, random_state=42)
lr_model.fit(X_train_vec, y_train)
print("       done in", round(time.time() - t, 1), "s")
joblib.dump(lr_model, os.path.join(models_dir, "logistic_regression.joblib"))
print("       model saved")

# naive bayes - alpha=0.1 worked better than default 1.0 in my tests
t = time.time()
print("\n [2/2] Naive Bayes...")
nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_vec, y_train)
print("       done in", round(time.time() - t, 1), "s")
joblib.dump(nb_model, os.path.join(models_dir, "naive_bayes.joblib"))
print("       model saved")


# step 5 - check accuracy on test set
print("\n" + "=" * 45)
print(" Results")
print("=" * 45)

print()
print(" {:<22} {:>11} {:>11} {:>11}".format("Model", "Train Acc", "Test Acc", "F1"))
print(" " + "-" * 57)

for model_name, model in [("Logistic Regression", lr_model), ("Naive Bayes", nb_model)]:
    tr_acc = accuracy_score(y_train, model.predict(X_train_vec))
    te_acc = accuracy_score(y_test, model.predict(X_test_vec))
    f1 = f1_score(y_test, model.predict(X_test_vec), average="weighted")
    print(" {:<22} {:>11.4f} {:>11.4f} {:>11.4f}".format(model_name, tr_acc, te_acc, f1))

print("\n all models saved to ./models/")
print(" run app.py to test on new text\n")
