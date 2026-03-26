import os
import sys
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import preprocess_series

# i set up all the folder paths here
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir  = os.path.join(base_dir, "Dataset (parquet)")
models_dir = os.path.join(base_dir, "models")
plots_dir = os.path.join(base_dir, "plots")
test_file = os.path.join(data_dir, "test-00000-of-00001.parquet")
os.makedirs(plots_dir, exist_ok=True)

class_names = ["Human", "AI-generated"]

# load saved models
print("loading models...")
vec = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.joblib"))
lr  = joblib.load(os.path.join(models_dir, "logistic_regression.joblib"))
nb  = joblib.load(os.path.join(models_dir, "naive_bayes.joblib"))
print("done")

# load and clean test data
print("loading test data...")
df_test = pd.read_parquet(test_file, columns=["text", "generated"])
X_test = vec.transform(preprocess_series(df_test["text"]))
y_test = df_test["generated"].values
print("test samples:", len(y_test))


# classification report - shows precision recall f1 for each class
print("\n" + "=" * 50)
print("  Classification Reports")
print("=" * 50)

for name, model in [("Logistic Regression", lr), ("Naive Bayes", nb)]:
    print("\n ---", name, "---")
    print(classification_report(y_test, model.predict(X_test), target_names=class_names, digits=4))


# confusion matrix plot
print("saving confusion matrix...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Confusion Matrices - AI vs Human Text Detection", fontsize=13)

for ax, (name, model) in zip(axes, [("Logistic Regression", lr), ("Naive Bayes", nb)]):
    cm = confusion_matrix(y_test, model.predict(X_test))
    pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    sns.heatmap(pct, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, ax=ax, cbar_kws={"format": "%.0f%%"})

    # also showing raw counts below the percentage value
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.72, "n=" + str(cm[i, j]),
                    ha="center", va="center", fontsize=9, color="gray")

    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

plt.tight_layout()
cm_save = os.path.join(plots_dir, "confusion_matrices.png")
plt.savefig(cm_save, dpi=150, bbox_inches="tight")
plt.close()
print("saved:", cm_save)


# ROC curves - AUC close to 1.0 is better
print("\nsaving ROC curve...")

fig, ax = plt.subplots(figsize=(8, 6))

for (name, model), color in zip([("Logistic Regression", lr), ("Naive Bayes", nb)], ["steelblue", "tomato"]):
    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    roc_score = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2, label=name + "  (AUC=" + str(round(roc_score, 4)) + ")")

ax.plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="random")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves")
ax.legend(loc="lower right")

roc_save = os.path.join(plots_dir, "roc_curves.png")
plt.tight_layout()
plt.savefig(roc_save, dpi=150, bbox_inches="tight")
plt.close()
print("saved:", roc_save)


# confidence distribution
# this shows whether the model is confident when it gets things right
# the threshold lines show where our decision boundaries are at 0.85 and 0.60
print("\nsaving confidence distribution...")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Prediction Confidence Distribution", fontsize=13)

for ax, (name, model) in zip(axes, [("Logistic Regression", lr), ("Naive Bayes", nb)]):
    probs = model.predict_proba(X_test)
    conf = probs.max(axis=1)

    ax.hist(conf[y_test == 0], bins=np.linspace(0.5, 1.0, 40), alpha=0.7, color="steelblue", label="Human")
    ax.hist(conf[y_test == 1], bins=np.linspace(0.5, 1.0, 40), alpha=0.7, color="tomato", label="AI-generated")
    ax.axvline(0.85, color="green", linestyle="--", lw=1.5, label="0.85 cutoff")
    ax.axvline(0.60, color="orange", linestyle="--", lw=1.5, label="0.60 cutoff")
    ax.set_title(name)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Count")
    ax.legend(fontsize=9)

plt.tight_layout()
conf_save = os.path.join(plots_dir, "confidence_distribution.png")
plt.savefig(conf_save, dpi=150, bbox_inches="tight")
plt.close()
print("saved:", conf_save)


# top features from logistic regression
# positive coefficient = word pushes prediction toward AI
# negative coefficient = word pushes prediction toward Human
print("\n" + "=" * 50)
print("  Top Features (Logistic Regression weights)")
print("=" * 50)

all_words = np.array(vec.get_feature_names_out())
coef = lr.coef_[0]

n = 20
top_ai    = np.argsort(coef)[-n:][::-1]
top_human = np.argsort(coef)[:n]

print("\n words that push toward AI:")
print("  {:<28} {:>8}".format("word", "weight"))
print("  " + "-" * 38)
for idx in top_ai:
    print("  {:<28} {:>8.4f}".format(all_words[idx], coef[idx]))

print("\n words that push toward Human:")
print("  {:<28} {:>8}".format("word", "weight"))
print("  " + "-" * 38)
for idx in top_human:
    print("  {:<28} {:>8.4f}".format(all_words[idx], coef[idx]))


# decision tier breakdown on the test set
print("\n" + "=" * 50)
print("  Decision Tier Breakdown")
print("=" * 50)

for name, model in [("Logistic Regression", lr), ("Naive Bayes", nb)]:
    conf = model.predict_proba(X_test).max(axis=1)
    total = len(conf)
    t1 = int((conf >= 0.85).sum())
    t2 = int(((conf >= 0.60) & (conf < 0.85)).sum())
    t3 = int((conf < 0.60).sum())

    print("\n " + name)
    print("  Tier 1 - Acceptable   (>= 0.85) :", t1, " ({:.1f}%)".format(100 * t1 / total))
    print("  Tier 2 - Needs Review (0.60-0.84):", t2, " ({:.1f}%)".format(100 * t2 / total))
    print("  Tier 3 - Uncertain    (< 0.60)  :", t3, " ({:.1f}%)".format(100 * t3 / total))

print("\n done. check the plots folder for the graphs.\n")
