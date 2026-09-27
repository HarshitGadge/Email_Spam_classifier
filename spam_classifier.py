# %% [markdown]
# # SMS spam classifier
#
# **Goal:** filter spam text messages while almost never blocking a real message. A false positive (a real message sent to spam)
# is worse than letting some spam through.
#
# **Data:** SMS Spam Collection (UCI / Kaggle), 5,572 labelled messages. Despite this repo's name, the dataset is SMS, not email.
#
# **What changed from the first version:** the TF-IDF vocabulary used to be fitted on all messages before the train/test split.
# It's now fitted inside a pipeline on training data only, three models are compared with cross-validation, and results are
# reported as spam precision and recall rather than accuracy alone.

# %%
import re, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score
SEED = 2

df = pd.read_csv("data/spam.csv", encoding="latin-1", usecols=["v1", "v2"]).rename(columns={"v1": "label", "v2": "message"})
df = df.drop_duplicates()
df["y"] = (df.label == "spam").astype(int)
print(f"{len(df):,} unique messages | spam share {df.y.mean():.3f}")

# %%
X_train, X_test, y_train, y_test = train_test_split(df.message, df.y, test_size=0.2, stratify=df.y, random_state=SEED)
tfidf = lambda: TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
models = {
    "multinomial naive Bayes (original)": make_pipeline(tfidf(), MultinomialNB()),
    "logistic regression": make_pipeline(tfidf(), LogisticRegression(C=10, max_iter=2000)),
    "linear SVM": make_pipeline(tfidf(), LinearSVC(C=1.0)),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
cv_tbl = pd.DataFrame({m: {k.replace("test_", ""): v.mean() for k, v in cross_validate(est, X_train, y_train, cv=cv, scoring=["precision", "recall", "f1"]).items() if k.startswith("test_")}
                       for m, est in models.items()}).T
display(cv_tbl.round(3))
best = cv_tbl.f1.idxmax(); print("selected on CV F1:", best)

# %% [markdown]
# ## Held-out test set

# %%
final = models[best].fit(X_train, y_train); pred = final.predict(X_test)
print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))
cm = confusion_matrix(y_test, pred); print("confusion matrix [[ham→ham ham→spam] [spam→ham spam→spam]]:\n", cm)
print(f"real messages wrongly sent to spam: {cm[0,1]} of {cm[0].sum()} | spam that got through: {cm[1,0]} of {cm[1].sum()}")

# %%
base = models["multinomial naive Bayes (original)"].fit(X_train, y_train).predict(X_test)
print(f"original model on the same split: spam precision {precision_score(y_test, base):.3f}, spam recall {recall_score(y_test, base):.3f}")

# %%
vec, clf = final[0], final[-1]
if hasattr(clf, "coef_"):
    w = pd.Series(clf.coef_[0], index=vec.get_feature_names_out()).sort_values()
    display(pd.DataFrame({"strongest spam signals": w.tail(15)[::-1].index, "strongest ham signals": w.head(15).index}))
