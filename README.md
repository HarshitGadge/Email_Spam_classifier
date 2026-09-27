# SMS Spam Classifier

Filtering spam text messages while almost never blocking a real one. (The repository is named "Email", but the dataset is the
**SMS Spam Collection**: 5,572 labelled text messages, 5,169 after removing duplicates, 12.6% spam.)

## Results (held-out test set, 1,034 messages)

| Model | Spam precision | Spam recall | Real messages sent to spam | Spam that got through |
|---|---|---|---|---|
| Multinomial naive Bayes (original approach) | 100% | 77.9% | 0 of 903 | 29 of 131 |
| **Linear SVM** on TF-IDF 1–2-grams (selected by CV F1) | **99.2%** | **91.6%** | 1 of 903 | 11 of 131 |

The linear SVM catches about 14 points more spam than naive Bayes, at the cost of one real message flagged out of 903.
If blocking a real message is unacceptable, naive Bayes is the safer choice. That's the trade-off to put in front of a product owner.

5-fold cross-validation on the training set (F1 on spam): naive Bayes 0.859, logistic regression 0.908, linear SVM 0.916.

## Method

- Remove duplicate messages, then a stratified 80/20 split.
- TF-IDF (unigrams and bigrams, English stop words removed, sublinear term frequency) is fitted **inside** the pipeline on training data only.
- Naive Bayes, logistic regression and a linear SVM are compared with 5-fold CV. The winner is scored once on the test set.
- The strongest spam signals are words like "txt", "claim", "150p", "won", "ringtone", "reply" and "mobile".

**Fix in this version.** The first notebook fitted the TF-IDF vocabulary on all messages before splitting, so the test set influenced the
features. It also read from a Kaggle-only path. The original notebook is kept in `archive/`.

## Run it

```bash
pip install -r requirements.txt
# put spam.csv in data/ (see data/README.md)
jupyter nbconvert --to notebook --execute spam_classifier.ipynb
```

Tools: pandas, scikit-learn.
