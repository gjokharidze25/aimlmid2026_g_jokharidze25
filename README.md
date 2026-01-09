# Finding the correlation

### 📌 Problem Description

Given a set of data points:

1. Extract the ((x, y)) coordinates of each point
2. Calculate **Pearson’s correlation coefficient**
3. Interpret the result
4. Visualize the data distribution

---

### 📐 Pearson Correlation Coefficient

The Pearson correlation coefficient is defined as:

r = Σ((xᵢ − x̄)(yᵢ − ȳ)) / √( Σ(xᵢ − x̄)² · Σ(yᵢ − ȳ)² )

where:
- xᵢ, yᵢ are individual data points
- x̄, ȳ are the mean values
- Σ denotes summation
---

### 📁 Project Structure

```
.
├──Finding the correlation
    ├── correlation.py     # Main Python script
    └── scatter.png         # Generated scatter plot (optional)
```

---

### 🧮 Data Used

The following data points were extracted from the graph:

| Point |   x |   y |
| ----: | --: | --: |
|     1 | -10 | -10 |
|     2 |  -5 |  -5 |
|     3 |  -3 |  -1 |
|     4 |  -5 |   2 |
|     5 |  -1 |   1 |
|     6 |   3 |   1 |
|     7 |   1 |  -2 |
|     8 |   5 |  -3 |
|     9 |   7 |  -2 |

---

### Run Script

Run the script:

```bash
python correlation.py
```

This version also generates a scatter plot saved as `scatter.png`.

---

### Output

```
Data points: [(-10, -10), (-5, -5), (-3, -1), (-5, 2), (-1, 1), (3, 1), (1, -2), (5, -3), (7, -2)]
Pearson r (manual): 0.448991
Pearson r (NumPy) : 0.448991
```

---

### Scatter Plot Visualization

The following scatter plot visualizes the distribution of the extracted data points
used to calculate Pearson’s correlation coefficient.

![Scatter Plot of Data Points](Finding%20the%20correlation/scatter.png)

###########################################################################################################################

# Spam email detection

### 📁 Project Structure

```
.
├──Spam email detection/
    │
    ├── requirements.txt
    │
    ├── data/
    │   └── g_jokharidze25_59612.csv
    │
    ├── src/
    │   ├── __init__.py
    │   └── app.py
    │
    ├── models/
    │   └── logistic_regression_model.joblib
    │
    ├── plots/
    │   ├── class_distribution.png
    │   ├── confusion_matrix_heatmap.png
    │   └── feature_coefficients.png
    │
    └── samples/
        ├── spam_email.txt
        └── legitimate_email.txt
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train + Validate (70/30 split)

```bash
python -m src.app train --data data/g_jokharidze25_59612.csv --model-out models.joblib
```

This prints:
- Accuracy
- Confusion Matrix
- Model coefficients + intercept

## Predict on new email text

Interactive (paste email text):

```bash
python -m src.app predict --model models.joblib
```

From a file:

```bash
python -m src.app predict --model models.joblib --email-file samples/spam_email.txt
```

## Plots

- ![Scatter Plot of Data Points](Spam%20email%20detection/plots/class_distribution.png)
- ![Scatter Plot of Data Points](Spam%20email%20detection/plots/confusion_matrix_heatmap.png)
- ![Scatter Plot of Data Points](Spam%20email%20detection/plots/feature_coefficients.png)

## 1) Dataset link (uploaded to GitHub)
Upload the provided dataset file to your repo:
- `Spam email detection/data/g_jokharidze25_59612.csv`
Then paste the GitHub link here: **https://github.com/gjokharidze25/aimlmid2026_g_jokharidze25/blob/main/Spam%20email%20detection/data/g_jokharidze25_59612_csv**

Columns:
- `words` — total word count in the email
- `links` — number of URL-like links (`http://`, `https://`, `www.`)
- `capital_words` — number of ALL-CAPS words (length ≥ 2)
- `spam_word_count` — count of spammy keyword hits
- `is_spam` — class label (1=spam, 0=legitimate)

## 2) Model training (70% training data)
### Data loading and processing (with code)
The dataset is loaded from CSV, required columns are checked, and features/labels are separated:

```python
df = pd.read_csv("data/g_jokharidze25_59612.csv")
X = df[["words", "links", "capital_words", "spam_word_count"]]
y = df["is_spam"]
```

The data is split into 70% training and 30% test, using stratification to keep the same class ratio:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)
```

### Logistic Regression model (with code)
```python
model = LogisticRegression(max_iter=1000, solver="liblinear")
model.fit(X_train, y_train)
```

### Coefficients found by the model
Using the provided dataset and a 70/30 split (random_state=42, stratify=y), the fitted coefficients are:

- words: +0.005497
- links: +0.624817
- capital_words: +0.350937
- spam_word_count: +0.586838
- intercept: -7.171792

(Interpretation: positive coefficients increase the probability of being spam.)

Source code link for training: **https://github.com/gjokharidze25/aimlmid2026_g_jokharidze25/blob/main/Spam%20email%20detection/src/app.py**

## 3) Validation (Confusion Matrix + Accuracy)
### Code
```python
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
```

### Results
Accuracy: **0.9547** (≈ 95.47%)

Confusion Matrix (rows=Actual, cols=Predicted):
```
[[353   9]
 [ 25 363]]
```

## 4) Classify a new email text (feature extraction)
The console application supports entering an email and extracting the same features as the dataset:
- `words` (word count)
- `links` (URL count)
- `capital_words` (ALL-CAPS word count)
- `spam_word_count` (hits from a spam keyword list)

Code (excerpt):
```python
feats = extract_features(email_text)
X_new = pd.DataFrame([feats.__dict__])
prob_spam = model.predict_proba(X_new)[0][1]
pred = model.predict(X_new)[0]
```

Run:
```bash
python -m src.app predict --model models.joblib
```

## 5) Manually composed SPAM email (should be classified as spam)
**Email text (spam):**
```
SUBJECT: URGENT — You are a WINNER! Claim your FREE BONUS now

CONGRATULATIONS! You have been selected for an exclusive limited offer.
Click now to claim your prize and get FREE cash bonus today.

Verify your account immediately to receive your money:
https://example-bonus-claim.com/verify
www.fast-cash-deals.com

ACT NOW — offer expires soon!
```

**Why it should be spam:**
- Contains many spammy keywords (FREE, WINNER, URGENT, claim, prize, bonus, money, verify, act now).
- Includes multiple links.
- Uses ALL-CAPS words to create pressure/urgency.

## 6) Manually composed LEGITIMATE email (should be classified as legitimate)
**Email text (legitimate):**
```
Subject: Meeting notes and next steps for Tuesday

Hi team,

Thanks for today’s discussion. Attached are the notes and the action items:
1) Finalize the sprint tasks
2) Confirm the deployment window
3) Update the documentation

No urgent action is needed tonight. We will review progress on Tuesday.

Best regards,
Giorgi
```

## 7) Visualizations (2 required)

### Visualization 1 — Class distribution
Python code:
```python
counts = y.value_counts().sort_index()
plt.bar(["Legitimate (0)", "Spam (1)"], counts.values)
plt.title("Class Distribution: Legitimate vs Spam")
plt.xlabel("Class")
plt.ylabel("Number of Emails")
plt.tight_layout()
plt.savefig("plots/class_distribution.png", dpi=200)
```

Explanation (2–3 sentences):
This plot shows how many spam and legitimate emails exist in the dataset. Here the classes are close to balanced, so the model is not strongly biased by class imbalance.

### Visualization 2 — Confusion matrix heatmap
Python code:
```python
plt.imshow(cm, interpolation="nearest")
plt.title("Confusion Matrix Heatmap")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.xticks([0,1], ["Legitimate (0)","Spam (1)"], rotation=30, ha="right")
plt.yticks([0,1], ["Legitimate (0)","Spam (1)"])
for (i,j), val in np.ndenumerate(cm):
    plt.text(j, i, str(val), ha="center", va="center")
plt.colorbar()
plt.tight_layout()
plt.savefig("plots/confusion_matrix_heatmap.png", dpi=200)
```

Explanation (2–3 sentences):
The heatmap visualizes where the model makes correct predictions (diagonal cells) and mistakes (off-diagonal cells). In this run, false negatives (spam predicted as legitimate) and false positives are relatively small compared to correct classifications, matching the ~95% accuracy.
