# IssueRoute – Research Experiment Notes

## 1. Dataset

Dataset: University Complaint Triage Dataset

- Total records: 800
- Features/columns:
  - text
  - department
  - urgency
- Departments: 8
- Each department has 100 complaints.
- Dataset is synthetic but realistic university complaint data.
- Current task: Department classification.
- Urgency is not used in the current routing model.

---

## 2. Input and Target

Input (X):
- Complaint text

Target (y):
- Department

Therefore, the model learns to predict the appropriate department from the complaint text.

---

## 3. Train-Test Split

Used:
- 80% training data
- 20% testing data
- Training samples: 640
- Testing samples: 160
- random_state = 42
- stratify = y

Reason:
- Training data is used to learn patterns.
- Testing data is used to evaluate the model on unseen complaints.
- stratify=y keeps department proportions similar in training and testing data.

---

# Experiment 1: Normal TF-IDF + Linear SVM

## TF-IDF

Used:
TfidfVectorizer()

Training TF-IDF shape:
(640, 3674)

Testing TF-IDF shape:
(160, 3674)

Meaning:
- 640 training complaints
- 160 testing complaints
- 3674 learned text features

Important:
- fit_transform() was used on training data.
- transform() was used on testing data.
- This prevents information from the test set being used while learning the vocabulary.

## Model

Model:
LinearSVC()

The model was trained using:
- X_train_tfidf
- y_train

Predictions:
- y_pred

## Result

Accuracy:
69.375% ≈ 69.38%

Macro F1:
≈ 0.69

## Classification observations

Strong classes:
- IT Support: F1 = 0.98
- Transport: F1 = 0.92

Weak classes:
- Hostels: F1 = 0.28
- Maintenance: F1 = 0.38

Main problem:
- Hostels and Maintenance complaints are frequently confused.

---

# Confusion Matrix – Normal TF-IDF

Class order:

1. Academics
2. Administration
3. Fees / Finance
4. Hostels
5. IT Support
6. Maintenance
7. Security / Discipline
8. Transport

Main observations:

- Actual Hostels → Maintenance: 9
- Actual Maintenance → Hostels: 8
- IT Support: 20/20 correctly classified

Main confusion:
Hostels ↔ Maintenance

Possible reason:
Both departments can contain similar facility/location-related words and contexts.

---

# Experiment 2: Bigram TF-IDF + Linear SVM

## Bigram TF-IDF

Used:

TfidfVectorizer(ngram_range=(1, 2))

Meaning:
- 1 = unigram
- 2 = bigram
- Model uses both individual words and two-word combinations.

Example:
- hostel
- room
- hostel room
- water
- leakage
- water leakage

Training shape:
(640, 18456)

Testing shape:
(160, 18456)

The number of features increased because both single words and two-word combinations are considered.

## Model

Model:
LinearSVC()

## Result

Bigram Accuracy:
71.25%

Improvement over normal TF-IDF:
71.25% - 69.38% = 1.87 percentage points

---

## Bigram Classification Results

F1-scores:

- Academics: 0.73
- Administration: 0.83
- Fees / Finance: 0.77
- Hostels: 0.25
- IT Support: 0.98
- Maintenance: 0.48
- Security / Discipline: 0.76
- Transport: 0.85

Observations:

- Academics improved.
- Administration improved.
- Maintenance improved.
- Hostels became slightly worse.
- Overall accuracy improved from 69.38% to 71.25%.

Conclusion:

Adding bigram features improved overall classification performance, but the Hostels–Maintenance confusion remained a major challenge.

---

# Research Observation So Far

The experiments show that:

1. TF-IDF can convert complaint text into numerical features for ML classification.
2. Linear SVM can classify student complaints into departments.
3. Adding bigram features improves overall accuracy.
4. However, semantically related departments such as Hostels and Maintenance remain difficult to distinguish.
5. This suggests that simple word-level features may not be sufficient for handling contextual or ambiguous complaints.

---

# Important Results Table

| Experiment | Accuracy |
|------------|----------|
| Normal TF-IDF + Linear SVM | 69.38% |
| Bigram TF-IDF + Linear SVM | 71.25% |


Experiment 3: Character TF-IDF + Linear SVM

Technique:
Character-level TF-IDF with n-grams from 3 to 5 characters.

Settings:
analyzer = "char"
ngram_range = (3, 5)
Classifier = LinearSVC

Result:
Accuracy = 77.50%
Macro F1 = 0.77

Observation:
Character TF-IDF performed better than normal word-level TF-IDF
(69.38%) and word-level bigram TF-IDF (71.25%).

Strong classes:
IT Support (F1 = 0.98)
Administration (F1 = 0.92)
Transport (F1 = 0.89)
Security / Discipline (F1 = 0.87)

Weak classes:
Hostels (F1 = 0.39)
Maintenance (F1 = 0.50)

Key finding:
Character-level features significantly improved overall classification,
but confusion between Hostels and Maintenance still remains.

Experiment 4: Balanced Linear SVM

Technique:
Linear SVM with class_weight="balanced".

Result:
Accuracy = 69.38%

Observation:
Class weighting did not improve the overall accuracy compared
with the normal TF-IDF + Linear SVM experiment (69.38%).
Therefore, balanced class weighting was not beneficial for
this dataset.

Macro F1:
0.6907 ≈ 0.69

Final observation:
Balanced class weighting did not improve either accuracy or Macro F1.
Therefore, it was not selected as the best approach.



Experiment 5: Word + Character TF-IDF

Technique:
Combined word-level TF-IDF and character-level TF-IDF features
using scipy.sparse.hstack.

Result:
Accuracy = 73.75%

Observation:
Combining word and character features did not improve performance.
Character TF-IDF alone performed better at 77.50%.
Therefore, the combined feature representation was not selected.


Macro F1:
0.7328 ≈ 0.73

Final observation:
The combined word + character TF-IDF representation achieved
73.75% accuracy and 0.73 Macro F1, which was lower than
Character TF-IDF alone (77.50% accuracy, 0.77 Macro F1).

Therefore, combining the two feature representations was
not selected as the best approach.



Hyperparameter Tuning – C Parameter

C = 0.1 → 72.50%
C = 0.5 → 75.625%
C = 1.0 → 77.50%
C = 2.0 → 76.25%

Observation:
The default C=1.0 produced the highest accuracy among the
tested values. Lower and higher C values did not improve
the Character TF-IDF + Linear SVM model.



Error Analysis – Character TF-IDF

Total incorrect predictions:
36 out of 160 test samples.

Major error pair:
Hostels → Maintenance: 7 errors
Maintenance → Hostels: 5 errors

Combined Hostels–Maintenance confusion:
12 out of 36 errors (33.3%).

Other notable error pairs:
Maintenance → Security / Discipline: 3
Hostels → Academics: 3
Academics → Maintenance: 3

Key finding:
The largest source of classification error is the overlap between
Hostels and Maintenance complaints. Location and facility-related
terms can make it difficult for the model to distinguish the
responsible department from the complaint context.


Character-level TF-IDF combined with Linear SVM achieved the highest performance among the tested classifiers. Logistic Regression performed moderately, while Multinomial Naive Bayes performed considerably lower.



Human Verification Threshold Experiment:
Decision-score margins were evaluated at thresholds of 0.1, 0.2, 0.3, 0.4, and 0.5. Increasing the threshold increased the proportion of complaints sent for human verification and improved error detection. A threshold of 0.3 provided a practical trade-off, flagging 43.75% of complaints for human verification while detecting 75.00% of model errors.



Simulated Human-in-the-Loop Performance:
At a decision-score margin threshold of 0.3, 43.75% of test complaints were flagged for human verification and 75.00% of model errors were detected. Assuming all flagged errors were correctly corrected by human reviewers, the simulated system accuracy increased from 77.50% to 96.875%. This represents a simulated upper-bound performance and not the standalone ML model accuracy.



New Complaint Prediction: The trained Character TF-IDF + Linear SVM model successfully generated department predictions for previously unseen complaint texts.



# IssueRoute: Research Progress and Observations

## 1. Research Dataset Preparation

A research-oriented dataset was created to extend the original student complaint classification problem. The dataset contains the following fields:

* `text` — complaint text
* `primary_department` — main responsible department
* `secondary_department` — second department for multi-department complaints
* `complaint_type` — single, multi, or unknown
* `language` — English, Hindi, or Hinglish

The dataset currently contains **262 complaint records**.

### Dataset Distribution

**Complaint Type**

* Single: 159
* Multi: 83
* Unknown: 20

**Language**

* English: 118
* Hinglish: 74
* Hindi: 70

**Primary Department**

* Hostels: 39
* Maintenance: 35
* Transport: 33
* Administration: 29
* IT Support: 28
* Fees / Finance: 28
* Academics: 25
* Security / Discipline: 25
* Unknown: 20

No duplicate rows were found.

### Observation

The research dataset extends the original dataset by introducing **complaint type, secondary department, and language information**. This makes it possible to investigate problems beyond simple department classification, particularly multi-department complaints and complaints that cannot be assigned to the known departments.

---

# 2. Research Experiment 1 — Complaint Type Classification

### Objective

The first research experiment was designed to determine whether a machine learning model can classify a complaint into:

* Single
* Multi
* Unknown

### Method

The following pipeline was used:

**Complaint Text → Train/Test Split → TF-IDF → Linear SVM → Complaint Type Prediction**

The dataset was divided using an **80:20 stratified split**:

* Training samples: 209
* Testing samples: 53

TF-IDF was fitted only on the training data and then used to transform the test data.

The resulting TF-IDF representation contained **474 features**.

A **Linear SVM (LinearSVC)** classifier was trained on the TF-IDF features.

---

# 3. Experimental Results

| Metric         |     Result |
| -------------- | ---------: |
| Accuracy       | **96.23%** |
| Macro F1-score | **0.9322** |
| Test samples   |     **53** |

### Classification Report

| Class   | Precision | Recall | F1-score |
| ------- | --------: | -----: | -------: |
| Multi   |      1.00 |   0.94 |     0.97 |
| Single  |      0.94 |   1.00 |     0.97 |
| Unknown |      1.00 |   0.75 |     0.86 |

---

# 4. Observations

### Observation 1 — Strong overall classification

The Linear SVM achieved **96.23% accuracy** and a **0.9322 Macro F1-score** on the test set, indicating strong initial performance in distinguishing single, multi, and unknown complaints.

### Observation 2 — Single complaints were classified very well

The model achieved **100% recall** for the Single class. All 32 single complaints in the test set were correctly identified.

### Observation 3 — Multi complaints were also classified effectively

The model correctly identified **16 out of 17** multi complaints. One multi complaint was incorrectly classified as single.

### Observation 4 — Unknown complaints are more challenging

The Unknown class achieved **75% recall**. One of the four unknown complaints in the test set was classified as single.

This indicates that identifying complaints outside the known department categories may require further research.

### Observation 5 — Main confusion

The confusion matrix showed only two classification errors:

```text
                Predicted
              Multi  Single  Unknown

Actual Multi    16      1       0
Actual Single    0     32       0
Actual Unknown   0      1       3
```

The main observed confusion was between **Single and Unknown** complaints.

### Observation 6 — Result should be treated as an initial experiment

Although the accuracy is high, the test set contains only **53 complaints**, including only **4 Unknown complaints**. Therefore, these results should be considered an **initial research finding** rather than a final measure of real-world performance.

A larger and more diverse dataset should be evaluated in future experiments.
