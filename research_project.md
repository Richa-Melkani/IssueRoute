# IssueRoute: Intelligent Student Complaint Classification and Department Routing

## 📌 Project Overview

**IssueRoute** is an NLP-based student complaint classification and routing system designed to automatically analyze student complaints and identify the department(s) responsible for handling them.

The system uses **Natural Language Processing (NLP)** and **Machine Learning** techniques to convert complaint text into numerical features and classify complaints into relevant categories.

The research focuses on an important challenge in complaint management: a single complaint may contain multiple issues requiring attention from different departments.

---

## 🎯 Research Objectives

The main objectives of this project are:

* To classify student complaints into appropriate departments.
* To identify whether a complaint is **single-issue, multi-issue, or unknown**.
* To detect complaints that require multiple departments.
* To investigate the limitations of traditional TF-IDF-based classification.
* To analyze uncertain and ambiguous predictions.
* To explore human verification for uncertain complaints.
* To investigate multilingual complaints involving English, Hindi, and Hinglish.
* To study explainability and understand why classification errors occur.

---

## 🧠 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TF-IDF Vectorization
* Linear Support Vector Machine (Linear SVM)
* One-vs-Rest Classification
* NLP
* Jupyter Notebook / Google Colab
* GitHub

---

## 📂 Project Structure

```text
IssueRoute/
│
├── dataset/
│   ├── complaints.csv
│   └── research_dataset.csv
│
├── model_analysis/
│   └── train_model.py
│
├── models/
│   └── issue_route_model.pkl
│
└── README.md
```

---

## 📊 Datasets

### Original Dataset

The original experiment uses a synthetic **University Complaint Triage Dataset** containing **800 complaint records**.

It contains:

* Complaint text
* Department
* Urgency

The dataset represents eight university departments:

1. Academics
2. Administration
3. Fees / Finance
4. Hostels
5. IT Support
6. Maintenance
7. Security / Discipline
8. Transport

### Research Dataset

A separate research dataset was created to investigate multi-issue and multilingual complaint routing.

It contains:

* Complaint ID
* Complaint text
* Primary department
* Secondary department
* Complaint type
* Language

The research dataset contains **262 complaints**:

* Single: 159
* Multi: 83
* Unknown: 20

Languages:

* English: 118
* Hinglish: 74
* Hindi: 70

---

# 🔬 Experiments and Results

## Experiment 1 — Complaint Type Classification

The first experiment classified complaints into:

* Single
* Multi
* Unknown

### Results

* Accuracy: **96.23%**
* Macro F1: **0.932**

This indicates that TF-IDF with Linear SVM can reliably distinguish different complaint types.

---

## Experiment 2 — Multi-Issue Detection

A binary classification experiment was performed to distinguish:

* Multi-issue
* Not multi-issue

### Results

* Accuracy: **98.11%**
* Macro F1: **0.979**

Only one complaint was incorrectly classified in the test set.

This demonstrates that detecting whether a complaint contains multiple issues is considerably easier than determining all departments responsible for those issues.

---

## Experiment 3 — Primary Department Classification

The complaint was classified into one of the eight departments plus an Unknown category.

### Results

* Accuracy: **77.36%**
* Macro F1: **0.769**

The model performed particularly well for categories such as Transport, Hostels, and Unknown, while Maintenance and some overlapping categories were more difficult.

A recurring confusion was observed between:

**Hostels ↔ Maintenance**

---

## Experiment 4 — Unknown Complaint Detection

This experiment classified complaints as:

* Known
* Unknown

### Results

* Accuracy: **98.11%**
* Macro F1: **0.924**

This indicates that the model was effective at distinguishing complaints outside the defined departmental scope.

---

## Experiment 5 — Language-wise Department Classification

Department classification was evaluated separately for English, Hindi, and Hinglish complaints.

### Results

| Language | Accuracy |
| -------- | -------: |
| English  |   81.82% |
| Hindi    |   61.54% |
| Hinglish |   83.33% |

The results should be interpreted cautiously because the language-specific test sets were small.

---

## Experiment 6 — Secondary Department Prediction

For multi-issue complaints, the model was tested for predicting the secondary responsible department.

### Results

* Accuracy: **58.82%**
* Macro F1: **0.449**

This was substantially lower than primary department classification.

The result indicates that identifying the **secondary responsible department** is a challenging problem because multiple department-specific terms can occur within the same complaint.

---

## Experiment 7 — Uncertainty-Aware Routing

SVM decision-score margins were used to identify potentially uncertain predictions.

Complaints with smaller margins were considered more uncertain and suitable for human verification.

At an exploratory threshold of 0.3:

* Complaints reviewed: **15 / 53**
* Review percentage: **28.30%**
* Errors caught in simulation: **7 / 12**
* Simulated corrected accuracy: **90.57%**
* Simulated Macro F1: **0.912**

These are **simulated human-verification results**, not actual human-review performance.

---

## Experiment 8 — Character TF-IDF

Character-level TF-IDF was tested as an alternative to word-level TF-IDF.

### Results

* Accuracy: **75.47%**
* Macro F1: **0.749**

Word-level TF-IDF performed better overall on the research split.

Therefore, character TF-IDF was retained as an experimental comparison rather than the primary approach.

---

## Experiment 9 — Threshold Sensitivity Analysis

Different uncertainty thresholds were tested for human-verification routing.

| Threshold |   Review % | Simulated Accuracy |  Macro F1 |
| --------: | ---------: | -----------------: | --------: |
|       0.1 |      9.43% |             84.91% |     0.848 |
|       0.2 |     18.87% |             88.68% |     0.892 |
|       0.3 |     28.30% |             90.57% |     0.912 |
|   **0.4** | **32.08%** |         **92.45%** | **0.934** |
|       0.5 |     35.85% |             92.45% |     0.934 |

Among the tested thresholds, **0.4 provided the best practical trade-off on this experimental split**, achieving the same simulated performance as 0.5 while requiring fewer reviews.

---

## Experiment 10 — Explainability Analysis

Feature-level analysis was performed to understand which words contributed strongly to department predictions.

Examples showed that contextual words can strongly influence predictions.

For example, complaints containing the word **"hostel"** were more likely to be classified as Hostels even when the actual issue was related to Maintenance.

This helps explain the recurring:

**Hostels ↔ Maintenance**

confusion.

---

## Experiment 11 — Multilingual Preprocessing Comparison

Character-level and word-level TF-IDF approaches were compared across English, Hindi, and Hinglish complaints.

Character TF-IDF showed some improvement for Hinglish but performed worse overall.

This suggests that character-level features alone are not sufficient to solve multilingual complaint routing.

---

# 🧩 Experiment 12 — Multi-Department Routing

A multi-label classification approach was developed for complaints requiring multiple departments.

Instead of forcing every complaint into one department, each complaint can have multiple department labels.

### Multi-label model

* TF-IDF
* One-vs-Rest classification
* Linear SVM

### Results at default threshold

* Subset Accuracy: **35.29%**
* Hamming Loss: **0.1176**
* Macro F1: **0.724**

Subset accuracy is strict: **every department label must be predicted correctly**.

### Threshold analysis

The best tested threshold was **−0.2**:

* Subset Accuracy: **47.06%**
* Hamming Loss: **0.1176**
* Macro F1: **0.776**
* False Positives: **8**
* False Negatives: **8**

The threshold was selected only as the best-performing value among the tested thresholds on this experimental split.

---

# 🔎 Key Research Findings

### 1. Multi-issue detection is highly feasible

The model achieved very high performance in identifying whether a complaint contains multiple issues.

### 2. Exact multi-department routing is considerably harder

Although multi-issue detection achieved 98.11% binary accuracy, assigning **all responsible departments** remained challenging.

### 3. Secondary department prediction is a major challenge

The model frequently identified the dominant department while missing the secondary responsible department.

### 4. Contextual words can influence routing

Terms such as "hostel" can dominate issue-specific words, contributing to confusion between Hostels and Maintenance.

### 5. Administration is frequently under-predicted

In multi-label routing, Administration showed high precision but very low recall, indicating that the model often missed Administration when it was one of several responsible departments.

### 6. Uncertainty-aware routing is promising

Decision-score-based human verification can potentially improve routing outcomes by sending uncertain complaints for manual review.

The current results are simulated and require validation with real human reviewers.

---

# 🚧 Research Limitations

The current research has several limitations:

* The datasets are relatively small.
* The original dataset is synthetic.
* The research dataset contains limited examples for some department combinations.
* Multilingual samples are limited.
* Multi-label evaluation uses a small test set.
* Threshold values were evaluated on an experimental split and should not be treated as globally optimal.
* Simulated human verification assumes perfect human correction.
* The current NLP approach relies primarily on TF-IDF features and Linear SVM.
* Real-world complaints may contain spelling errors, mixed languages, sarcasm, incomplete information, and multiple complex issues.

---

# 🚀 Future Work

Future development can focus on:

* Larger real-world complaint datasets.
* Improved multilingual and Hinglish processing.
* Multi-issue and multi-department routing.
* Unknown and out-of-domain complaint detection.
* Better uncertainty estimation.
* Human-in-the-loop verification.
* Explainable complaint routing.
* Transformer-based NLP models.
* Flask-based backend integration.
* Web-based student complaint interface.
* Administrative dashboard.
* Complaint tracking and status management.
* Real-world evaluation with human reviewers.

---

# 🏗️ Proposed System Architecture

```text
Student
   ↓
Complaint Input
   ↓
Language Detection
   ↓
Text Preprocessing
   ↓
Multi-Issue Detection
   ↓
Single / Multi / Unknown
   ↓
Department Prediction
   ↓
Uncertainty Check
   ↓
 ┌───────────────────────┐
 │                       │
Confident            Uncertain
 │                       │
 ↓                       ↓
Auto Routing       Human Verification
 │                       │
 └───────────┬───────────┘
             ↓
        Explanation
             ↓
     Complaint Dashboard
```

---

# 📌 Main Research Contribution

The project investigates an **uncertainty-aware, multi-issue student complaint routing approach** rather than treating every complaint as a simple single-class classification problem.

The key research direction is to identify:

> **Whether a complaint can be automatically routed with confidence, whether it requires multiple departments, and when human verification should be requested.**

---

# 👩‍💻 Project Status

**Current status:** Research experimentation completed for the initial NLP and multi-department routing stages.

The project is currently focused on analyzing experimental findings and preparing the research paper.

---

## 📜 License

This project is intended for academic and research purposes.
