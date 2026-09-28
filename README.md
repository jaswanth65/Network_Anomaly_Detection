# 🌲 Network Anomaly Detection — Random Forest

A complete pipeline for classifying network traffic as **Normal** or as one of four **attack categories** (DoS, Probe, Privilege Escalation, Access) using a **Random Forest** classifier on the **NSL-KDD** dataset — with a proper train / validation / test split, per-class evaluation, and a saved, reusable model.

This repo walks through the full journey: raw network connections → encoded features → trained model → evaluated, saved classifier.

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [How It Works](#-how-it-works)
- [Preprocessing Pipeline](#-preprocessing-pipeline)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Future Improvements](#-future-improvements)
- [Acknowledgments](#-acknowledgments)

---

## 🔍 Overview

Malicious network activity — flooding, port scanning, password guessing, privilege abuse — hides inside enormous volumes of normal traffic. This project builds a detector using a **Random Forest**, an algorithm well suited to this problem because it handles complex, high-dimensional, mixed-type data, needs no feature scaling, and resists overfitting.

Instead of hand-writing rules for what an attack "looks like," the model learns from labeled examples which combinations of connection features (bytes transferred, error rates, service type, ...) point to each kind of traffic.

The model predicts one of five classes for every connection:

| Label | Class                | Meaning                                              |
| :---: | -------------------- | ---------------------------------------------------- |
| `0`   | Normal               | Regular traffic                                      |
| `1`   | DoS                  | Denial of Service — floods a target to knock it offline |
| `2`   | Probe                | Scans the network hunting for vulnerabilities        |
| `3`   | Privilege Escalation | Tries to gain unauthorized admin-level control       |
| `4`   | Access               | Tries to breach system access controls               |

> ⚠️ **A note on terminology:** the project is called "anomaly detection", but the model is a **supervised multi-class classifier** — it learns from rows labeled with their true class, including specific attack types. It does *not* learn "normal only" behavior and flag deviations (that would be one-class / unsupervised anomaly detection). A binary target (`attack_flag`: normal vs. attack) is also created in the notebook as a simpler starting point, but the model is trained and evaluated on the 5-class target.

---

## 📊 Dataset

The project uses **NSL-KDD**, a widely used intrusion-detection benchmark containing labeled normal and malicious connections. Each row is one network connection with 41 features plus an `attack` label and a difficulty `level`. The notebook uses the combined file `KDD+.txt` (no header row, so column names are supplied manually).

**Key stats:**

| Property           | Value                                                            |
| ------------------ | ---------------------------------------------------------------- |
| Total rows         | 148,517                                                          |
| Original columns   | 43 (45 after adding the two targets)                             |
| Missing values     | None                                                             |
| Column types       | 24 integer, 15 float, 4 text (`protocol_type`, `service`, `flag`, `attack`) |

The classes are **heavily imbalanced** (class mix of the test split shown below), which matters a lot for how the model must be evaluated (see [Results](#-results)):

| Class     | Test rows | Share  |
| --------- | --------: | -----: |
| Normal    | 15,402    | 51.9%  |
| DoS       | 10,721    | 36.1%  |
| Probe     | 2,796     | 9.4%   |
| Access    | 764       | 2.6%   |
| Privilege | 21        | 0.07%  |

### Feature groups

| Group             | Examples                                             | Description                                                                 |
| ----------------- | ---------------------------------------------------- | --------------------------------------------------------------------------- |
| **Basic**         | `duration`, `src_bytes`, `dst_bytes`                 | Taken directly from a single TCP/UDP connection                             |
| **Content**       | `hot`, `num_failed_logins`, `root_shell`, `num_root` | Behavior *inside* the connection                                            |
| **Traffic**       | `serror_rate`, `dst_host_srv_diff_host_rate`         | Statistics over a recent window (last 2 seconds or last 100 connections)    |

Two traffic features are especially informative:

- **`serror_rate`** — share of recent connections with a **SYN error** (connection started, three-way handshake never completed). High values strongly indicate a **SYN-flood DoS** or aggressive port scan.
- **`dst_host_srv_diff_host_rate`** — over the last 100 connections to the same host, the share hitting the *same service* from *different source machines*. Low = one user repeatedly using a service; high = many machines at once, which is normal for a public web server but may also signal a **DDoS**.

---

## 🧠 How It Works

### Decision trees

A decision tree learns simple yes/no rules from data. *Example:* to decide whether to play tennis it asks "Is it sunny? Is it windy? Is it humid?" and follows the answers to a decision.

| Component      | Role                                                        |
| -------------- | ----------------------------------------------------------- |
| **Root node**  | Starting point, holds the entire dataset                    |
| **Internal nodes** | Test a feature (e.g. `serror_rate > 0.5`) and branch    |
| **Leaf nodes** | Terminal nodes holding the final prediction                 |

At each node the tree picks the feature and threshold that best separates the classes, measured by **Gini impurity**, **entropy**, or **information gain**. A single tree tends to **overfit** — it memorizes the training data and stumbles on new data.

### Random forests

A random forest is an **ensemble** of many decision trees (100 by default in scikit-learn). Instead of trusting one tree, the whole "crowd" votes:

- **Classification** → each tree votes; the **majority** wins.
- **Regression** → predictions are **averaged**.

Two layers of randomness make the trees different from one another, which is what makes the ensemble strong:

**1. Bootstrapping (row randomness)** — each tree trains on a random sample of the rows drawn *with replacement*.

> Example with 5 rows: Tree 1 draws rows `2, 2, 5, 1, 1` — rows 3 and 4 were never picked, so they act as unseen "out-of-bag" test data for that tree. Tree 2 draws `3, 4, 4, 5, 3`, leaving rows 1 and 2 out.

**2. Feature randomness (column randomness)** — at *every split*, a tree may only consider a random subset of features, typically `m = √p` for classification.

> Example with 4 features (Age, Income, Credit Score, Distance to Store): `m = √4 = 2`. At the root the tree might see only Age and Credit Score and pick Age. At the next node the pool resets and it might see only Income and Distance.
>
> **Why?** If Income were overwhelmingly predictive, every unrestricted tree would split on it first, yielding a forest of near-identical trees that all make the same mistakes. Restricting features forces diversity.

With this project's **107 input features**, each split considers roughly `√107 ≈ 10` of them.

---

## 🧹 Preprocessing Pipeline

The goal is to turn raw connection records into a numeric matrix the model can learn from.

### 1. Binary target

`attack_flag` = `0` for normal, `1` for any attack.

```python
df['attack_flag'] = df['attack'].apply(lambda a: 0 if a == 'normal' else 1)
```

### 2. Multi-class target

Individual attack names are grouped into four categories by `map_attack()`, producing `attack_map`:

| Class             | Attacks mapped to it                                                                                                                                        |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **DoS (1)**       | apache2, back, land, neptune, mailbomb, pod, processtable, smurf, teardrop, udpstorm, worm                                                                  |
| **Probe (2)**     | ipsweep, mscan, nmap, portsweep, saint, satan                                                                                                               |
| **Privilege (3)** | buffer_overflow, loadmodule\*, perl, ps, rootkit, sqlattack, xterm                                                                                          |
| **Access (4)**    | ftp_write, guess_passwd, http_tunnel, imap, multihop, named, phf, sendmail, snmpgetattack, snmpguess, spy, warezclient, warezmaster, xclock, xsnoop          |
| **Normal (0)**    | `normal`, plus any attack name not listed above (default)                                                                                                   |

```python
def map_attack(attack):
    if attack in dos_attacks:         return 1
    elif attack in probe_attacks:     return 2
    elif attack in privilege_attacks: return 3
    elif attack in access_attacks:    return 4
    else:                             return 0

df['attack_map'] = df['attack'].apply(map_attack)
```

> ⚠️ **Known issue:** the notebook's `privilege_attacks` list contains the typo `'loadmdoule'` (should be `'loadmodule'`), so those attacks fall through to the default and are silently labeled **Normal**. Similarly, run `df['attack'].unique()` and confirm every attack name in your file matches a list entry exactly (e.g. `http_tunnel` vs `httptunnel`, `xclock` vs `xlock`). Results in this README reflect the notebook *as run*, before this fix.

### 3. One-hot encode categorical features

Models need numbers, so `protocol_type` and `service` become 0/1 indicator columns (3 protocol + 70 service = 73 columns):

```
protocol_type   service               protocol_type_tcp  protocol_type_udp  service_http  service_dns  service_ftp
tcp             http        --->             1                  0                1             0            0
udp             dns                          0                  1                0             1            0
tcp             ftp                          1                  0                0             0            1
```

```python
features_to_encode = ['protocol_type', 'service']
encoded = pd.get_dummies(df[features_to_encode])
```

### 4. Select numeric features

34 numeric columns (basic, content, and traffic features) are kept. The `level` difficulty score is deliberately **excluded** — it is metadata about the label, not something observable on live traffic.

### 5. Combine into the final feature matrix

```python
train_set = encoded.join(df[numeric_features])   # 73 + 34 = 107 columns
multi_y   = df['attack_map']
```

### 6. Split into train / validation / test

Data is split in two stages so the test set stays untouched until the very end:

```python
# Stage 1 — lock away 20% as the final test set
train_X, test_X, train_y, test_y = train_test_split(
    train_set, multi_y, test_size=0.2, random_state=1337
)

# Stage 2 — split the remaining 80% into training and validation
multi_train_X, multi_val_X, multi_train_y, multi_val_y = train_test_split(
    train_X, train_y, test_size=0.3, random_state=1337
)
```

| Set        | Variables                    | Rows   | Purpose                            |
| ---------- | ---------------------------- | -----: | ---------------------------------- |
| Training   | `multi_train_X/y`            | 83,169 | Fit the model                      |
| Validation | `multi_val_X/y`              | 35,644 | Check and tune the model           |
| Test       | `test_X/y`                   | 29,704 | Final, unbiased score (used once)  |

`random_state=1337` fixes the shuffle so everyone gets the identical split; without it, each run gives different rows and different scores.

> **The workflow:** train on the training set → score on the validation set → tweak settings → repeat → only when finished, run the test set **once**. Peeking at the test set repeatedly would leak information and inflate the final score.

---

## 🤖 Model Training & Evaluation

### Training

```python
rf_model_multi = RandomForestClassifier(random_state=1337)
rf_model_multi.fit(multi_train_X, multi_train_y)
```

Default hyperparameters are used (100 trees, `max_features='sqrt'`); `random_state=1337` makes training reproducible. **No hyperparameter tuning was performed** in this version.

### Evaluation

Predictions on the validation and test sets are scored with:

| Metric               | Meaning                                                                                       |
| -------------------- | --------------------------------------------------------------------------------------------- |
| **Accuracy**         | Share of all predictions that were correct                                                    |
| **Precision**        | Of everything predicted as class X, how much really was X                                     |
| **Recall**           | Of everything that really was class X, how much was found                                     |
| **F1-Score**         | Harmonic mean of precision and recall                                                         |
| **Confusion matrix** | Rows = actual class, columns = predicted class; shows exactly which classes get confused      |

```python
predictions = rf_model_multi.predict(multi_val_X)

accuracy  = accuracy_score(multi_val_y, predictions)
precision = precision_score(multi_val_y, predictions, average='weighted')
recall    = recall_score(multi_val_y, predictions, average='weighted')
f1        = f1_score(multi_val_y, predictions, average='weighted')

class_labels = ['Normal', 'DoS', 'Probe', 'Privilege', 'Access']
sns.heatmap(confusion_matrix(multi_val_y, predictions), annot=True, fmt='d',
            cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)

print(classification_report(multi_val_y, predictions, target_names=class_labels))
```

`average='weighted'` averages per-class scores weighted by class size. The same code is then run once on `test_X` / `test_y` for the final score.

---

## 📈 Results

### Overall performance (weighted)

| Metric    | Validation | Test   |
| --------- | ---------- | ------ |
| Accuracy  | 0.9950     | 0.9949 |
| Precision | 0.9949     | 0.9947 |
| Recall    | 0.9950     | 0.9949 |
| F1-Score  | 0.9949     | 0.9947 |

### Per-class performance — test set (29,704 rows)

| Class        | Precision | Recall | F1-Score | Support |
| ------------ | --------- | ------ | -------- | ------- |
| Normal       | 0.99      | 1.00   | 1.00     | 15,402  |
| DoS          | 1.00      | 1.00   | 1.00     | 10,721  |
| Probe        | 0.99      | 1.00   | 1.00     | 2,796   |
| Privilege    | 0.62      | 0.24   | 0.34     | 21      |
| Access       | 0.96      | 0.92   | 0.94     | 764     |
| **Accuracy** |           |        | **0.99** | 29,704  |
| Macro avg    | 0.91      | 0.83   | 0.85     | 29,704  |
| Weighted avg | 0.99      | 0.99   | 0.99     | 29,704  |

**Confusion matrix (test set)** — rows are actual, columns are predicted:

|                  | Pred. Normal | Pred. DoS | Pred. Probe | Pred. Privilege | Pred. Access |
| ---------------- | -----------: | --------: | ----------: | --------------: | -----------: |
| **Actual Normal**    | 15,349   | 7         | 16          | 2               | 28           |
| **Actual DoS**       | 11       | 10,708    | 2           | 0               | 0            |
| **Actual Probe**     | 6        | 2         | 2,788       | 0               | 0            |
| **Actual Privilege** | 14       | 0         | 0           | 5               | 2            |
| **Actual Access**    | 59       | 0         | 1           | 1               | 703          |

> **Why look at both macro and weighted averages?** The weighted average is dominated by the huge Normal, DoS, and Probe classes, so it stays at 0.99 no matter what happens to rare classes. The **macro average** treats every class equally and exposes the real weakness: Privilege Escalation (F1 = 0.34) drags macro F1 down to 0.85.

---

## 💡 Key Insights

1. **Very high overall accuracy (~99.5%)** — and validation and test scores are nearly identical (0.9950 vs 0.9949), so the pipeline generalizes to unseen data from the same dataset.
2. **DoS and Probe are detected almost perfectly.** These attacks leave strong statistical fingerprints (e.g. a high `serror_rate` for SYN floods) that the traffic features capture well.
3. **Privilege Escalation is the weak spot.** Only 21 of these samples are in the test set (~0.07% of the data). The model finds just 5 of them (recall 0.24) — too few examples for the trees to learn from.
4. **The costly errors are attacks classified as Normal.** In security, a missed attack usually costs more than a false alarm. On the test set, **59 of 764 Access attacks (~7.7%)** and **14 of 21 Privilege attacks (~67%)** slipped through as Normal, while only **53 of 15,402 normal connections (~0.3%)** were wrongly flagged.
5. **Class imbalance matters more than model choice here.** Improvement should focus on handling the rare classes rather than on a fancier algorithm.

---

## 📁 Project Structure

> Adjust this to match your actual repo layout before publishing.

```
network-anomaly-detection/
├── Network_Anomaly_Detection.ipynb            # Full notebook: preprocessing → training → evaluation
├── Network_Anomaly_Detection.pdf              # Exported notebook with outputs
├── notes/
│   ├── 0_intro.md                             # Theory: decision trees, random forests, NSL-KDD
│   ├── 1_Preprocessing_and_splitting_the_Dataset.md
│   └── 2_Training_and_Evaluation.md
├── KDD+.txt                                   # NSL-KDD dataset (downloaded)
├── network_anomaly_detection_model.joblib     # Saved trained model
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

```bash
git clone https://github.com/<your-username>/network-anomaly-detection.git
cd network-anomaly-detection

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**`requirements.txt`:**

```
numpy
pandas
scikit-learn
seaborn
matplotlib
joblib
requests
jupyter
```

**Download the dataset:**

```python
import requests, zipfile, io

url = "https://cdn.services-k8s.prod.aws.htb.systems/content/modules/292/KDD_dataset.zip"
response = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(response.content))
z.extractall('.')   # extracts into the current directory
```

The notebook expects a file named `KDD+.txt`. If your extracted file has a different name, update `file_path` in the notebook.

---

## 🚀 Usage

### Run the full pipeline

```bash
jupyter notebook Network_Anomaly_Detection.ipynb
```

Run the notebook top to bottom — it covers loading, target creation, encoding, splitting, training, validation, final testing, and saving the model to `network_anomaly_detection_model.joblib`.

### Load the saved model and classify new traffic

```python
import joblib
import pandas as pd

model = joblib.load("network_anomaly_detection_model.joblib")
labels = {0: "Normal", 1: "DoS", 2: "Probe", 3: "Privilege", 4: "Access"}

# new_df: raw connections in the same format as the original data
# (same column names as KDD+.txt, without the target columns)
encoded = pd.get_dummies(new_df[["protocol_type", "service"]])
X_new = encoded.join(new_df).reindex(columns=model.feature_names_in_, fill_value=0)

predictions = model.predict(X_new)
print([labels[p] for p in predictions])
```

> ⚠️ **Critical:** the `reindex` step is required. One-hot encoding creates columns only for the values present, so new data may lack some `service_*` columns the model expects. Reindexing to `model.feature_names_in_` guarantees the exact same 107 columns in the exact same order as training.

---

## 🔮 Future Improvements

- **Fix the label lists** (`loadmodule` typo, verify all attack names) and re-run.
- **Handle class imbalance** — `class_weight='balanced'` in the forest, or oversampling with SMOTE (`imbalanced-learn`) — to lift Privilege and Access recall.
- **Tune hyperparameters** (`n_estimators`, `max_depth`, `min_samples_leaf`) with `GridSearchCV` / `RandomizedSearchCV`.
- **Retrain on the full training split** (train + validation) after tuning; the current model saw only 83,169 rows.
- **Inspect `feature_importances_`** to see which traffic features drive predictions.
- **Add the excluded columns** (`flag`, `land`, `logged_in`, `is_host_login`, `is_guest_login`) and compare.
- **Try other models** (Gradient Boosting, XGBoost) and add an unsupervised detector (Isolation Forest) to catch attack types never seen in training.
- **Optimize for attack recall** or tune the decision threshold, since missed attacks cost more than false alarms.
- Wrap the saved model in a small **Flask/FastAPI** service for real-time inference.

---

## 🙏 Acknowledgments

- **NSL-KDD dataset** — Tavallaee, Bagheri, Lu & Ghorbani (2009), *A Detailed Analysis of the KDD CUP 99 Data Set*. Dataset page: https://www.unb.ca/cic/datasets/nsl.html
- **scikit-learn** for `RandomForestClassifier`, `train_test_split`, and the evaluation metrics.
- **pandas / NumPy** for data handling, and **seaborn / matplotlib** for visualization.
- Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32.

---

## 📄 License

Add a license (e.g. MIT) here if you intend for others to reuse this code.
