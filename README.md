# Replication Package – BERT-based Aspect Classification with Span Attention

Tested Environment

Python 3.10
PyTorch 2.1
Transformers 4.37
Operating System: macOS / Linux

---

# Overview

This replication package contains the implementation of the model used in our paper for **aspect classification using BERT with span-attention over entity tokens**.

The model combines:

* `[CLS]` embedding (sentence representation)
* entity span representation (tokens between `[A]` and `[/A]`)

The entity representation is computed using **attention pooling over the entity span tokens**.

The concatenated representation is passed through fully connected layers for **binary classification**.

---

# 1. Requirements

Python version:

```
Python 3.8+
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# 2. Dataset Format

The dataset format while executing the code must be a CSV file with the following columns:

| Column                  | Description                                    |
| ----------------------- | ---------------------------------------------- |
| `Sentences_withTaggedA` | Sentence containing entity tags `[A] ... [/A]` |
| `label`                 | Binary label (0 or 1)                          |

Example:

```
Sentence,label
"The API shows [A]poor usability[/A] when handling large files.",1
```

The dataset should be placed in "data" folder and correct path needs to be provided in the src/run_model.py

---

# 3. Running the Experiment

Run the model using:

```
cd src
python run_model.py
```

The script performs:

* BERT tokenization
* span-attention over entity tokens
* training with **5-fold cross-validation**
* evaluation using **classification report**
* prediction export for each fold

---

# 4. Output

Results will be saved in:

```
results/
```

Generated files include:

```
results/
 ├── fold1/
 │   ├── model_fold1.pt
 │   ├── training_sentences_fold1.csv
 │   └── ep1/
 │       └── predictions_fold1.csv
 │
 ├── fold2/
 ├── fold3/
 ├── fold4/
 ├── fold5/
 │
 ├── summary_all_folds_epochs.csv
 └── summary_avg_over_folds_by_epoch.csv
```

These files contain:

* predictions for each epoch
* classification reports
* averaged performance metrics across folds

---

# 5. Model Details

Base model:

```
bert-base-uncased
```

Architecture:

```
[CLS] embedding
        +
Entity span attention embedding
        ↓
Concatenation
        ↓
Fully Connected Layer
        ↓
ReLU
        ↓
Dropout
        ↓
Classification Layer
```

---

# 6. Hardware

The experiments automatically use GPU if available:

```
cuda if torch.cuda.is_available()
```

Otherwise they run on CPU.

---

# 7. Reproducibility

Random seeds are fixed for:

```
torch
numpy
```

Cross-validation:

```
5-fold KFold
```

---

# 8. Contact

For questions regarding this replication package, please contact the authors.
