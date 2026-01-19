# Financial Chart Pattern Detection with CNN

## 1. Problem Description and Motivation

Financial markets are noisy, highly stochastic, and dominated by periods where **nothing meaningful happens**. Most price movements are random or insignificant, while only a **small fraction of events** correspond to strong, directional moves driven by information, liquidity shocks, or regime changes.

The core problem addressed in this project is:

> **Can a Convolutional Neural Network (CNN) learn to identify rare but meaningful price patterns from financial charts that precede strong market moves?**

Instead of predicting prices at every timestep, this project reframes the task as an **event-detection problem**:

* Detect **high-conviction market situations**
* Accept **few signals**
* Prioritize **quality over quantity**

This philosophy aligns with real-world trading, where missing most days is acceptable if the detected opportunities have a strong statistical edge.

The motivation for this project is both academic and practical:

* Academically, it explores the limits of CNNs applied to financial time series represented as images.
* Practically, it aims to build tools that **support discretionary or systematic trading decisions**, continuing the same line of thinking developed in Capstone Project 1.

---

## 2. Project Overview and Development Process

### 2.1 Data Representation

Instead of feeding raw numerical time series to the model, price data is transformed into **chart images**:

* Each image represents a fixed-length window of historical prices
* Technical indicators (e.g. moving averages) are embedded visually
* The CNN learns **visual market structure**, not hand-crafted rules

This approach mimics how human traders visually interpret charts.

### 2.2 Labeling Strategy

Labels are generated based on **future price behavior**:

* A positive label corresponds to a strong upward move after the chart window
* A negative label corresponds to the absence of such a move

This naturally produces **class imbalance**, which is expected and realistic.

### 2.3 Model Architecture and Training

The core model is a CNN trained using TensorFlow/Keras:

* Multiple convolutional layers
* Pooling layers for spatial abstraction
* Dense layers for classification

Key discoveries during training:

* Accuracy alone is misleading due to class imbalance
* The model tends to output probabilities clustered around 0.5, revealing how difficult the task is
* Performance must be interpreted in the context of **rare-event detection**, not standard classification

### 2.4 Problems, Discoveries, and Successes

**Problems encountered:**

* Extremely noisy labels
* Low signal-to-noise ratio
* CNN outputs collapsing near 0.5

**Discoveries:**

* The model behaves like an *event filter*, not a predictor
* Few signals with controlled drawdowns can still outperform benchmarks

**Successes:**

* Full end-to-end ML pipeline
* Clean project structure
* Reproducible training and evaluation

---

## 3. Project Structure

Below is the high-level structure of the project:

```
financial-chart-pattern-cnn/
│
├── data/
│   ├── raw/
│   ├── processed/
│
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   │   ├── train_cnn.py
│   │   └── evaluate_cnn.py
│   └── visualization/
│
├── models/
│   └── cnn_chart_model.keras
│
├── requirements.txt
└── README.md
```

*(Structure image provided separately during development)*

---

## 4. How to Run the Project Locally (Beginner-Friendly)

### 4.1 Prerequisites

* Python 3.10+
* Git
* Virtual environment tool (venv or conda)

### 4.2 Clone the Repository

```bash
git clone https://github.com/your-username/financial-chart-pattern-cnn.git
cd financial-chart-pattern-cnn
```

### 4.3 Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # macOS / Linux
```

### 4.4 Install Dependencies

```bash
pip install -r requirements.txt
```

### 4.5 Run Training

```bash
python src/models/train_cnn.py
```

### 4.6 Run Evaluation

```bash
python src/models/evaluate_cnn.py
```

Alternatively, you can run the notebooks in order:

1. `01_data_preparation.ipynb`
2. `02_eda.ipynb`
3. `03_training.ipynb`
4. `04_evaluation.ipynb`

---

## 5. Why This Project Does NOT Use Docker

Docker is extremely useful for:

* Production deployment
* Microservices
* Team-based infrastructure

However, this project is:

* **Research-oriented**
* **Educational**
* **Single-user, local execution**

All dependencies are:

* Explicitly listed in `requirements.txt`
* Easily reproducible via a virtual environment

Adding Docker would increase complexity without providing meaningful benefits for the intended use case. Therefore, the project consciously prioritizes **clarity and simplicity** over containerization.

---

## 6. Why the Model Is Not Deployed

Model deployment is intentionally excluded because:

* The model is **experimental**, not production-ready
* Outputs require **contextual interpretation**, not automated execution
* Financial decision systems carry real-world risk

This project focuses on:

* Feature engineering
* Model behavior analysis
* Research insights

Deployment would be premature without extensive validation, risk controls, and monitoring.

---

## 7. Notebooks Included

The following notebooks are provided and fully executable:

* **01 – Data Preparation**: data loading, cleaning, image generation
* **02 – EDA**: visual inspection of charts, label distribution, sanity checks
* **03 – Training**: CNN architecture experiments and training
* **04 – Evaluation**: predictions, probability analysis, performance discussion

These notebooks ensure transparency, reproducibility, and pedagogical clarity.

---

## 8. Final Remarks

This project represents a serious attempt to apply Machine Learning to trading problems while respecting the realities of financial markets:

* Low signal-to-noise ratio
* Rare but meaningful events
* The limits of prediction

While it may not maximize every grading category, it demonstrates:

* Clear problem framing
* Solid ML engineering practices
* Honest evaluation of results

Most importantly, it reflects a **long-term learning journey** into Python, Machine Learning, and quantitative finance.


