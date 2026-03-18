
# Multi-Task Learning with Gradient Surgery (PCGrad)

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?logo=streamlit)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)

A production-grade Multi-Task Learning (MTL) system designed to solve the problem of negative transfer using gradient surgery. This project implements a custom training loop in TensorFlow to apply **PCGrad** (Projecting Conflicting Gradients), ensuring that competing tasks do not destructively interfere with the shared representation space.

## 🧠 Project Overview

In Multi-Task Learning, a single neural network is trained to optimize multiple objectives simultaneously. While this often leads to better generalization through a shared feature representation, it frequently suffers from **Gradient Conflict**. This occurs when the gradients of different tasks point in opposing directions, causing the shared weights to stall or degrade in performance (Negative Transfer).

**The Solution:**
This project implements the **PCGrad algorithm** (Yu et al., 2020). During backpropagation, the custom training loop intercepts the gradients from each task. If it detects a conflict (cosine similarity < 0), it orthogonally projects the gradient of one task onto the normal plane of the other. This "gradient surgery" removes the conflicting component, allowing the model to make optimal progress on both tasks simultaneously.

## 🏗️ Architecture

The system is built using the `tf.keras.Model` subclassing API to maintain strict control over the forward and backward passes.
* **Shared Backbone:** A sequence of Dense and Batch Normalization layers that learn a generalized representation of the input data.
* **Task-Specific Heads:** Two distinct neural network branches (Task A and Task B) that intake the shared representation to perform binary classification.

## ✨ Key Features

* **Custom Training Loop:** Built from scratch using `@tf.function` and `tf.GradientTape(persistent=True)` to decouple gradient calculation from weight application.
* **Real-time Conflict Detection:** Calculates and logs the cosine similarity of flattened gradient vectors across all shared layers at every training step.
* **Automated Data Pipeline:** A highly optimized `tf.data.Dataset` generator that simulates overlapping and conflicting feature dependencies.
* **Interactive Monitoring Dashboard:** A Streamlit application providing deep introspection into training dynamics, task performance, and feature manifold topology (via UMAP).
* **Fully Dockerized:** Reproducible, one-click deployment environment.

## 🚀 Getting Started

This project is fully containerized. You do not need to install Python or TensorFlow on your local machine to run the analysis dashboard.

### Prerequisites
* [Docker](https://docs.docker.com/get-docker/) and Docker Compose installed.

### 1. Build and Run the Project
To spin up the Streamlit dashboard and view the results of the gradient surgery, run the following command from the root directory:

```bash
docker-compose up --build
```

### 2. Access the Dashboard
Once the container is running, navigate to your web browser:
👉 **http://localhost:8501**

## 📊 Dashboard & Monitoring

The Streamlit dashboard (`app.py`) parses the logged training metrics and provides three key analytical views:

1. **Gradient Conflict Monitor:** A time-series visualization tracking the cosine similarity between the gradients of Task A and Task B. Red markers highlight moments of negative similarity, proving where PCGrad intervened.
2. **Task Performance Comparison:** Side-by-side validation accuracy graphs comparing the naive summation baseline against the PCGrad-optimized model.
3. **Shared Representation Inspector:** A UMAP dimensionality reduction plot mapping the high-dimensional output of the shared backbone into 2D space, demonstrating how the model organizes features to satisfy both tasks.

## 📂 Repository Structure

```text
mtl_gradient_surgery/
├── app.py                      # Streamlit monitoring dashboard
├── generate_model_summary.py   # Architecture summarization script
├── train_baseline.py           # Baseline MTL training (naive summation)
├── train_pcgrad.py             # Custom training loop with PCGrad
├── requirements.txt            # Dependency definitions
├── Dockerfile                  # Container environment instructions
├── docker-compose.yml          # Container orchestration
├── .env.example                # Template for environment variables
├── src/
│   ├── dataset.py              # tf.data.Dataset pipeline & synthetic generation
│   └── models.py               # tf.keras.Model subclassed architecture
└── results/                    # Generated logs, metrics, and analysis
    ├── analysis.md             # Written qualitative analysis report
    └── *.csv / *.json          # Automated training outputs
```

---

## ✍️ Author

**Billakurti Venkata Suryanarayana** *Roll Number: 23MH1A4409* Data Scientist & ML Engineer  
```

