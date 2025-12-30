# 📊 Customer Segmentation Using Machine Learning

## 📌 Project Overview
This project implements **Customer Segmentation** using **unsupervised machine learning (K-Means clustering)** to group customers based on their **purchase behavior**.  
The goal is to help businesses **understand customer patterns**, improve **targeted marketing**, and support **data-driven decision-making**.

The project was originally developed by another contributor and later **reviewed, improved, and professionally documented** to meet academic and industry standards.

---

## 🎯 Problem Statement
Businesses often fail to personalize services because they treat all customers the same.  
This project solves that issue by:
- Identifying **distinct customer groups**
- Understanding **spending behavior**
- Enabling **strategic marketing and retention planning**

---

## 🎯 Objectives
- Clean and preprocess customer transaction data  
- Analyze customer purchase patterns  
- Segment customers using **K-Means clustering**  
- Visualize clusters and category distribution  
- Evaluate clustering performance using metrics  
- Save the trained model for reuse  

---

## 🧠 Machine Learning Model Explanation

### 🔹 Algorithm Used: K-Means Clustering
K-Means is an **unsupervised learning algorithm** that partitions data into **K clusters** based on similarity.

### 🔹 Features Used
- **PurchaseAmount**
- **Month**

### 🔹 Model Workflow
```
Raw Data → Cleaning → Feature Selection → K-Means → Visualization → Evaluation → Model Saving
```

---

## 📐 Model Evaluation Results

| Metric | Value |
|------|------|
| Silhouette Score | 0.20 |
| Davies–Bouldin Score | 0.41 |

---

## 📊 Visualizations
- Customer Clusters
- Category Distribution
- Model Metrics

---

## 📂 Project Structure
```
customer-segmentation-ml/
├── data/
├── notebooks/
├── src/
├── models/
├── images/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack
Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

---

## 📜 License
Educational use only.

