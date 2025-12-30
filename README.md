📊 Customer Segmentation Using Machine Learning
📌 Project Overview

This project implements Customer Segmentation using unsupervised machine learning (K-Means clustering) to group customers based on their purchase behavior.
The goal is to help businesses understand customer patterns, improve targeted marketing, and support data-driven decision-making.

The project was originally developed by another contributor and later reviewed, improved, and professionally documented to meet academic and industry standards.

🎯 Problem Statement

Businesses often fail to personalize services because they treat all customers the same.
This project solves that issue by:

Identifying distinct customer groups

Understanding spending behavior

Enabling strategic marketing and retention planning

🎯 Objectives

Clean and preprocess customer transaction data

Analyze customer purchase patterns

Segment customers using K-Means clustering

Visualize clusters and category distribution

Evaluate clustering performance using metrics

Save the trained model for reuse

🧠 Machine Learning Model Explanation
🔹 Algorithm Used: K-Means Clustering

K-Means is an unsupervised learning algorithm that partitions data into K clusters based on similarity.

🔹 Features Used

PurchaseAmount

Month

These features help identify:

Customer spending levels

Purchase timing patterns

🔹 Model Workflow
Raw Data
   ↓
Data Cleaning
   ↓
Feature Selection
   ↓
K-Means Clustering
   ↓
Visualization
   ↓
Model Evaluation
   ↓
Model Saving

🔹 Core Model Logic (Code Snippet)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(features)

silhouette = silhouette_score(features, labels)
db_score = davies_bouldin_score(features, labels)

📐 Model Evaluation Results
Metric	Value	Interpretation
Silhouette Score	0.20	Moderate cluster separation
Davies–Bouldin Score	0.41	Compact and acceptable clusters

📌 These values are reasonable for small datasets and can be improved using more data or additional features.

📊 Visualizations & Insights
🔹 Customer Clusters

Insights:

Cluster 0: Low-spending customers

Cluster 1: High-value / premium customers

Cluster 2: Regular customers

📌 Enables identification of budget, regular, and premium customers.

🔹 Category Distribution

Insights:

Balanced transactions across:

Electronics

Fashion

Groceries

No dominant category bias in the dataset

🔹 Model Metrics & Saving

Evaluation metrics computed successfully

Trained model saved for future use

💾 Model Persistence

The trained K-Means model is saved using serialization, allowing:

Reuse without retraining

Integration with dashboards or APIs

Deployment in real-world applications

🛠️ Tech Stack

Programming Language: Python

Libraries:

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

Environment: Jupyter Notebook

📂 Project Structure
customer-segmentation-ml/
│
├── data/
│   └── processed/
│       └── transactions.csv
│
├── notebooks/
│   └── customer_segmentation.ipynb
│
├── src/
│   └── train_model.py
│
├── models/
│   └── kmeans_model.pkl
│
├── images/
│   ├── customer_clusters.png
│   ├── category_distribution.png
│   └── model_metrics.png
│
├── docs/
│   └── project_report.pdf
│
├── requirements.txt
├── .gitignore
└── README.md

💼 Business Applications

🎯 Targeted marketing campaigns

💳 High-value customer identification

🛍 Personalized offers & promotions

📊 Customer behavior analysis

⚠️ Limitations

Small dataset size

Limited behavioral features

Static historical data

🔮 Future Enhancements

Add RFM analysis

Try DBSCAN or Hierarchical Clustering

Integrate real-time customer data

Deploy using Streamlit or Flask

Improve model with additional features

👤 Contributor Note

This project was originally developed by another contributor.
The current version focuses on:

Clear documentation

Insight-driven explanations

Academic and professional readiness

📜 License

This project is intended for educational and learning purposes only.

⭐ Acknowledgment

Special thanks to open-source tools and libraries that enabled this project.
