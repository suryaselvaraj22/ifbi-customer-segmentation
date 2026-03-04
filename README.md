# Customer Segmentation & Acquisition Engine

![Databricks](https://img.shields.io/badge/Databricks-Serverless-FF3621?style=for-the-badge&logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-MLlib-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white)
![Unsupervised Learning](https://img.shields.io/badge/Machine_Learning-K--Means-0194E2?style=for-the-badge)

## Executive Summary

This project implements an end-to-end Unsupervised Machine Learning pipeline to transition an insurance provider's marketing strategy from "broad-stroke" campaigns to hyper-personalized, persona-driven targeting.

By leveraging **K-Means Clustering** within a Databricks Unity Catalog environment, this engine groups 100,000 synthetic inbound leads based on hidden demographic and behavioral patterns, enabling the business to:

* **Reduce Customer Acquisition Cost (CAC):** By tailoring Call-to-Action (CTA) logic to specific engagement styles.

* **Optimize Ad Spend:** Identifying "Digital Window Shoppers" (high compute/browsing, low intent) vs. "High-Intent Buyers".

* **Drive New Originations:** Routing high-value, low-tech "Traditionalists" directly to human agents rather than digital funnels.

## The Tech Stack

* **Core Logic:** Python, PySpark (Spark MLlib)

* **Platform:** Databricks Serverless (Unity Catalog)

* **Feature Engineering:** `VectorAssembler`, `StandardScaler` (Z-Score Normalization)

* **Data Storage:** Managed Delta Tables

## Key Results & Business Impact

The K-Means algorithm (k=3) successfully discovered three distinct, highly separated marketing personas, achieving an exceptionally high **Silhouette Score of 0.7791**.

### Discovered Business Profiles (Cluster Analysis)

By joining the mathematical cluster predictions back to the unscaled business data, we successfully profiled the following customer segments:

* **Cluster 0: The Wealthy Traditionalists**

  * *Profile:* Avg Age 67 | Avg Income $120k+ | Low Web Activity (3.5 pages)

  * *Strategy:* High Lifetime Value. Route to white-glove human advisory channels; do not rely on digital self-service.

* **Cluster 1: The High-Intent Buyers**

  * *Profile:* Avg Age 45 | Avg Income $74k | Focused Web Activity (7.5 pages)

  * *Strategy:* The core target demographic. Serve direct-to-purchase CTAs and competitive rate comparisons.

* **Cluster 2: The Digital Window Shoppers**

  * *Profile:* Avg Age 29 | Avg Income $45k | High Web Activity (20+ pages)

  * *Strategy:* High traffic but low immediate intent. Nurture via automated email drip campaigns; minimize expensive retargeting ad spend.

## Solution Architecture

This repository is modularized into a 3-stage enterprise pipeline:

### `01_data_simulation.py`

Engineered a realistic synthetic dataset of 100,000 inbound marketing leads using PySpark's random distribution functions (`rand`, `randn`), purposefully embedding hidden behavioral correlations to simulate real-world traffic.

### `02_feature_engineering.py`

Constructed a PySpark ML `Pipeline` to prepare the data for distance-based clustering. Utilized `StandardScaler` to perform Z-Score normalization, ensuring large-magnitude financial data (Income) did not mathematically overshadow behavioral data (Web Pages Visited) in Euclidean distance calculations.

### `03_kmeans_clustering.py`

Trained the K-Means clustering model and evaluated cluster separation via Silhouette Score. Aggregated the model's abstract numeric predictions back against the raw demographic data to output human-readable business personas for the marketing team.