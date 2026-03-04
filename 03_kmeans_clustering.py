# 03_kmeans_clustering.py
# Objective: Train a K-Means clustering model to discover hidden customer personas,
# evaluate cluster quality, and profile the business metrics for each segment.

from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import avg, count, round

spark = SparkSession.builder.appName("IFBI_KMeans_Modeling").getOrCreate()
print("Starting K-Means Clustering Modeling Phase...")

# 1. Load the Data
engineered_table = "workspace.default.ifbi_engineered_leads"
raw_table = "workspace.default.ifbi_simulated_leads"

df_engineered = spark.table(engineered_table)
df_raw = spark.table(raw_table)

# 2. Train the K-Means Model
# We set k=3 because we are looking for 3 distinct marketing personas
kmeans = KMeans(featuresCol="features", predictionCol="cluster", k=3, seed=42)
model = kmeans.fit(df_engineered)

# 3. Make Predictions (Assign each lead to a cluster)
predictions = model.transform(df_engineered)

# 4. Evaluate the Model (Silhouette Score)
# Silhouette Score measures how similar an object is to its own cluster compared to other clusters.
# Scores range from -1 to +1. A higher score means perfectly separated, distinct clusters.
evaluator = ClusteringEvaluator(featuresCol="features", predictionCol="cluster", metricName="silhouette")
silhouette_score = evaluator.evaluate(predictions)

print("\n" + "=" * 50)
print(f"🏆 MODEL EVALUATION: Silhouette Score = {silhouette_score:.4f} 🏆")
print("=" * 50)

# 5. Business Interpretation: Cluster Profiling
# We join the cluster assignments back to the raw data to understand the characteristics of each cluster
print("\nProfiling the discovered clusters...")
df_joined = df_raw.join(predictions.select("lead_id", "cluster"), on="lead_id")

# For each cluster, we calculate the average values of the original features to understand the personas
cluster_profiles = df_joined.groupBy("cluster").agg(
    count("lead_id").alias("num_leads"),
    round(avg("age"), 1).alias("avg_age"),
    round(avg("income_bracket"), 0).alias("avg_income"),
    round(avg("web_pages_visited"), 1).alias("avg_web_pages"),
    round(avg("time_on_site_mins"), 1).alias("avg_time_on_site"),
    round(avg("previous_inquiries"), 1).alias("avg_previous_inquiries")
).orderBy("cluster")

# 6. Save the results to Unity Catalog
output_table = "workspace.default.ifbi_cluster_profiles"
cluster_profiles.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("✅ Clustering complete! Business profiles saved to Catalog.")
display(cluster_profiles)