# 02_feature_engineering.py
# Objective: Prepare the simulated lead data for K-Means clustering by assembling
# features into vectors and standardizing their scales.

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

spark = SparkSession.builder.appName("IFBI_Feature_Engineering").getOrCreate()
print("Starting Feature Engineering for K-Means Clustering...")

# 1. Load the simulated leads data
input_table = "workspace.default.ifbi_simulated_leads"
df_raw = spark.table(input_table)

# 2. Define the Feature Columns
feature_cols = [
    "age", 
    "income_bracket", 
    "web_pages_visited", 
    "time_on_site_mins", 
    "previous_inquiries"
]

# 3. Step A: VectorAssembler
# K-Means requires all input features to be in a single array column
assembler = VectorAssembler(inputCols=feature_cols, outputCol="unscaled_features")

# 4. Step B: StandardScaler (Crucial for K-Means!)
# Because 'income' is in the tens of thousands and 'previous_inquiries' is between 0-5,
# 'income' would unfairly dominate the math. StandardScaler levels the playing field
# so every feature has an equal vote in determining the clusters.

scaler = StandardScaler(
    inputCol="unscaled_features", 
    outputCol="features", 
    withStd=True,
    withMean=True
)

# 5. Build and Run the ML Pipeline
print("Building and fitting the VectorAssembler and StandardScaler pipeline...")
pipeline = Pipeline(stages=[assembler, scaler])

# Fit (calculate the means/standard deviations) and Transform (apply the math)
pipeline_model = pipeline.fit(df_raw)
df_engineered = pipeline_model.transform(df_raw)

# 6. Save the Engineered Data
output_table = "workspace.default.ifbi_engineered_leads"
print(f"Saving engineered features to Unity Catalog: {output_table}...")

# Select only the columns needed for the modeling phase to keep it clean
modeling_dataset = df_engineered.select("lead_id", "features")
modeling_dataset.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("✅ Feature Engineering complete! Data is scaled and ready for Clustering.")
display(modeling_dataset.limit(5))
