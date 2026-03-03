# 01_data_simulation.py
# Objective: Generate synthetic customer acquisition data with hidden "personas"
# to demonstrate K-Means clustering for targeted marketing and reduced CAC.

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand, when, round, randn

spark = SparkSession.builder.appName("IFBI_Lead_Simulation").getOrCreate()
print("Starting Acquisition Lead Data Simulation...")

# 1. Generate Base DataFrame (100,000 potential marketing leads)
num_leads = 100000
df_base = spark.range(0, num_leads).withColumnRenamed("id", "lead_id")

# 2. Assign Hidden Personas (This ensures our K-Means finds actual clusters later)
# We use a random number between 0 and 1 to split the population into 3 hidden groups

df_personas = df_base.withColumn("rand_persona", rand(seed=42)) 

print("Engineering behavioral features based on hidden marketing personas...")
# Digital Shoppers: Young ---> Low Income ---> High browsing activity ---> Just browsing, no inquiries ---> Low actual intent
# High-Intent Buyers: Mid Age ---> Solid Income ---> Focused browsing ---> High intent, asking questions
# Wealthy Traditionalists: Older ---> High Income ---> Barely uses the website ---> Low time on side, but highly valuable
df_features = df_personas \
    .withColumn("age",
                when(col("rand_persona") < 0.35, round(22 + (rand() * 15)))     
                .when(col("rand_persona") < 0.70, round(35 + (rand() * 20)))    
                .otherwise(round(55 + (rand() * 25)))) \
    .withColumn("income_bracket",
                when(col("rand_persona") < 0.35, round(45000 + (randn() * 10000))) 
                .when(col("rand_persona") < 0.70, round(75000 + (randn() * 20000))) 
                .otherwise(round(120000 + (randn() * 35000)))) \
    .withColumn("web_pages_visited",
                when(col("rand_persona") < 0.35, round(15 + (rand() * 10)))
                .when(col("rand_persona") < 0.70, round(5 + (rand() * 5)))
                .otherwise(round(2 + (rand() * 3)))) \
    .withColumn("time_on_site_mins",
                when(col("rand_persona") < 0.35, round(20 + (randn() * 5), 1))
                .when(col("rand_persona") < 0.70, round(10 + (randn() * 3), 1))
                .otherwise(round(3 + (randn() * 2), 1))) \
    .withColumn("previous_inquiries",
                when(col("rand_persona") < 0.35, round((rand() * 1)))
                .when(col("rand_persona") < 0.70, round(2 + (rand() * 3)))
                .otherwise(rand() * 2)) 

# 3. Clean up the data (remove negatives caused by randn and drop the hidden persona column)
# We drop the 'rand_persona' because Unsupervised Learning isn't allowed to see the "answer key"!
df_clean = df_features \
    .withColumn("income_bracket", when(col("income_bracket") < 20000, 20000).otherwise(col("income_bracket"))) \
    .withColumn("time_on_site_mins", when(col("time_on_site_mins") < 0, 0.5).otherwise(col("time_on_site_mins"))) \
    .drop("rand_persona")

print(f"Successfully generated {num_leads} synthetic leads.")

# 4. Save to Unity Catalog Managed Delta Table
output_table = "workspace.default.ifbi_simulated_leads"
print(f"Saving data to Unity Catalog: {output_table}...")

df_clean.write.format("delta").mode("overwrite").saveAsTable(output_table)

print("✅ Data simulation complete! Ready for Feature Engineering.")
display(df_clean)
                
    
