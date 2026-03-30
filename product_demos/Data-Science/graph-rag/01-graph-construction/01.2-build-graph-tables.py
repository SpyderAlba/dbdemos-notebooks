# Databricks notebook source
# MAGIC %md
# MAGIC # Build Graph Tables from Extracted Entities and Relationships
# MAGIC
# MAGIC In this notebook, we:
# MAGIC 1. Create `graph_vertices` from `entities_raw` (deduplicated)
# MAGIC 2. Create `graph_edges` from `relationships_raw`
# MAGIC 3. Build a GraphFrame and run PageRank for vertex importance
# MAGIC 4. Run label propagation for community detection
# MAGIC 5. Verify multi-hop paths with `g.find()`
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=01.2-build-graph-tables&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq mlflow>=3.10.1 graphframes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Graph Vertices Table
# MAGIC
# MAGIC We deduplicate the extracted entities, preferring the longest description for each entity ID.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE graph_vertices AS
# MAGIC SELECT
# MAGIC   entity_id AS id,
# MAGIC   entity_type,
# MAGIC   name,
# MAGIC   description,
# MAGIC   CAST(NULL AS MAP<STRING, STRING>) AS properties,
# MAGIC   CAST(NULL AS DOUBLE) AS pagerank_score,
# MAGIC   CAST(NULL AS INT) AS community_id
# MAGIC FROM (
# MAGIC   SELECT *,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY entity_id ORDER BY LENGTH(description) DESC) AS rn
# MAGIC   FROM entities_raw
# MAGIC )
# MAGIC WHERE rn = 1;

# COMMAND ----------

# DBTITLE 1,Vertex statistics
# MAGIC %sql
# MAGIC SELECT entity_type, COUNT(*) AS count FROM graph_vertices GROUP BY entity_type ORDER BY count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create Graph Edges Table
# MAGIC
# MAGIC Deduplicate edges and assign weights based on relationship type.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE graph_edges AS
# MAGIC SELECT
# MAGIC   src,
# MAGIC   dst,
# MAGIC   relationship,
# MAGIC   CASE
# MAGIC     WHEN relationship = 'RESOLVES_ERROR' THEN 1.0
# MAGIC     WHEN relationship = 'AFFECTS_PRODUCT' THEN 1.0
# MAGIC     WHEN relationship = 'HAS_FEATURE' THEN 1.0
# MAGIC     WHEN relationship = 'APPLIES_TO_PRODUCT' THEN 0.5
# MAGIC     WHEN relationship = 'RELATED_PRODUCT' THEN 0.3
# MAGIC     ELSE 0.5
# MAGIC   END AS weight,
# MAGIC   description
# MAGIC FROM (
# MAGIC   SELECT *,
# MAGIC     ROW_NUMBER() OVER (PARTITION BY src, dst, relationship ORDER BY LENGTH(description) DESC) AS rn
# MAGIC   FROM relationships_raw
# MAGIC   -- Only keep edges where both endpoints exist in graph_vertices
# MAGIC   WHERE src IN (SELECT id FROM graph_vertices)
# MAGIC     AND dst IN (SELECT id FROM graph_vertices)
# MAGIC )
# MAGIC WHERE rn = 1;

# COMMAND ----------

# DBTITLE 1,Edge statistics
# MAGIC %sql
# MAGIC SELECT relationship, COUNT(*) AS count FROM graph_edges GROUP BY relationship ORDER BY count DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Run PageRank with GraphFrames
# MAGIC
# MAGIC PageRank helps us rank entity importance. Products with many connections (errors, features, solutions) will have higher scores, making them more prominent in search results.

# COMMAND ----------

from graphframes import GraphFrame

# Load Delta tables as DataFrames
vertices_df = spark.table("graph_vertices").select("id", "entity_type", "name", "description")
edges_df = spark.table("graph_edges").select(
    spark.table("graph_edges")["src"],
    spark.table("graph_edges")["dst"],
    spark.table("graph_edges")["relationship"]
)

# Create GraphFrame
g = GraphFrame(vertices_df, edges_df)

print(f"Graph has {g.vertices.count()} vertices and {g.edges.count()} edges")

# COMMAND ----------

# Run PageRank
pagerank_results = g.pageRank(resetProbability=0.15, maxIter=10)

# Show top entities by PageRank
display(pagerank_results.vertices.orderBy("pagerank", ascending=False).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Merge PageRank scores back to `graph_vertices`

# COMMAND ----------

# Write PageRank scores to a temp view
pagerank_results.vertices.select("id", "pagerank").createOrReplaceTempView("pagerank_scores")

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO graph_vertices AS v
# MAGIC USING pagerank_scores AS pr
# MAGIC ON v.id = pr.id
# MAGIC WHEN MATCHED THEN UPDATE SET v.pagerank_score = pr.pagerank;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Community Detection with Label Propagation
# MAGIC
# MAGIC Label propagation clusters related entities together. This helps identify which products, errors, and solutions form natural groups.

# COMMAND ----------

# Run label propagation
lp_results = g.labelPropagation(maxIter=5)

# Show community distribution
display(lp_results.groupBy("label").count().orderBy("count", ascending=False).limit(10))

# COMMAND ----------

# Write community labels to temp view
lp_results.select("id", "label").createOrReplaceTempView("community_labels")

# COMMAND ----------

# MAGIC %sql
# MAGIC MERGE INTO graph_vertices AS v
# MAGIC USING community_labels AS cl
# MAGIC ON v.id = cl.id
# MAGIC WHEN MATCHED THEN UPDATE SET v.community_id = CAST(cl.label AS INT);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Verify Multi-Hop Paths
# MAGIC
# MAGIC The key value of a graph is enabling multi-hop reasoning. Let's verify we can traverse paths like:
# MAGIC - **Product -> Error -> Solution** (find solutions for a product's errors)
# MAGIC - **Solution -> Error -> Product** (find which products a solution applies to)

# COMMAND ----------

# DBTITLE 1,Find: Product -> Error -> Solution paths
# Find multi-hop paths: Product <- Error <- Solution
# "Which solutions exist for errors affecting a given product?"
motif_results = g.find("(solution)-[resolves]->(error); (error)-[affects]->(product)") \
    .filter("resolves.relationship = 'RESOLVES_ERROR'") \
    .filter("affects.relationship = 'AFFECTS_PRODUCT'") \
    .filter("product.entity_type = 'Product'") \
    .select(
        "product.name", "product.id",
        "error.name", "error.id",
        "solution.name", "solution.id"
    )

display(motif_results.limit(20))

# COMMAND ----------

# DBTITLE 1,Example: Solutions for Router X500 errors
# Specific example: What solutions exist for Router X500 errors?
router_solutions = g.find("(solution)-[resolves]->(error); (error)-[affects]->(product)") \
    .filter("resolves.relationship = 'RESOLVES_ERROR'") \
    .filter("affects.relationship = 'AFFECTS_PRODUCT'") \
    .filter("product.name = 'Router X500'") \
    .select("product.name", "error.name", "solution.name")

print("Solutions for Router X500 errors:")
display(router_solutions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Graph Statistics Summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   (SELECT COUNT(*) FROM graph_vertices) AS total_vertices,
# MAGIC   (SELECT COUNT(*) FROM graph_edges) AS total_edges,
# MAGIC   (SELECT COUNT(DISTINCT entity_type) FROM graph_vertices) AS entity_types,
# MAGIC   (SELECT COUNT(DISTINCT relationship) FROM graph_edges) AS relationship_types,
# MAGIC   (SELECT COUNT(DISTINCT community_id) FROM graph_vertices) AS communities

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Top entities by PageRank
# MAGIC SELECT id, entity_type, name, ROUND(pagerank_score, 4) AS pagerank, community_id
# MAGIC FROM graph_vertices
# MAGIC ORDER BY pagerank_score DESC
# MAGIC LIMIT 15

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC Our knowledge graph is built! We have:
# MAGIC - Deduplicated vertices with PageRank scores and community IDs
# MAGIC - Weighted edges representing typed relationships
# MAGIC - Verified multi-hop paths (Product -> Error -> Solution)
# MAGIC
# MAGIC Next, we'll create the Vector Search index and register graph traversal tools.
# MAGIC
# MAGIC Open [02-indexing-and-retrieval/02.1-vector-search-index]($../02-indexing-and-retrieval/02.1-vector-search-index) to continue.
