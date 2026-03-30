# Databricks notebook source
# MAGIC %md
# MAGIC # Graph Retrieval Tools as Unity Catalog Functions
# MAGIC
# MAGIC In this notebook, we register 3 UC functions that enable the agent to traverse the knowledge graph:
# MAGIC
# MAGIC 1. **`find_entities`** - Fuzzy entity lookup with PageRank ordering
# MAGIC 2. **`get_entity_neighbors`** - 1-2 hop graph traversal using SQL joins
# MAGIC 3. **`find_solutions_for_product`** - Multi-hop Product -> Error -> Solution path query
# MAGIC
# MAGIC These functions become tools the agent can call to answer relationship-aware questions.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=02.2-graph-retrieval-tools&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq mlflow>=3.10.1 databricks-langchain pydantic
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. `find_entities` - Fuzzy Entity Lookup
# MAGIC
# MAGIC This function allows the agent to search for entities by name or description using fuzzy matching (LIKE). Results are ordered by PageRank score so the most important entities appear first.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION find_entities(
# MAGIC   search_term STRING COMMENT 'The search term to look for in entity names and descriptions. Use keywords like product names, error codes, or feature names.'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   id STRING,
# MAGIC   entity_type STRING,
# MAGIC   name STRING,
# MAGIC   description STRING,
# MAGIC   pagerank_score DOUBLE
# MAGIC )
# MAGIC COMMENT 'Searches the knowledge graph for entities (Products, Features, Errors, Solutions) matching the search term. Use this to find entity IDs before traversing the graph. Returns results ordered by importance (PageRank).'
# MAGIC RETURN
# MAGIC   SELECT id, entity_type, name, description, pagerank_score
# MAGIC   FROM graph_vertices
# MAGIC   WHERE LOWER(name) LIKE CONCAT('%', LOWER(search_term), '%')
# MAGIC      OR LOWER(description) LIKE CONCAT('%', LOWER(search_term), '%')
# MAGIC      OR LOWER(id) LIKE CONCAT('%', LOWER(search_term), '%')
# MAGIC   ORDER BY pagerank_score DESC
# MAGIC   LIMIT 10;

# COMMAND ----------

# DBTITLE 1,Test find_entities
# MAGIC %sql
# MAGIC SELECT * FROM find_entities('Router X500')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM find_entities('VLAN')

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM find_entities('ERR-012')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. `get_entity_neighbors` - Graph Traversal (1-2 hops)
# MAGIC
# MAGIC This function traverses the graph from a given entity ID, returning its direct neighbors (1-hop) and optionally 2-hop neighbors. This enables the agent to discover related entities by following edges.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION get_entity_neighbors(
# MAGIC   entity_id STRING COMMENT 'The entity ID to start traversal from (e.g., PROD-001, ERR-012). Use find_entities first to discover entity IDs.',
# MAGIC   max_hops INT DEFAULT 1 COMMENT 'Maximum number of hops: 1 for direct neighbors, 2 for neighbors of neighbors. Default is 1.',
# MAGIC   relationship_filter STRING DEFAULT NULL COMMENT 'Optional: filter by relationship type (HAS_FEATURE, AFFECTS_PRODUCT, RESOLVES_ERROR, RELATED_PRODUCT, APPLIES_TO_PRODUCT). NULL returns all relationships.'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   hop INT,
# MAGIC   source_id STRING,
# MAGIC   relationship STRING,
# MAGIC   neighbor_id STRING,
# MAGIC   neighbor_type STRING,
# MAGIC   neighbor_name STRING,
# MAGIC   neighbor_description STRING
# MAGIC )
# MAGIC COMMENT 'Traverses the knowledge graph from a given entity, returning neighbors within the specified number of hops. Use this to explore relationships: what features a product has, what errors affect it, what solutions resolve those errors, etc.'
# MAGIC RETURN
# MAGIC   -- 1-hop neighbors (outgoing edges)
# MAGIC   SELECT 1 AS hop, e.src AS source_id, e.relationship, v.id AS neighbor_id,
# MAGIC          v.entity_type AS neighbor_type, v.name AS neighbor_name, v.description AS neighbor_description
# MAGIC   FROM graph_edges e
# MAGIC   JOIN graph_vertices v ON e.dst = v.id
# MAGIC   WHERE e.src = entity_id
# MAGIC     AND (relationship_filter IS NULL OR e.relationship = relationship_filter)
# MAGIC   UNION ALL
# MAGIC   -- 1-hop neighbors (incoming edges)
# MAGIC   SELECT 1 AS hop, e.dst AS source_id, e.relationship, v.id AS neighbor_id,
# MAGIC          v.entity_type AS neighbor_type, v.name AS neighbor_name, v.description AS neighbor_description
# MAGIC   FROM graph_edges e
# MAGIC   JOIN graph_vertices v ON e.src = v.id
# MAGIC   WHERE e.dst = entity_id
# MAGIC     AND (relationship_filter IS NULL OR e.relationship = relationship_filter)
# MAGIC   UNION ALL
# MAGIC   -- 2-hop neighbors (only if max_hops >= 2, outgoing then outgoing)
# MAGIC   SELECT 2 AS hop, e2.src AS source_id, e2.relationship, v2.id AS neighbor_id,
# MAGIC          v2.entity_type AS neighbor_type, v2.name AS neighbor_name, v2.description AS neighbor_description
# MAGIC   FROM graph_edges e1
# MAGIC   JOIN graph_edges e2 ON e1.dst = e2.src
# MAGIC   JOIN graph_vertices v2 ON e2.dst = v2.id
# MAGIC   WHERE e1.src = entity_id AND max_hops >= 2
# MAGIC     AND (relationship_filter IS NULL OR e2.relationship = relationship_filter)
# MAGIC     AND v2.id != entity_id
# MAGIC   UNION ALL
# MAGIC   -- 2-hop neighbors (incoming then outgoing)
# MAGIC   SELECT 2 AS hop, e2.src AS source_id, e2.relationship, v2.id AS neighbor_id,
# MAGIC          v2.entity_type AS neighbor_type, v2.name AS neighbor_name, v2.description AS neighbor_description
# MAGIC   FROM graph_edges e1
# MAGIC   JOIN graph_edges e2 ON e1.src = e2.src
# MAGIC   JOIN graph_vertices v2 ON e2.dst = v2.id
# MAGIC   WHERE e1.dst = entity_id AND max_hops >= 2
# MAGIC     AND e1.src != e2.dst
# MAGIC     AND (relationship_filter IS NULL OR e2.relationship = relationship_filter)
# MAGIC     AND v2.id != entity_id;

# COMMAND ----------

# DBTITLE 1,Test: what features does Router X500 have?
# MAGIC %sql
# MAGIC SELECT * FROM get_entity_neighbors('PROD-001', 1, 'HAS_FEATURE')

# COMMAND ----------

# DBTITLE 1,Test: what errors affect Router X500?
# MAGIC %sql
# MAGIC SELECT * FROM get_entity_neighbors('PROD-001', 1, 'AFFECTS_PRODUCT')

# COMMAND ----------

# DBTITLE 1,Test: 2-hop from ERR-012 (Memory Exhaustion)
# MAGIC %sql
# MAGIC SELECT * FROM get_entity_neighbors('ERR-012', 2, NULL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. `find_solutions_for_product` - Multi-Hop Path Query
# MAGIC
# MAGIC This is the key Graph RAG function: given a product search term, it traverses the full **Product -> Error -> Solution** path to find all relevant solutions. This is exactly the kind of multi-hop reasoning that vector search alone cannot do.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE FUNCTION find_solutions_for_product(
# MAGIC   product_search STRING COMMENT 'The product name or keyword to search for (e.g., "Router X500", "Switch SW-3000", "Modem"). Partial matches are supported.'
# MAGIC )
# MAGIC RETURNS TABLE (
# MAGIC   product_name STRING,
# MAGIC   product_id STRING,
# MAGIC   error_name STRING,
# MAGIC   error_id STRING,
# MAGIC   error_description STRING,
# MAGIC   solution_description STRING,
# MAGIC   solution_id STRING
# MAGIC )
# MAGIC COMMENT 'Finds all solutions for errors that affect a given product by traversing the knowledge graph path: Product <- AFFECTS_PRODUCT <- Error <- RESOLVES_ERROR <- Solution. Use this for questions like "What solutions exist for errors related to [product]?" or "How do I fix problems with [product]?"'
# MAGIC RETURN
# MAGIC   SELECT
# MAGIC     p.name AS product_name,
# MAGIC     p.id AS product_id,
# MAGIC     err.name AS error_name,
# MAGIC     err.id AS error_id,
# MAGIC     err.description AS error_description,
# MAGIC     sol.description AS solution_description,
# MAGIC     sol.id AS solution_id
# MAGIC   FROM graph_vertices p
# MAGIC   JOIN graph_edges e_affects ON e_affects.dst = p.id AND e_affects.relationship = 'AFFECTS_PRODUCT'
# MAGIC   JOIN graph_vertices err ON e_affects.src = err.id AND err.entity_type = 'Error'
# MAGIC   JOIN graph_edges e_resolves ON e_resolves.dst = err.id AND e_resolves.relationship = 'RESOLVES_ERROR'
# MAGIC   JOIN graph_vertices sol ON e_resolves.src = sol.id AND sol.entity_type = 'Solution'
# MAGIC   WHERE p.entity_type = 'Product'
# MAGIC     AND LOWER(p.name) LIKE CONCAT('%', LOWER(product_search), '%')
# MAGIC   ORDER BY p.name, err.name, sol.id;

# COMMAND ----------

# DBTITLE 1,Test: solutions for Router X500 errors
# MAGIC %sql
# MAGIC SELECT * FROM find_solutions_for_product('Router X500')

# COMMAND ----------

# DBTITLE 1,Test: solutions for Switch SW-3000 errors
# MAGIC %sql
# MAGIC SELECT * FROM find_solutions_for_product('Switch SW-3000')

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Side-by-Side Comparison: Graph Tools vs. Vector Search
# MAGIC
# MAGIC Let's compare what we get from our graph tools versus vector search for the same multi-hop question.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)
vs_index_fullname = f"{catalog}.{db}.document_chunks_vs_index"

# Question requiring multi-hop reasoning
question = "What solutions exist for errors related to the Router X500 series?"

print("=" * 70)
print("VECTOR SEARCH RESULTS")
print("=" * 70)
results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["chunk_id", "product_name", "chunk_text"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])
for doc in docs:
    print(f"  Product: {doc[1]}, Text: {doc[2][:150]}...")
    print()

# COMMAND ----------

print("=" * 70)
print("GRAPH TOOL RESULTS (find_solutions_for_product)")
print("=" * 70)

graph_results = spark.sql("SELECT product_name, error_name, error_id, solution_description FROM find_solutions_for_product('Router X500')")
display(graph_results)

# COMMAND ----------

# MAGIC %md
# MAGIC The graph tool returns **structured, complete results**: every error affecting the Router X500 with every corresponding solution, obtained by traversing the graph path. Vector search returns the most semantically similar document chunks, which may or may not contain the complete answer.
# MAGIC
# MAGIC ## Next Steps
# MAGIC
# MAGIC Now that our tools are registered, let's build an agent that combines both approaches.
# MAGIC
# MAGIC Open [03-agent-eval/03.1-graph-rag-agent-evaluation]($../03-agent-eval/03.1-graph-rag-agent-evaluation) to build, evaluate, and deploy the Graph RAG agent.
