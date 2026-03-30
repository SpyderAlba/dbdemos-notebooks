# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Graph RAG: Combining Knowledge Graphs with Vector Search for Multi-Hop Reasoning
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-chain-1.png?raw=true" width="800px" style="float: right">
# MAGIC
# MAGIC Pure vector search works well for direct, content-based questions like *"How do I configure WiFi 6E?"* But it **fails for multi-hop reasoning questions** such as:
# MAGIC
# MAGIC > *"What solutions exist for errors related to the Router X500 series?"*
# MAGIC
# MAGIC This question requires traversing entity relationships: **Product -> Error -> Solution**. A knowledge graph captures these relationships explicitly, enabling the agent to follow paths that vector similarity alone cannot discover.
# MAGIC
# MAGIC In this demo, we build a **Graph RAG** system that combines:
# MAGIC - **Knowledge Graph** (Delta tables + GraphFrames) for structured entity relationships
# MAGIC - **Vector Search** for semantic retrieval over document chunks
# MAGIC - **UC Functions** as agent tools for graph traversal
# MAGIC - **Mosaic AI Agent Framework** for evaluation and deployment
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=00-graph-rag-introduction&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 1/ Build the Knowledge Graph
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-1.png?raw=true" width="500px" style="float: right">
# MAGIC
# MAGIC We start by ingesting PDF technical documentation, extracting entities (Products, Features, Errors, Solutions) and their relationships using AI, and building a graph stored as Delta tables.
# MAGIC
# MAGIC - Parse PDFs with `ai_parse_document()`
# MAGIC - Extract entities and relationships with `AI_QUERY()`
# MAGIC - Build `graph_vertices` and `graph_edges` Delta tables
# MAGIC - Run PageRank and community detection with GraphFrames

# COMMAND ----------

# MAGIC %md
# MAGIC Open [01-graph-construction/01.1-document-ingestion-entity-extraction]($./01-graph-construction/01.1-document-ingestion-entity-extraction) to get started.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 2/ Create Vector Search Index and Graph Retrieval Tools
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-2.png?raw=true" width="500px" style="float: right">
# MAGIC
# MAGIC Next, we create a Vector Search index for semantic document retrieval and register Unity Catalog functions that enable graph traversal:
# MAGIC
# MAGIC - `find_entities()` - fuzzy entity lookup with PageRank ordering
# MAGIC - `get_entity_neighbors()` - 1-2 hop graph traversal
# MAGIC - `find_solutions_for_product()` - multi-hop Product -> Error -> Solution paths

# COMMAND ----------

# MAGIC %md
# MAGIC Open [02-indexing-and-retrieval/02.1-vector-search-index]($./02-indexing-and-retrieval/02.1-vector-search-index) to create the vector search index.
# MAGIC
# MAGIC Then open [02-indexing-and-retrieval/02.2-graph-retrieval-tools]($./02-indexing-and-retrieval/02.2-graph-retrieval-tools) to register the graph traversal tools.

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## 3/ Build, Evaluate, and Deploy the Graph RAG Agent
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/ai-agent/mlflow-evaluate-0.png?raw=true" width="500px" style="float: right">
# MAGIC
# MAGIC We build a LangGraph-based agent that combines graph traversal tools with vector search, evaluate it against multi-hop reasoning questions, and compare its performance to a vector-only baseline.
# MAGIC
# MAGIC - Agent with both graph tools and vector search retriever
# MAGIC - MLflow evaluation with multi-hop test questions
# MAGIC - Side-by-side comparison: Graph RAG vs. vector-only
# MAGIC - Register and deploy to Model Serving

# COMMAND ----------

# MAGIC %md
# MAGIC Open [03-agent-eval/03.1-graph-rag-agent-evaluation]($./03-agent-eval/03.1-graph-rag-agent-evaluation) to evaluate and deploy the agent.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4/ Deploy a Chatbot Frontend
# MAGIC
# MAGIC Deploy a Gradio chatbot application using Databricks Lakehouse Apps that hits the model serving endpoint, allowing users to ask both content-based and multi-hop relationship questions.
# MAGIC
# MAGIC Open [04-deploy-app/04-Deploy-Frontend-Lakehouse-App]($./04-deploy-app/04-Deploy-Frontend-Lakehouse-App) to deploy the frontend.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5/ (Optional) Low-Latency Graph Queries with Lakebase
# MAGIC
# MAGIC For production workloads requiring sub-millisecond graph lookups, explore using Lakebase (managed Postgres) with recursive CTEs for real-time graph traversal.
# MAGIC
# MAGIC Open [05-optional-lakebase/05.1-lakebase-graph-storage]($./05-optional-lakebase/05.1-lakebase-graph-storage) to explore Lakebase integration.
