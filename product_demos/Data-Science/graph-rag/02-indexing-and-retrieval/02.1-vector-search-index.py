# Databricks notebook source
# MAGIC %md
# MAGIC # Vector Search Index for Document Chunks
# MAGIC
# MAGIC In this notebook, we:
# MAGIC 1. Create/verify a Vector Search endpoint
# MAGIC 2. Create a Delta Sync index on `document_chunks`
# MAGIC 3. Test similarity search
# MAGIC 4. **Demonstrate the limitations of vector-only search** for multi-hop reasoning questions
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=02.1-vector-search-index&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq mlflow>=3.10.1 langchain==0.3.27 langgraph==0.6.11 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] databricks-feature-engineering==0.12.1 protobuf<5 cryptography<43
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 1. Vector Search Endpoint
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-2.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC Vector search endpoints serve your indexes. Let's create one if it doesn't exist yet.

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
vsc = VectorSearchClient(disable_notice=True)

if not endpoint_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME):
    vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")

wait_for_vs_endpoint_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME)
print(f"Endpoint named {VECTOR_SEARCH_ENDPOINT_NAME} is ready.")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC
# MAGIC ## 2. Create the Vector Search Index
# MAGIC
# MAGIC <img src="https://github.com/databricks-demos/dbdemos-resources/blob/main/images/product/chatbot-rag/rag-basic-prep-3.png?raw=true" style="float: right; margin-left: 10px" width="400px">
# MAGIC
# MAGIC We create a managed embeddings index on the `document_chunks` table. Databricks will automatically embed the `chunk_text` column using the GTE model and synchronize the index.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

# The table we'd like to index
source_table_fullname = f"{catalog}.{db}.document_chunks"
# Where we want to store our index
vs_index_fullname = f"{catalog}.{db}.document_chunks_vs_index"

if not index_exists(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname):
  print(f"Creating index {vs_index_fullname} on endpoint {VECTOR_SEARCH_ENDPOINT_NAME}...")
  vsc.create_delta_sync_index(
    endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
    index_name=vs_index_fullname,
    source_table_name=source_table_fullname,
    pipeline_type="TRIGGERED",
    primary_key="chunk_id",
    embedding_source_column='chunk_text',
    embedding_model_endpoint_name=EMBEDDING_MODEL_NAME
  )
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
else:
  wait_for_index_to_be_ready(vsc, VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname)
  vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).sync()

print(f"Index {vs_index_fullname} on table {source_table_fullname} is ready")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Test Similarity Search
# MAGIC
# MAGIC Let's verify our index works with a direct content question.

# COMMAND ----------

# DBTITLE 1,Content question: works well with vector search
question = "How do I configure WiFi 6E on the Router X500?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["chunk_id", "product_name", "chunk_text"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])

for doc in docs:
    print(f"Product: {doc[1]}, Score: {doc[-1]:.4f}")
    print(f"Text preview: {doc[2][:200]}...")
    print("---")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Demonstrating Vector Search Limitations
# MAGIC
# MAGIC Vector search works great for content-based questions. But what about **multi-hop reasoning** questions?
# MAGIC
# MAGIC Let's try a question that requires traversing entity relationships:

# COMMAND ----------

# DBTITLE 1,Multi-hop question: vector search struggles
question = "What solutions exist for errors related to the Router X500 series?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["chunk_id", "product_name", "chunk_text"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])

print("Vector search results for multi-hop question:")
print("=" * 60)
for doc in docs:
    print(f"Product: {doc[1]}, Score: {doc[-1]:.4f}")
    print(f"Text preview: {doc[2][:200]}...")
    print("---")

print("\nNotice: vector search returns documents that are semantically similar")
print("to the query, but it can't traverse the Product -> Error -> Solution")
print("relationship graph to systematically find ALL solutions for ALL errors")
print("affecting the Router X500.")

# COMMAND ----------

# DBTITLE 1,Another multi-hop question
question = "Which products share the VLAN support feature?"

results = vsc.get_index(VECTOR_SEARCH_ENDPOINT_NAME, vs_index_fullname).similarity_search(
  query_text=question,
  columns=["chunk_id", "product_name", "chunk_text"],
  num_results=3)
docs = results.get('result', {}).get('data_array', [])

print("Vector search results for reverse traversal question:")
print("=" * 60)
for doc in docs:
    print(f"Product: {doc[1]}, Score: {doc[-1]:.4f}")
    print("---")

print("\nVector search may find some relevant docs, but it cannot reliably")
print("enumerate ALL products with a specific feature - that requires")
print("graph traversal: Feature -> HAS_FEATURE <- Product.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Takeaway
# MAGIC
# MAGIC | Question Type | Vector Search | Graph RAG |
# MAGIC |---|---|---|
# MAGIC | "How do I configure WiFi 6E?" | Works well | Works well (uses VS) |
# MAGIC | "What solutions exist for Router X500 errors?" | Partial - finds similar docs | Excellent - traverses Product->Error->Solution |
# MAGIC | "Which products share VLAN support?" | Unreliable | Excellent - traverses Feature->Product |
# MAGIC | "What is ERR-012 and how do I fix it?" | Good for description | Excellent - gets error + all solutions |
# MAGIC
# MAGIC **This is why we need Graph RAG**: the graph tools complement vector search for relationship-aware reasoning.
# MAGIC
# MAGIC Next, let's register the graph retrieval tools as UC functions.
# MAGIC
# MAGIC Open [02.2-graph-retrieval-tools]($./02.2-graph-retrieval-tools) to continue.
