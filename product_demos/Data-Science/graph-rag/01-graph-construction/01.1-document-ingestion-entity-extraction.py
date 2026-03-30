# Databricks notebook source
# MAGIC %md
# MAGIC # Document Ingestion & Entity Extraction for Graph RAG
# MAGIC
# MAGIC In this notebook, we:
# MAGIC 1. Parse PDF technical documentation using `ai_parse_document()`
# MAGIC 2. Create a `document_chunks` table with Change Data Feed enabled (for Vector Search)
# MAGIC 3. Extract entities (Products, Features, Errors, Solutions) using `AI_QUERY()`
# MAGIC 4. Extract relationships between entities
# MAGIC 5. Store results in staging tables `entities_raw` and `relationships_raw`
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=01.1-document-ingestion-entity-extraction&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq mlflow>=3.10.1 langchain==0.3.27 langgraph==0.6.11 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] databricks-feature-engineering==0.12.1 protobuf<5 cryptography<43 graphframes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Inspect the PDF Documents
# MAGIC
# MAGIC Our technical documentation PDFs are stored in the volume. Let's see what we have.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT path FROM READ_FILES('/Volumes/main_build/dbdemos_graph_rag/raw_data/pdf_documentation/', format => 'binaryFile') LIMIT 5

# COMMAND ----------

# DBTITLE 1,Parse a sample PDF with ai_parse_document
# MAGIC %sql
# MAGIC -- ai_parse_document is available in DBR 17.1 or serverless runtime
# MAGIC SELECT ai_parse_document(content) AS parsed_document
# MAGIC   FROM READ_FILES('/Volumes/main_build/dbdemos_graph_rag/raw_data/pdf_documentation/', format => 'binaryFile') LIMIT 2

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Create the Document Chunks Table
# MAGIC
# MAGIC We create a `document_chunks` table with Change Data Feed enabled. This table will be the source for our Vector Search index.
# MAGIC
# MAGIC Since our PDFs are relatively small product docs, we'll keep each document as a single chunk to preserve full context. For larger documents, you'd want to split into overlapping chunks.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE TABLE IF NOT EXISTS document_chunks (
# MAGIC   chunk_id BIGINT GENERATED ALWAYS AS IDENTITY,
# MAGIC   doc_uri STRING,
# MAGIC   product_name STRING,
# MAGIC   chunk_text STRING,
# MAGIC   chunk_index INT)
# MAGIC   TBLPROPERTIES (delta.enableChangeDataFeed = true);

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Ingest PDFs into Document Chunks
# MAGIC
# MAGIC Parse PDFs with `ai_parse_document()` and extract product name with `ai_extract()`.

# COMMAND ----------

# MAGIC %sql
# MAGIC INSERT OVERWRITE TABLE document_chunks (doc_uri, product_name, chunk_text, chunk_index)
# MAGIC SELECT doc_uri, ai_extract.product_name, content, 0 AS chunk_index
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     ai_extract(content, array('product_name')) AS ai_extract,
# MAGIC     content,
# MAGIC     doc_uri
# MAGIC   FROM (
# MAGIC     SELECT array_join(
# MAGIC             transform(parsed_document:document.elements::ARRAY<STRUCT<content:STRING>>, x -> x.content), '\n') AS content,
# MAGIC            path as doc_uri
# MAGIC     FROM (
# MAGIC       SELECT ai_parse_document(content) AS parsed_document, path
# MAGIC       FROM READ_FILES('/Volumes/main_build/dbdemos_graph_rag/raw_data/pdf_documentation/', format => 'binaryFile')
# MAGIC     )
# MAGIC   )
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Check document chunks
# MAGIC %sql
# MAGIC SELECT chunk_id, product_name, doc_uri, LENGTH(chunk_text) AS text_length FROM document_chunks

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Extract Entities Using AI_QUERY
# MAGIC
# MAGIC Now we use `AI_QUERY()` to extract structured entities from each document chunk. The LLM will identify Products, Features, Errors, and Solutions mentioned in the text.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE entities_raw AS
# MAGIC SELECT
# MAGIC   chunk_id,
# MAGIC   product_name,
# MAGIC   inline(AI_QUERY(
# MAGIC     'databricks-claude-3-7-sonnet',
# MAGIC     CONCAT(
# MAGIC       'Extract all entities from this technical documentation. Return a JSON array of objects with fields: entity_id (e.g. PROD-001, ERR-012, FEAT-003, SOL-005), entity_type (one of: Product, Feature, Error, Solution), name, description (brief). Only extract entities explicitly mentioned in the text.\n\nText:\n',
# MAGIC       chunk_text
# MAGIC     ),
# MAGIC     responseFormat => 'ARRAY<STRUCT<entity_id: STRING, entity_type: STRING, name: STRING, description: STRING>>'
# MAGIC   )) AS (entity_id, entity_type, name, description)
# MAGIC FROM document_chunks;

# COMMAND ----------

# DBTITLE 1,Check extracted entities
# MAGIC %sql
# MAGIC SELECT entity_type, COUNT(*) as count FROM entities_raw GROUP BY entity_type ORDER BY count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM entities_raw LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Extract Relationships Using AI_QUERY
# MAGIC
# MAGIC Next, we extract relationships between the entities we found. The LLM identifies how entities relate to each other.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE relationships_raw AS
# MAGIC SELECT
# MAGIC   chunk_id,
# MAGIC   inline(AI_QUERY(
# MAGIC     'databricks-claude-3-7-sonnet',
# MAGIC     CONCAT(
# MAGIC       'Given these entities and the source text, extract all relationships between entities. Return a JSON array of objects with fields: src (source entity_id), dst (target entity_id), relationship (one of: HAS_FEATURE, AFFECTS_PRODUCT, RESOLVES_ERROR, RELATED_PRODUCT, APPLIES_TO_PRODUCT), description (brief description of the relationship).\n\nEntities:\n',
# MAGIC       entities_json,
# MAGIC       '\n\nSource text:\n',
# MAGIC       chunk_text
# MAGIC     ),
# MAGIC     responseFormat => 'ARRAY<STRUCT<src: STRING, dst: STRING, relationship: STRING, description: STRING>>'
# MAGIC   )) AS (src, dst, relationship, description)
# MAGIC FROM (
# MAGIC   SELECT
# MAGIC     dc.chunk_id,
# MAGIC     dc.chunk_text,
# MAGIC     TO_JSON(COLLECT_LIST(STRUCT(er.entity_id, er.entity_type, er.name))) AS entities_json
# MAGIC   FROM document_chunks dc
# MAGIC   JOIN entities_raw er ON dc.chunk_id = er.chunk_id
# MAGIC   GROUP BY dc.chunk_id, dc.chunk_text
# MAGIC );

# COMMAND ----------

# DBTITLE 1,Check extracted relationships
# MAGIC %sql
# MAGIC SELECT relationship, COUNT(*) as count FROM relationships_raw GROUP BY relationship ORDER BY count DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM relationships_raw LIMIT 20

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC We've successfully:
# MAGIC 1. Parsed PDF technical documentation into text chunks
# MAGIC 2. Created a `document_chunks` table with CDF enabled for Vector Search
# MAGIC 3. Extracted entities (Products, Features, Errors, Solutions) using AI
# MAGIC 4. Extracted relationships between entities
# MAGIC
# MAGIC Next, we'll build the graph tables from these raw extractions.
# MAGIC
# MAGIC Open [01.2-build-graph-tables]($./01.2-build-graph-tables) to continue.
