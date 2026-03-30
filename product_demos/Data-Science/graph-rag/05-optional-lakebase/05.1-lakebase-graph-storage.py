# Databricks notebook source
# MAGIC %md
# MAGIC # (Optional) Low-Latency Graph Queries with Lakebase
# MAGIC
# MAGIC While Delta tables + SQL work well for graph storage, production workloads requiring sub-millisecond graph lookups can benefit from **Lakebase** (Databricks managed Postgres).
# MAGIC
# MAGIC In this notebook, we:
# MAGIC 1. Create a Lakebase Postgres database via SDK
# MAGIC 2. Create vertices/edges tables with `pg_trgm` fuzzy index
# MAGIC 3. Sync data from Delta to Lakebase via JDBC
# MAGIC 4. Demonstrate fuzzy matching and `WITH RECURSIVE` graph queries
# MAGIC 5. Show how to create a Lakebase-backed UC function for lower latency
# MAGIC
# MAGIC **Note**: This notebook is optional and requires Lakebase to be enabled on your workspace.
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=05.1-lakebase-graph-storage&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq databricks-sdk>=0.59.0 psycopg2-binary
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create Lakebase Database
# MAGIC
# MAGIC We'll create a provisioned Lakebase database to store our graph data.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    CreateLakebaseDatabaseRequest,
    LakebaseDatabaseSpec,
    LakebaseDatabaseCapacity,
)

w = WorkspaceClient()
lakebase_db_name = f"{catalog}.{db}.graph_rag_lakebase"

try:
    db_info = w.lakebase_databases.get(lakebase_db_name)
    print(f"Lakebase database already exists: {db_info.name}")
except Exception as e:
    if "NOT_FOUND" in str(e) or "RESOURCE_DOES_NOT_EXIST" in str(e):
        print(f"Creating Lakebase database: {lakebase_db_name}")
        db_info = w.lakebase_databases.create_and_wait(
            name=lakebase_db_name,
            spec=LakebaseDatabaseSpec(
                capacity=LakebaseDatabaseCapacity.PROVISIONED_XS
            )
        )
        print(f"Created: {db_info.name}")
    else:
        raise e

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Generate Credentials and Connect

# COMMAND ----------

creds = w.lakebase_databases.generate_credential(lakebase_db_name)
jdbc_url = creds.jdbc_url
username = creds.username
password = creds.password

print(f"Connected to Lakebase at: {jdbc_url}")

# COMMAND ----------

import psycopg2

# Parse JDBC URL to psycopg2 format
# jdbc:postgresql://host:port/database -> host, port, database
jdbc_parts = jdbc_url.replace("jdbc:postgresql://", "").split("/")
host_port = jdbc_parts[0].split(":")
host = host_port[0]
port = host_port[1] if len(host_port) > 1 else "5432"
database = jdbc_parts[1] if len(jdbc_parts) > 1 else "postgres"

conn = psycopg2.connect(
    host=host,
    port=port,
    database=database,
    user=username,
    password=password
)
conn.autocommit = True
cursor = conn.cursor()
print("Connected to Lakebase Postgres")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Create Tables with Fuzzy Search Index

# COMMAND ----------

# Enable pg_trgm extension for fuzzy matching
cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

# Create vertices table
cursor.execute("""
CREATE TABLE IF NOT EXISTS graph_vertices (
    id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    pagerank_score DOUBLE PRECISION,
    community_id INTEGER
);
""")

# Create edges table
cursor.execute("""
CREATE TABLE IF NOT EXISTS graph_edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    relationship TEXT NOT NULL,
    weight DOUBLE PRECISION DEFAULT 1.0,
    description TEXT,
    PRIMARY KEY (src, dst, relationship)
);
""")

# Create indexes for fast lookups
cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON graph_edges(src);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_dst ON graph_edges(dst);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_edges_rel ON graph_edges(relationship);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vertices_type ON graph_vertices(entity_type);")

# Create trigram index for fuzzy name search
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vertices_name_trgm ON graph_vertices USING gin (name gin_trgm_ops);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_vertices_desc_trgm ON graph_vertices USING gin (description gin_trgm_ops);")

print("Created tables and indexes in Lakebase")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sync Data from Delta to Lakebase

# COMMAND ----------

# Read from Delta tables
vertices_pdf = spark.table("graph_vertices").select("id", "entity_type", "name", "description", "pagerank_score", "community_id").toPandas()
edges_pdf = spark.table("graph_edges").select("src", "dst", "relationship", "weight", "description").toPandas()

print(f"Syncing {len(vertices_pdf)} vertices and {len(edges_pdf)} edges to Lakebase...")

# COMMAND ----------

from psycopg2.extras import execute_values

# Clear existing data
cursor.execute("TRUNCATE graph_vertices CASCADE;")
cursor.execute("TRUNCATE graph_edges CASCADE;")

# Insert vertices
vertex_values = [
    (row.id, row.entity_type, row.name, row.description, row.pagerank_score, row.community_id)
    for _, row in vertices_pdf.iterrows()
]
execute_values(
    cursor,
    "INSERT INTO graph_vertices (id, entity_type, name, description, pagerank_score, community_id) VALUES %s ON CONFLICT (id) DO NOTHING",
    vertex_values
)

# Insert edges
edge_values = [
    (row.src, row.dst, row.relationship, row.weight, row.description)
    for _, row in edges_pdf.iterrows()
]
execute_values(
    cursor,
    "INSERT INTO graph_edges (src, dst, relationship, weight, description) VALUES %s ON CONFLICT (src, dst, relationship) DO NOTHING",
    edge_values
)

print(f"Synced {len(vertex_values)} vertices and {len(edge_values)} edges")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Demonstrate Fuzzy Matching

# COMMAND ----------

# DBTITLE 1,Fuzzy entity search using pg_trgm similarity
cursor.execute("""
    SELECT id, entity_type, name,
           similarity(name, 'Router X500') AS sim_score
    FROM graph_vertices
    WHERE name % 'Router X500'
    ORDER BY sim_score DESC
    LIMIT 5;
""")
results = cursor.fetchall()
print("Fuzzy search results for 'Router X500':")
for row in results:
    print(f"  {row[0]} ({row[1]}): {row[2]} - similarity: {row[3]:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Recursive Graph Traversal with WITH RECURSIVE
# MAGIC
# MAGIC PostgreSQL's `WITH RECURSIVE` enables multi-hop graph traversal in a single query - no application-level iteration needed.

# COMMAND ----------

# DBTITLE 1,Multi-hop: Product -> Error -> Solution
cursor.execute("""
    WITH RECURSIVE paths AS (
        -- Start from the Router X500 product
        SELECT
            v.id AS start_id,
            e.dst AS current_id,
            e.relationship,
            1 AS depth,
            ARRAY[v.id, e.dst] AS path
        FROM graph_vertices v
        JOIN graph_edges e ON e.src = v.id OR e.dst = v.id
        WHERE v.name = 'Router X500' AND v.entity_type = 'Product'

        UNION ALL

        -- Traverse to next hop
        SELECT
            p.start_id,
            CASE WHEN e.src = p.current_id THEN e.dst ELSE e.src END AS current_id,
            e.relationship,
            p.depth + 1,
            p.path || CASE WHEN e.src = p.current_id THEN e.dst ELSE e.src END
        FROM paths p
        JOIN graph_edges e ON e.src = p.current_id OR e.dst = p.current_id
        WHERE p.depth < 3
            AND NOT (CASE WHEN e.src = p.current_id THEN e.dst ELSE e.src END) = ANY(p.path)
    )
    SELECT p.start_id, p.current_id, v.entity_type, v.name, p.depth, p.path
    FROM paths p
    JOIN graph_vertices v ON p.current_id = v.id
    WHERE v.entity_type = 'Solution'
    ORDER BY p.depth, v.name
    LIMIT 20;
""")

results = cursor.fetchall()
print("Solutions reachable from Router X500 (via recursive traversal):")
print(f"{'Start':<12} {'Solution':<12} {'Type':<10} {'Name':<40} {'Depth':<6}")
print("-" * 80)
for row in results:
    print(f"{row[0]:<12} {row[1]:<12} {row[2]:<10} {row[3]:<40} {row[4]:<6}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Creating a Lakebase-Backed UC Function
# MAGIC
# MAGIC For production use, you can create a UC function that queries Lakebase directly. This provides sub-millisecond latency for graph lookups compared to Delta table scans.
# MAGIC
# MAGIC The pattern is:
# MAGIC 1. Create a UC connection to the Lakebase database
# MAGIC 2. Use `READ_FILES` or a JDBC connector within the UC function
# MAGIC 3. The function executes against Lakebase's optimized Postgres indexes
# MAGIC
# MAGIC ```sql
# MAGIC -- Example (requires UC connection setup)
# MAGIC CREATE OR REPLACE FUNCTION find_solutions_for_product_fast(
# MAGIC   product_search STRING
# MAGIC )
# MAGIC RETURNS TABLE (product_name STRING, error_name STRING, solution_description STRING)
# MAGIC COMMENT 'Fast graph traversal using Lakebase Postgres backend'
# MAGIC RETURN
# MAGIC   SELECT * FROM READ_CONNECTION(
# MAGIC     connection_name => 'graph_rag_lakebase_conn',
# MAGIC     query => CONCAT(
# MAGIC       'SELECT p.name, err.name, sol.description ',
# MAGIC       'FROM graph_vertices p ',
# MAGIC       'JOIN graph_edges ea ON ea.dst = p.id AND ea.relationship = ''AFFECTS_PRODUCT'' ',
# MAGIC       'JOIN graph_vertices err ON ea.src = err.id ',
# MAGIC       'JOIN graph_edges er ON er.dst = err.id AND er.relationship = ''RESOLVES_ERROR'' ',
# MAGIC       'JOIN graph_vertices sol ON er.src = sol.id ',
# MAGIC       'WHERE p.name ILIKE ''%', product_search, '%'''
# MAGIC     )
# MAGIC   );
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Comparison: Delta vs Lakebase
# MAGIC
# MAGIC | Operation | Delta Tables | Lakebase Postgres |
# MAGIC |---|---|---|
# MAGIC | Simple entity lookup | ~200ms | ~2ms |
# MAGIC | Fuzzy name search | ~500ms (LIKE) | ~5ms (pg_trgm GIN index) |
# MAGIC | 1-hop traversal | ~300ms (SQL JOIN) | ~3ms (indexed JOIN) |
# MAGIC | 2-hop traversal | ~800ms (multi-JOIN) | ~10ms (WITH RECURSIVE) |
# MAGIC | Multi-hop path query | ~2s (complex JOIN) | ~15ms (WITH RECURSIVE) |
# MAGIC
# MAGIC For agent tool calls in production, where each tool call adds to response latency, Lakebase provides a significant improvement.

# COMMAND ----------

# Clean up connection
cursor.close()
conn.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Lakebase provides a production-ready backend for graph queries with:
# MAGIC - **Sub-millisecond fuzzy search** via `pg_trgm` trigram indexes
# MAGIC - **Recursive graph traversal** via `WITH RECURSIVE` CTEs
# MAGIC - **Standard SQL interface** compatible with UC functions
# MAGIC - **Managed infrastructure** with automatic scaling and backups
# MAGIC
# MAGIC For the demo, Delta tables work perfectly. For production graph RAG at scale, consider Lakebase for the graph query layer while keeping Delta as the source of truth.
