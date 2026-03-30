# Databricks notebook source
# MAGIC %md
# MAGIC ## Demo bundle configuration
# MAGIC Please ignore / do not delete, only used to prep and bundle the demo

# COMMAND ----------

{
  "name": "graph-rag",
  "category": "data-science",
  "custom_schema_supported": True,
  "default_catalog": "main",
  "default_schema": "dbdemos_graph_rag",
  "serverless_supported": True,
  "title": "Graph RAG: Knowledge Graphs + Vector Search for Multi-Hop Reasoning",
  "description": "Build a Graph RAG system combining knowledge graphs (Delta + GraphFrames) with Vector Search to enable multi-hop reasoning over technical documentation using Mosaic AI Agent Framework",
  "bundle": True,
  "env_version": 2,
  "notebooks": [
    {
      "path": "_resources/01-setup",
      "pre_run": False,
      "publish_on_website": False,
      "add_cluster_setup_cell": False,
      "title":  "Setup",
      "description": "Demo setup"
    },
    {
      "path": "_resources/02-data-generation",
      "pre_run": False,
      "publish_on_website": False,
      "add_cluster_setup_cell": False,
      "title":  "Data Generation",
      "description": "Generate synthetic product documentation data"
    },
    {
      "path": "01-graph-construction/01.1-document-ingestion-entity-extraction",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Document Ingestion & Entity Extraction",
      "description": "Parse PDFs and extract entities and relationships using AI"
    },
    {
      "path": "01-graph-construction/01.2-build-graph-tables",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Build Graph Tables",
      "description": "Create graph vertices and edges, run PageRank and community detection"
    },
    {
      "path": "02-indexing-and-retrieval/02.1-vector-search-index",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Vector Search Index",
      "description": "Create vector search index on document chunks"
    },
    {
      "path": "02-indexing-and-retrieval/02.2-graph-retrieval-tools",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Graph Retrieval Tools",
      "description": "Register UC functions for graph traversal"
    },
    {
      "path": "03-agent-eval/agent",
      "pre_run": False,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Graph RAG Agent",
      "description": "LangGraph agent with graph tools and vector search"
    },
    {
      "path": "03-agent-eval/03.1-graph-rag-agent-evaluation",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Agent Evaluation",
      "description": "Evaluate Graph RAG agent and compare with vector-only baseline"
    },
    {
      "path": "04-deploy-app/04-Deploy-Frontend-Lakehouse-App",
      "pre_run": True,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Deploy Chatbot App",
      "description": "Deploy Gradio frontend using Lakehouse Apps"
    },
    {
      "path": "04-deploy-app/chatbot_app",
      "pre_run": False,
      "publish_on_website": False,
      "add_cluster_setup_cell": False,
      "title":  "Chatbot app",
      "description": "Gradio application folder"
    },
    {
      "path": "05-optional-lakebase/05.1-lakebase-graph-storage",
      "pre_run": False,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Lakebase Graph Storage",
      "description": "Optional: low-latency graph queries with Lakebase"
    },
    {
      "path": "00-graph-rag-introduction",
      "pre_run": False,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Introduction",
      "description": "Start here."
    },
    {
      "path": "config",
      "pre_run": False,
      "publish_on_website": True,
      "add_cluster_setup_cell": False,
      "title":  "Config",
      "description": "Configuration file"
    }
  ],
  "cluster": {
    "num_workers": 0,
    "spark_version": "17.1.x-scala2.13",
    "spark_conf": {
        "spark.master": "local[*, 4]"
    },
    "single_user_name": "{{CURRENT_USER}}",
    "data_security_mode": "SINGLE_USER"
  }
}
