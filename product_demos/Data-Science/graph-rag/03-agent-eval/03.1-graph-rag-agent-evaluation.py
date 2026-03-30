# Databricks notebook source
# MAGIC %md
# MAGIC # Graph RAG Agent: Evaluation and Deployment
# MAGIC
# MAGIC In this notebook, we:
# MAGIC 1. Load and test the Graph RAG agent (multi-hop + content questions)
# MAGIC 2. Log the model with `mlflow.pyfunc.log_model()`
# MAGIC 3. Generate synthetic eval data + manual multi-hop questions
# MAGIC 4. Evaluate with scorers (Groundedness, Relevance, Safety, Guidelines)
# MAGIC 5. **Compare with vector-only baseline** to demonstrate Graph RAG improvement
# MAGIC 6. Register to Unity Catalog and deploy
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=03.1-graph-rag-agent-evaluation&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# DBTITLE 1,Library Installs
# MAGIC %pip install -U -qqqq mlflow>=3.10.1 langchain==0.3.27 langgraph==0.6.11 databricks-langchain pydantic databricks-agents unitycatalog-langchain[databricks] databricks-feature-engineering==0.12.1 protobuf<5 cryptography<43 graphframes
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load and Test the Agent
# MAGIC
# MAGIC Let's load our agent and test it with different types of questions.

# COMMAND ----------

import mlflow
import yaml, sys, os
import mlflow.models

# Add the agent path
agent_eval_path = os.path.abspath(os.path.join(os.getcwd(), "../03-agent-eval"))
sys.path.append(agent_eval_path)

# Set experiment
mlflow.set_experiment(os.getcwd()+"/03.1-graph-rag-agent-evaluation")

# Load the agent config
conf_path = os.path.join(agent_eval_path, 'agent_config.yaml')
model_config = mlflow.models.ModelConfig(development_config=conf_path)

# COMMAND ----------

from agent import AGENT

# List loaded tools
print("Agent tools:", AGENT.list_tools())

# COMMAND ----------

# DBTITLE 1,Test: Multi-hop question (requires graph traversal)
request_multihop = "What solutions exist for errors related to the Router X500 series?"
answer = AGENT.predict({"input": [{"role": "user", "content": request_multihop}]})
print(answer['output'][-1]['content'][-1]['text'])

# COMMAND ----------

# DBTITLE 1,Test: Content question (uses vector search)
request_content = "How do I configure WiFi 6E on the Router X500?"
answer = AGENT.predict({"input": [{"role": "user", "content": request_content}]})
print(answer['output'][-1]['content'][-1]['text'])

# COMMAND ----------

# DBTITLE 1,Test: Reverse traversal question
request_reverse = "Which products share the VLAN support feature?"
answer = AGENT.predict({"input": [{"role": "user", "content": request_reverse}]})
print(answer['output'][-1]['content'][-1]['text'])

# COMMAND ----------

# DBTITLE 1,Test: Hybrid question (error + fix)
request_hybrid = "What is ERR-012 and how do I fix it?"
answer = AGENT.predict({"input": [{"role": "user", "content": request_hybrid}]})
print(answer['output'][-1]['content'][-1]['text'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Log the Agent Model

# COMMAND ----------

# Agent captures required resources for agent execution
for r in AGENT.get_resources():
    print(f"Resource: {type(r).__name__}:{r.name}")

# COMMAND ----------

with mlflow.start_run(run_name=model_config.get('config_version_name')):
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agent",
        python_model=agent_eval_path+"/agent.py",
        model_config=conf_path,
        input_example={"input": [{"role": "user", "content": request_multihop}]},
        resources=AGENT.get_resources(),
        extra_pip_requirements=["databricks-connect", "graphframes"]
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Generate Evaluation Dataset
# MAGIC
# MAGIC We create an evaluation dataset combining:
# MAGIC - Synthetic questions generated from document chunks
# MAGIC - Manual multi-hop reasoning questions that specifically test graph traversal

# COMMAND ----------

from databricks.agents.evals import generate_evals_df

docs = spark.table('document_chunks')

agent_description = """
The Agent is a Graph RAG assistant that answers technical questions about networking products
(routers, switches, modems, access points, firewalls). It has access to a knowledge graph
containing products, features, errors, and solutions with their relationships, as well as
a vector search over technical documentation. It excels at multi-hop reasoning questions
that require traversing entity relationships (e.g., finding all solutions for all errors
affecting a specific product).
"""

question_guidelines = """
# User personas
- A network engineer troubleshooting equipment issues
- A support technician looking up product capabilities
- A customer asking about product features and error resolution

# Example questions
- What solutions exist for errors related to the Router X500 series?
- Which products support WiFi 6E?
- How do I fix ERR-012 Memory Exhaustion on the Switch SW-3000?

# Additional Guidelines
- Questions should be succinct and human-like
- Include questions that require multi-hop reasoning (product -> error -> solution)
"""

# Generate synthetic eval dataset
evals = generate_evals_df(
    docs,
    num_evals=10,
    agent_description=agent_description,
    question_guidelines=question_guidelines
)
evals["inputs"] = evals["inputs"].apply(lambda x: {"question": x["messages"][0]["content"]})

# COMMAND ----------

# Add manual multi-hop questions that specifically test graph traversal
import pandas as pd

manual_evals = pd.DataFrame({
    "inputs": [
        {"question": "What solutions exist for errors related to the Router X500 series?"},
        {"question": "Which products share the VLAN support feature?"},
        {"question": "What is ERR-012 and how do I fix it?"},
        {"question": "What errors can occur on the Switch SW-3000 and what are their solutions?"},
        {"question": "Which products support PoE and what PoE-related errors might I encounter?"},
        {"question": "List all features of the Firewall FW-2000 and any related security issues."},
    ]
})

# Combine synthetic and manual eval datasets
full_evals = pd.concat([evals, manual_evals], ignore_index=True)
display(full_evals)

# COMMAND ----------

# Save to MLflow evaluation dataset
eval_dataset_table_name = f"{catalog}.{db}.graph_rag_mlflow_eval"
eval_dataset = mlflow.genai.datasets.create_dataset(eval_dataset_table_name)
eval_dataset.merge_records(full_evals)
print(f"Saved {len(full_evals)} evaluation records to {eval_dataset_table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Evaluate the Graph RAG Agent

# COMMAND ----------

from mlflow.genai.scorers import RetrievalGroundedness, RelevanceToQuery, Safety, Guidelines
import pandas as pd

eval_dataset = mlflow.genai.datasets.get_dataset(eval_dataset_table_name)
scorers = get_scorers()

# Load the model and create a prediction function
loaded_model = mlflow.pyfunc.load_model(f"runs:/{logged_agent_info.run_id}/agent")

def predict_wrapper(question):
    model_input = pd.DataFrame({
        "input": [[{"role": "user", "content": question}]]
    })
    response = loaded_model.predict(model_input)
    return response['output'][-1]['content'][-1]['text']

print("Running Graph RAG agent evaluation...")
with mlflow.start_run(run_name='eval_graph_rag_agent'):
    results_graph_rag = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_wrapper, scorers=scorers)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Compare with Vector-Only Baseline
# MAGIC
# MAGIC To demonstrate the value of graph tools, let's evaluate a vector-only agent (same LLM and retriever, but no graph tools) on the same dataset.

# COMMAND ----------

# Create a vector-only config (remove graph tools, keep only retriever)
import yaml, copy

vector_only_config_path = "/tmp/vector_only_agent_config.yaml"
config = yaml.safe_load(open(conf_path))
vector_only_config = copy.deepcopy(config)
vector_only_config["config_version_name"] = "vector_only_baseline"
vector_only_config["uc_tool_names"] = []  # No UC tools (no graph functions)
vector_only_config["system_prompt"] = """You are a technical support assistant for networking products. You have access to a document search tool.
Answer questions using the information found in the technical documentation.
Provide clear, actionable answers. Do NOT mention tools or reasoning steps."""

yaml.dump(vector_only_config, open(vector_only_config_path, "w"))

# COMMAND ----------

# Create a simple vector-only agent for comparison
from databricks_langchain import ChatDatabricks, VectorSearchRetrieverTool

llm = ChatDatabricks(endpoint=config["llm_endpoint_name"])
retriever_config = config["retriever_config"]

vs_tool = VectorSearchRetrieverTool(
    index_name=retriever_config["index_name"],
    name=retriever_config["tool_name"],
    description=retriever_config["description"],
    num_results=retriever_config["num_results"],
)

# Simple vector-only prediction function
def predict_vector_only(question):
    # Search documents
    from databricks.vector_search.client import VectorSearchClient
    vsc = VectorSearchClient(disable_notice=True)
    vs_index_fullname = retriever_config["index_name"]
    endpoint = VECTOR_SEARCH_ENDPOINT_NAME

    results = vsc.get_index(endpoint, vs_index_fullname).similarity_search(
        query_text=question, columns=["chunk_text"], num_results=3
    )
    docs = results.get('result', {}).get('data_array', [])
    context = "\n\n".join([doc[0] for doc in docs])

    # Ask LLM with retrieved context only
    messages = [
        {"role": "system", "content": vector_only_config["system_prompt"]},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
    ]
    response = llm.invoke(messages)
    return response.content

# COMMAND ----------

print("Running vector-only baseline evaluation...")
with mlflow.start_run(run_name='eval_vector_only_baseline'):
    results_vector_only = mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_vector_only, scorers=scorers)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Evaluation Comparison
# MAGIC
# MAGIC Compare the evaluation results side-by-side. The Graph RAG agent should outperform the vector-only baseline on multi-hop reasoning questions.

# COMMAND ----------

# Display comparison summary
print("=" * 60)
print("EVALUATION COMPARISON")
print("=" * 60)
print(f"\nGraph RAG Agent metrics:")
for k, v in results_graph_rag.metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

print(f"\nVector-Only Baseline metrics:")
for k, v in results_vector_only.metrics.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register and Deploy the Agent
# MAGIC
# MAGIC Now that we've validated the Graph RAG agent outperforms vector-only, let's register it to Unity Catalog and deploy.

# COMMAND ----------

from mlflow import MlflowClient

UC_MODEL_NAME = f"{catalog}.{db}.{MODEL_NAME}"

# Register the model to UC
client = MlflowClient()
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri,
    name=UC_MODEL_NAME,
    tags={"model": "graph_rag_agent", "model_version": "with_graph_tools_and_retriever"}
)

client.set_registered_model_alias(name=UC_MODEL_NAME, alias="model-to-deploy", version=uc_registered_model_info.version)
displayHTML(f'<a href="/explore/data/models/{catalog}/{db}/{MODEL_NAME}" target="_blank">Open Unity Catalog to see Registered Agent</a>')

# COMMAND ----------

from databricks import agents

endpoint_name = f'{MODEL_NAME}_{catalog}_{db}'[:60]

if len(agents.get_deployments(model_name=UC_MODEL_NAME, model_version=uc_registered_model_info.version)) == 0:
    agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, endpoint_name=endpoint_name, tags={"project": "dbdemos"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: Deploy a Chatbot Frontend
# MAGIC
# MAGIC Now that our Graph RAG agent is deployed as a model serving endpoint, let's create a chatbot interface for end users.
# MAGIC
# MAGIC Open [04-deploy-app/04-Deploy-Frontend-Lakehouse-App]($../04-deploy-app/04-Deploy-Frontend-Lakehouse-App) to continue.
