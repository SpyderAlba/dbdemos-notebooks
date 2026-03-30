# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Deploying the Graph RAG Chatbot with Lakehouse Applications
# MAGIC
# MAGIC Our Graph RAG agent is deployed as a model serving endpoint. Now let's create a chatbot frontend using Databricks Lakehouse Apps and Gradio.
# MAGIC
# MAGIC This frontend lets users ask both content-based questions (answered via vector search) and multi-hop relationship questions (answered via graph traversal).
# MAGIC
# MAGIC <!-- Collect usage data (view). Remove it to disable collection or disable tracker during installation. View README for more details.  -->
# MAGIC <img width="1px" src="https://ppxrzfxige.execute-api.us-west-2.amazonaws.com/v1/analytics?category=data-science&org_id=1444828305810485&notebook=04-Deploy-Frontend-Lakehouse-App&demo_name=graph-rag&event=VIEW">

# COMMAND ----------

# MAGIC %pip install --quiet -U mlflow[databricks]>=3.10.1 databricks-sdk>=0.59.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../_resources/01-setup

# COMMAND ----------

# MAGIC %md
# MAGIC ## Add your application configuration
# MAGIC
# MAGIC Lakehouse apps let you work with any Python framework. We'll create a configuration file with the model serving endpoint name.

# COMMAND ----------

print(f"The Databricks APP will be using the following model serving endpoint: {ENDPOINT_NAME}")

# COMMAND ----------

import yaml

# Our frontend application will hit the model endpoint we deployed
yaml_app_config = {"command": ["uvicorn", "main:app", "--workers", "1"],
                    "env": [{"name": "MODEL_SERVING_ENDPOINT", "value": ENDPOINT_NAME}]
                  }
try:
    with open('chatbot_app/app.yaml', 'w') as f:
        yaml.dump(yaml_app_config, f)
except Exception as e:
    print(f'pass to work on build job - {e}')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the Application
# MAGIC
# MAGIC Our application has 2 files under the `chatbot_app` folder:
# MAGIC - `main.py` containing our Gradio chatbot code
# MAGIC - `app.yaml` containing our configuration

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.apps import App, AppResource, AppResourceServingEndpoint, AppResourceServingEndpointServingEndpointPermission, AppDeployment

w = WorkspaceClient()
app_name = "dbdemos-graph-rag-app"

# COMMAND ----------

serving_endpoint = AppResourceServingEndpoint(name=ENDPOINT_NAME,
                                              permission=AppResourceServingEndpointServingEndpointPermission.CAN_QUERY
                                              )

rag_endpoint = AppResource(name="rag-endpoint", serving_endpoint=serving_endpoint)

rag_app = App(name=app_name,
              description="Graph RAG Technical Support Assistant",
              default_source_code_path=os.path.join(os.getcwd(), 'chatbot_app'),
              resources=[rag_endpoint])
try:
  app_details = w.apps.create_and_wait(app=rag_app)
  print(app_details)
except Exception as e:
  if "already exists" in str(e):
    print("App already exists, you can deploy it")
  else:
    raise e

# COMMAND ----------

# MAGIC %md
# MAGIC Once the app is created, we can deploy the code:

# COMMAND ----------

import mlflow

xp_name = os.getcwd().rsplit("/", 1)[0]+"/03-agent-eval/03.1-graph-rag-agent-evaluation"
mlflow.set_experiment(xp_name)

# COMMAND ----------

deployment = AppDeployment(source_code_path=os.path.join(os.getcwd(), 'chatbot_app'))

app_details = w.apps.deploy_and_wait(app_name=app_name, app_deployment=deployment)

# COMMAND ----------

# Let's access the application
w.apps.get(name=app_name).url

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC ## Your Graph RAG Chatbot is Ready!
# MAGIC
# MAGIC Open the UI to start asking both content-based and multi-hop relationship questions.
# MAGIC
# MAGIC **Try these example questions:**
# MAGIC - "What solutions exist for errors related to the Router X500 series?" (graph multi-hop)
# MAGIC - "How do I configure WiFi 6E on the Router X500?" (vector search)
# MAGIC - "Which products share the VLAN support feature?" (graph reverse traversal)
# MAGIC - "What is ERR-012 and how do I fix it?" (hybrid)
# MAGIC
# MAGIC ## Next: (Optional) Low-Latency Graph Queries with Lakebase
# MAGIC
# MAGIC For production workloads requiring sub-millisecond graph lookups, explore using Lakebase (managed Postgres).
# MAGIC
# MAGIC Open [05-optional-lakebase/05.1-lakebase-graph-storage]($../05-optional-lakebase/05.1-lakebase-graph-storage) to continue.
