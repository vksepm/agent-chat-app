import logging
import os
from smolagents import CodeAgent, AzureOpenAIModel
from dotenv import load_dotenv

load_dotenv(override=True)

# This captures all outgoing requests and incoming responses from httpx
logging.basicConfig(
    format="%(levelname)s [%(asctime)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG
)

# Explicitly set the httpx logger to DEBUG to see full URL and status codes
logging.getLogger("httpx").setLevel(logging.DEBUG)

# 1. Initialize the Model
# Ensure these environment variables match your HTTP request details:
# MODEL_ID: gpt-4.1-2025-04-14
# AZURE_ENDPOINT: https://example.com
# AZURE_API_VERSION: 2025-04-14
model = AzureOpenAIModel(
    model_id=os.environ.get("MODEL_ID", ""),
    azure_endpoint=os.environ.get("AZURE_ENDPOINT", ""),
    api_key=os.environ.get("MODEL_API_KEY", ""),
    api_version=os.environ.get("AZURE_API_VERSION", "")
)

# 2. Initialize the Agent
# We use CodeAgent as it's the standard for smolagents
agent = CodeAgent(
    tools=[], 
    model=model,
    verbosity_level=2)

# 3. Run a Test Task
try:
    print("--- Starting Agent Test ---")
    response = agent.run("What is the capital of France?")
    print(f"Agent Response: {response}")
    print("--- Test Successful ---")
except Exception as e:
    print(f"--- Test Failed ---")
    print(f"Error: {e}")
