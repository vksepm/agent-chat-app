"""
Debug script to diagnose if the agent is making duplicate tool calls.

This script helps identify:
1. Whether tool calls are actually duplicates (same arguments)
2. Whether observations from previous calls are being incorporated
3. Whether streaming mode properly handles multi-step reasoning

Run with: python debug_agent_loop.py
"""

import os
import sys
import logging

# Configure logging to see agent verbosity
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s [%(levelname)s] %(message)s'
)

os.environ['MODEL_TYPE'] = 'litellm'
os.environ['MODEL_ID'] = 'gpt-4-mini'
os.environ['MODEL_API_KEY'] = 'sk-test'
os.environ['MCP_SERVER_URL_1'] = 'https://example.com/1'
os.environ['MCP_SERVER_URL_2'] = 'https://example.com/2'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk'
os.environ['AGENT_VERBOSITY_LEVEL'] = '2'  # DEBUG

from smolagents import ToolCallingAgent, Tool
from src.agent import build_agent
from src.config import load_config

class WeatherTool(Tool):
    """Simulates a weather API tool."""
    name = "get_weather"
    description = "Get current weather for a location"
    inputs = {
        "location": {
            "description": "City name (e.g., 'Paris', 'London')",
            "type": "string"
        }
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.call_count = 0
        self.calls = []  # Track all calls
    
    def forward(self, location: str) -> str:
        self.call_count += 1
        self.calls.append({
            'call_num': self.call_count,
            'location': location,
        })
        return f"Weather in {location}: Sunny, 25°C"

# Create and test agent
print("=" * 80)
print("DEBUG: Agent Tool Call Analysis")
print("=" * 80)
print()

config = load_config()
config.agent_max_steps = 3

weather_tool = WeatherTool()
agent = build_agent([weather_tool], config)

# Test scenario: ask agent to check weather for multiple cities
prompt = "What's the weather in Paris and London? Check Paris first, then use that information to check London."

print(f"PROMPT: {prompt}")
print()
print("-" * 80)
print("RUNNING AGENT (streaming mode)")
print("-" * 80)
print()

# Run in streaming mode (like the app does)
from smolagents.agents import ToolOutput

events = []
tool_responses = {}

try:
    for event in agent.run(prompt, stream=True, reset=False):
        event_type = type(event).__name__
        events.append((event_type, event))
        
        if event_type == 'ToolOutput':
            tool_id = getattr(event, 'id', '')
            observation = getattr(event, 'observation', '')
            tool_responses[tool_id] = observation
            print(f"  [{event_type}] Tool Response: {observation}")
        elif event_type in ['ActionStep', 'PlanningStep', 'FinalAnswerStep']:
            print(f"  [{event_type}]")
        elif event_type == 'ChatMessageStreamDelta':
            print(f"  [{event_type}]")
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("-" * 80)
print("ANALYSIS")
print("-" * 80)
print()

print(f"Total tool calls made: {weather_tool.call_count}")
print()
print("Tool call history:")
for call in weather_tool.calls:
    print(f"  Call #{call['call_num']}: location='{call['location']}'")

print()
print("Agent memory steps:")
for i, step in enumerate(agent.memory.steps):
    step_type = type(step).__name__
    print(f"  Step {i}: {step_type}")
    
    if hasattr(step, 'tool_calls'):
        for j, tc in enumerate(step.tool_calls):
            tc_name = getattr(tc, 'name', 'unknown')
            tc_args = getattr(tc, 'arguments', {})
            print(f"    Tool call {j}: name={tc_name}, args={tc_args}")
    
    if hasattr(step, 'observations'):
        obs = getattr(step, 'observations', '')
        print(f"    Observations: {obs[:100]}..." if len(str(obs)) > 100 else f"    Observations: {obs}")

print()
print("=" * 80)
print("DIAGNOSIS")
print("=" * 80)
print()

# Check for duplicate calls
if weather_tool.call_count > 1:
    locations_called = [call['location'] for call in weather_tool.calls]
    print(f"Locations called: {locations_called}")
    
    # Check if all calls are identical
    if len(set(locations_called)) == 1:
        print("❌ ISSUE FOUND: All tool calls have the SAME arguments!")
        print("   The agent is not using observations from previous calls.")
    else:
        print("✓ CORRECT: Tool calls have DIFFERENT arguments")
        print("  The agent is properly using observations from previous steps.")
else:
    print("Only one tool call was made. Agent may have stopped early.")

print()
