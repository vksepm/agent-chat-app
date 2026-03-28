#!/usr/bin/env python3
"""
Diagnostic script to identify why the agent makes duplicate tool calls.

This script simulates the exact app.py streaming pattern and checks if
observations are properly incorporated into the agent's context.
"""

import os
import sys

# Fix encoding issue on Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

os.environ['MODEL_TYPE'] = 'litellm'
os.environ['MODEL_ID'] = 'gpt-4-mini'
os.environ['MODEL_API_KEY'] = 'sk-test'
os.environ['MCP_SERVER_URL_1'] = 'https://example.com/1'
os.environ['MCP_SERVER_URL_2'] = 'https://example.com/2'
os.environ['LANGFUSE_PUBLIC_KEY'] = 'pk'
os.environ['LANGFUSE_SECRET_KEY'] = 'sk'
os.environ['AGENT_MAX_STEPS'] = '5'

from smolagents import ToolCallingAgent, Tool
from src.agent import build_agent
from src.config import load_config

class WeatherTool(Tool):
    """Simulates get_current_weather MCP tool."""
    name = "get_current_weather"
    description = "Get current weather for a location"
    inputs = {
        "location_name": {"description": "City name", "type": "string"},
        "temperature_unit": {"description": "celsius or fahrenheit", "type": "string", "enum": ["celsius", "fahrenheit"], "nullable": True}
    }
    output_type = "string"
    
    def __init__(self):
        super().__init__()
        self.calls = []
    
    def forward(self, location_name: str, temperature_unit: str | None = None) -> str:
        if temperature_unit is None:
            temperature_unit = "celsius"
        call_info = {
            'call_num': len(self.calls) + 1,
            'location_name': location_name,
            'temperature_unit': temperature_unit,
        }
        self.calls.append(call_info)
        return f"Weather in {location_name}: Sunny, 25°{temperature_unit[0].upper()} (call #{call_info['call_num']})"

# Initialize
config = load_config()
weather_tool = WeatherTool()
agent = build_agent([weather_tool], config)

print("="*80)
print("DIAGNOSTIC: Agent Tool Call Loop Analysis")
print("="*80)
print()

# Simulate the app.py streaming pattern
query = "What's the current temperature in Tokyo?"

print(f"Query: {query}")
print()
print("-"*80)
print("STREAMING EXECUTION (simulating app.py pattern)")
print("-"*80)
print()

from smolagents.agents import ChatMessageStreamDelta, ToolOutput
from smolagents.memory import ActionStep, FinalAnswerStep, PlanningStep

print("Events from agent.run():")
step_count = 0
try:
    for event in agent.run(query, stream=True, reset=False):
        event_type = type(event).__name__
        
        if isinstance(event, ActionStep):
            step_count += 1
            print()
            print(f"  [ActionStep #{step_count}]")
            # Show what tool calls this step made
            for tc in getattr(event, 'tool_calls', []):
                tc_name = getattr(tc, 'name', 'unknown')
                tc_args = getattr(tc, 'arguments', {})
                print(f"    → Tool call: {tc_name}({tc_args})")
            # Show what observations this step received
            obs = getattr(event, 'observations', '')
            if obs:
                print(f"    ← Observation: {obs[:80]}...")
            else:
                print(f"    ← Observation: (none)")
                
        elif isinstance(event, ToolOutput):
            obs_short = getattr(event, 'observation', '')[:80]
            print(f"  [ToolOutput] {obs_short}...")
            
        elif isinstance(event, FinalAnswerStep):
            print(f"  [FinalAnswerStep]")
            final = getattr(event, 'final_answer', '')
            print(f"    Answer: {final[:80]}...")
            
        elif isinstance(event, PlanningStep):
            print(f"  [PlanningStep]")
            
        elif isinstance(event, ChatMessageStreamDelta):
            pass  # Skip these for clarity
            
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print()
print("-"*80)
print("DIAGNOSIS")
print("-"*80)
print()

print(f"Tool calls made: {len(weather_tool.calls)}")
for call in weather_tool.calls:
    print(f"  Call #{call['call_num']}: {call['location_name']} ({call['temperature_unit']})")

print()
print("Agent memory analysis:")
print(f"  Total steps: {len(agent.memory.steps)}")

seen_observations = []
for i, step in enumerate(agent.memory.steps):
    if isinstance(step, ActionStep):
        obs = getattr(step, 'observations', '')
        if obs:
            seen_observations.append(obs)
            print(f"  Step {i}: Has observations ✓")
        else:
            print(f"  Step {i}: NO observations ✗")

print()
if len(weather_tool.calls) > 1:
    # Check if all calls are identical
    all_same = all(
        call['location_name'] == weather_tool.calls[0]['location_name'] and
        call['temperature_unit'] == weather_tool.calls[0]['temperature_unit']
        for call in weather_tool.calls
    )
    
    if all_same:
        print("❌ PROBLEM CONFIRMED:")
        print("   The agent called the same tool with identical arguments multiple times.")
        print("   This indicates the agent is NOT using observations from previous calls.")
        print()
        print("POSSIBLE CAUSES:")
        print("  1. Observations are not being populated in ActionStep")
        print("  2. Agent's context doesn't include previous observations when streaming")
        print("  3. reset=False is not properly preserving state during streaming")
        print("  4. The streaming generator is interrupting the agent's reasoning loop")
    else:
        print("✓ WORKING CORRECTLY:")
        print("   The agent made different tool calls based on previous responses.")
else:
    print("⚠ INCONCLUSIVE:")
    print("   Only one tool call was made. Agent may have stopped early or finished correctly.")

print()
print("="*80)
