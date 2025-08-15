from typing import Annotated, Literal, TypedDict
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from enum import Enum


# Define the state
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# Define the multiply tool
@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b


# Initialize the model
model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0)
model_with_tools = model.bind_tools([multiply])


def call_model(state: State) -> dict:
    """Call the model node."""
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

class InterruptType(str, Enum):
    SOMETHING = "foo"

class Interrupt(TypedDict):
    type: InterruptType
    

def tool_node(state: State) -> dict:
    """Tool node that throws interrupt on first line."""
    interrupt(Interrupt(type=InterruptType.SOMETHING))
    
    # This code never executes due to the interrupt above
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        tool_call = last_message.tool_calls[0]
        args = tool_call.get('args', {})
        result = multiply.invoke(args)
        
        tool_message = ToolMessage(
            content=str(result),
            tool_call_id=tool_call['id']
        )
        return {"messages": [tool_message]}
    
    return {"messages": []}


def should_continue(state: State) -> Literal["tool", "end"]:
    """Conditional edge from call_model."""
    messages = state["messages"]
    last_message = messages[-1]
    
    if isinstance(last_message, AIMessage) and hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool"
    return "end"


# Build the graph
workflow = StateGraph(State)

# Add the 2 nodes
workflow.add_node("call_model", call_model)
workflow.add_node("tool", tool_node)

# Add edges
workflow.add_edge(START, "call_model")
workflow.add_conditional_edges("call_model", should_continue, {"tool": "tool", "end": END})
workflow.add_edge("tool", "call_model")

graph = workflow.compile()
