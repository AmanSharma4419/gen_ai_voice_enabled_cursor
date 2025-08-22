from typing import Annotated
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode



load_dotenv()



@tool
def run_command(cmd: str):
    """Execute a command on the system and return its output."""
    try:
        return os.popen(cmd).read()
    except Exception as e:
        return f"Error running command: {e}"



llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools([run_command])



class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [run_command]
tools_by_name = {tool.name: tool for tool in tools}



def chatbot(state: State):
    """LLM node that can decide to call tools."""
    result = llm_with_tools.invoke(state["messages"])
    
    # Return just the AI message - tool execution will be handled by ToolNode
    return {"messages": [result]}


def tool_node(state: State):
    """Custom tool node to handle tool execution properly."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_responses = []
        for tool_call in last_message.tool_calls:
            # Get the tool function
            tool_func = tools_by_name.get(tool_call["name"])
            
            if tool_func:
                try:
                    # Execute the tool
                    result = tool_func.invoke(tool_call["args"])
                    # Create proper ToolMessage response
                    tool_responses.append(
                        ToolMessage(
                            content=str(result),
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"]
                        )
                    )
                except Exception as e:
                    tool_responses.append(
                        ToolMessage(
                            content=f"Error executing tool: {e}",
                            tool_call_id=tool_call["id"],
                            name=tool_call["name"]
                        )
                    )
        
        return {"messages": tool_responses}
    
    return {"messages": []}



graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")


def route_from_chatbot(state: State):
    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"
    return END


graph_builder.add_conditional_edges("chatbot", route_from_chatbot)

graph_builder.add_edge("tools", "chatbot")


def graphstreamwithcheckpointer(checkpointer=None):
    return graph_builder.compile(checkpointer=checkpointer)