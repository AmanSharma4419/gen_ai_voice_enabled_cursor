from typing import Annotated
import requests
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

@tool
def get_weather(city: str):
    """Fetch real weather forecast using OpenWeatherMap API."""
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        response.raise_for_status()   # raises HTTPError if bad request
        data = response.json()
        print(data,"data")
        temp = data["main"]["temp"]
        condition = data["weather"][0]["description"]
        return f"Weather in {city}: {condition}, {temp}°C"

    except requests.exceptions.HTTPError as e:
        return f"HTTP error: {e}"
    except requests.exceptions.RequestException as e:
        return f"Network error: {e}"
    except KeyError:
        return "Unexpected API response format."
    except Exception as e:
        return f"Unknown error: {e}"


llm = init_chat_model(model_provider="openai", model="gpt-4.1")
llm_with_tools = llm.bind_tools([run_command,get_weather])



class State(TypedDict):
    messages: Annotated[list, add_messages]


tools = [run_command,get_weather]
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