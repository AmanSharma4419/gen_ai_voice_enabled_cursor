from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

llm = init_chat_model(model_provider="openai" ,model="gpt-4.1")


class State(TypedDict):
    messages:Annotated[list,add_messages]
    
    
def chatbot(state=State):
    return {"messages": [llm.invoke(state["messages"])]}


graph_builder = StateGraph(State)

graph_builder.add_node("chatbot",chatbot)

graph_builder.add_edge(START,"chatbot")
graph_builder.add_edge("chatbot",END)



def graphstreamwithcheckpointer(checkpointer):
    return graph_builder.compile(checkpointer = checkpointer)