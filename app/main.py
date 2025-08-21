from graph import graphstreamwithcheckpointer
import speech_recognition as sr
from langchain_core.messages import HumanMessage
from pymongo import MongoClient
from langgraph.checkpoint.mongodb import MongoDBSaver

# --- MongoDB Setup ---
client = MongoClient("mongodb://root:root@localhost:27017/")
checkpointer = MongoDBSaver(
    client,
    db_name="voice-enabled-cursor",
    collection_name="voice-enabled-cursor-cl"
)

# Build graph with checkpointer
graphwithmongo = graphstreamwithcheckpointer(checkpointer=checkpointer)

# Config identifies conversation thread (required for checkpoints)
config = {"configurable": {"thread_id": "chat-session-1"}}

recognizer = sr.Recognizer()
recognizer.pause_threshold = 2

def voice_chatbot():
    while True:
        with sr.Microphone() as source:
            print("🎤 Say something (say 'exit' to quit)...")
            audio = recognizer.listen(source)
            print("processing:")

        try:
            text = recognizer.recognize_google(audio)
            print("You said:", text)

            # Exit condition
            if text.lower() in ["exit", "quit", "stop"]:
                print("👋 Exiting chatbot...")
                break

            for event in graphwithmongo.stream(
                {"messages": [HumanMessage(content=text)]},
                stream_mode="values",
                config=config
            ):
                if "messages" in event:
                    event["messages"][-1].pretty_print()

        except sr.UnknownValueError:
            print(" Sorry, I could not understand the audio.")
        except sr.RequestError:
            print(" Could not request results, check your internet connection.")

voice_chatbot()
