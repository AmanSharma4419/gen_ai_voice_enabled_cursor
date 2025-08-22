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
config = {"configurable": {"thread_id": "chat-session-01"}}

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

            # Process the message through the graph
            final_state = None
            for event in graphwithmongo.stream(
                {"messages": [HumanMessage(content=text)]},
                stream_mode="values",
                config=config
            ):
                final_state = event
                if "messages" in event and event["messages"]:
                    # Print only the latest message to avoid duplication
                    if not isinstance(event["messages"][-1], HumanMessage):
                        event["messages"][-1].pretty_print()

        except sr.UnknownValueError:
            print("❌ Sorry, I could not understand the audio.")
        except sr.RequestError:
            print("⚠️ Could not request results, check your internet connection.")
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    voice_chatbot()