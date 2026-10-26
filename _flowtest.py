"""End-to-end test of the guided support flow via HTTP."""
import requests, json

B = "http://127.0.0.1:5001"
s = requests.Session()
s.post(f"{B}/login", data={"username": "customer", "password": "customer123"})

def chat(payload):
    r = s.post(f"{B}/api/chat", json=payload, timeout=60)
    return r.json()["bot"]

print("=== Turn 1: a question with NO direct KB match (natural phrasing) ===")
b1 = chat({"query": "My parcel is stuck somewhere and I am worried",
           "category": "Order Related", "stage": "ask"})
print(" stage:", b1["stage"], "| source:", b1["source"])
print(" reply:", b1["reply"][:100])
print(" clarify_step:", b1.get("clarify_step"))

print("\n=== Turn 2: answer first clarifying question ===")
b2 = chat({"query": "Order ID is ORD12345", "category": "Order Related",
           "stage": "clarify", "clarify_step": b1["clarify_step"],
           "original_query": b1["original_query"], "collected": b1["collected"]})
print(" stage:", b2["stage"], "| source:", b2["source"])
print(" reply:", b2["reply"][:100])
print(" clarify_step:", b2.get("clarify_step"))

print("\n=== Turn 3: answer second clarifying question -> should EXHAUST + offer agent ===")
b3 = chat({"query": "It has not shipped yet", "category": "Order Related",
           "stage": "clarify", "clarify_step": b2["clarify_step"],
           "original_query": b2["original_query"], "collected": b2["collected"]})
print(" stage:", b3["stage"], "| source:", b3["source"])
print(" offer_agent:", b3.get("offer_agent"))
print(" reply:", b3["reply"][:140])

print("\n=== Verify a matching query returns verified answer + follow_up ===")
b4 = chat({"query": "reverse pickup enquiry", "category": "Returns", "stage": "ask"})
print(" stage:", b4["stage"], "| source:", b4["source"])
print(" follow_up:", b4.get("follow_up"))
print(" reply:", b4["reply"][:100])
