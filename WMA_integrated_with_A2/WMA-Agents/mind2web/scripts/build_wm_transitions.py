import json
import re
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data/memory/exemplars.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/transitions/wm_raw_transitions.jsonl")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


def extract_observation(text):
    match = re.search(r"Observation:\s*`(.*)`", text, re.DOTALL)
    return match.group(1).strip() if match else None

def extract_action(text):
    match = re.search(r"Action:\s*`(.*)`", text)
    return match.group(1).strip() if match else None

with open(INPUT_PATH, "r") as f:
    data = json.load(f)

total_transitions = 0

with open(OUTPUT_PATH, "w") as out:
    for traj in data:

        # Extract intent from first user message
        first_user = traj[0]
        intent_match = re.search(r"Task:\s*(.*)", first_user["content"])
        intent = intent_match.group(1).strip() if intent_match else ""

        for i in range(len(traj) - 2):

            if (
                traj[i]["role"] == "user" and
                traj[i+1]["role"] == "assistant" and
                traj[i+2]["role"] == "user"
            ):
                o_t = extract_observation(traj[i]["content"])
                a_t = extract_action(traj[i+1]["content"])
                o_tp1 = extract_observation(traj[i+2]["content"])

                if o_t and a_t and o_tp1:
                    record = {
                        "intent": intent,
                        "o_t": o_t,
                        "a_t": a_t,
                        "o_tp1": o_tp1
                    }
                    out.write(json.dumps(record) + "\n")
                    total_transitions += 1

print("Total transitions extracted:", total_transitions)
