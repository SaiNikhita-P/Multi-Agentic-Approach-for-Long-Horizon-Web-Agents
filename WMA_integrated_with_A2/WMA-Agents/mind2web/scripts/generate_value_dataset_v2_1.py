import json
import os
import re
from openai import OpenAI
from mind2web.utils.llm import extract_from_response

# =========================
# Config
# =========================

# INPUT_PATH = "../data/transitions/wm_raw_transitions.jsonl"
# OUTPUT_PATH = "../data/value/value_dataset_branching.jsonl"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data", "transitions", "wm_raw_transitions.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "value", "value_dataset_v2_1.jsonl")

TOP_K = 3

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

# =========================
# Helper
# =========================

# def normalize_action(action):
#     return action.strip().lower()



def normalize_action(action: str) -> str:
    """
    Normalize action string into canonical form:
    - Extract element ID if present
    - Lowercase
    - Strip whitespace
    - Remove wrappers like <select>, id=, etc.
    """

    if action is None:
        return ""

    action = action.strip().lower()

    # 1️⃣ Extract numeric id (most reliable signal)
    id_match = re.search(r"\b\d+\b", action)
    if id_match:
        return id_match.group(0)

    # 2️⃣ Extract id=XXX format
    id_match = re.search(r"id\s*=\s*(\d+)", action)
    if id_match:
        return id_match.group(1)

    # 3️⃣ Extract inside SELECT [136]
    bracket_match = re.search(r"\[(\d+)\]", action)
    if bracket_match:
        return bracket_match.group(1)

    # 4️⃣ Fallback: remove special chars and spaces
    action = re.sub(r"[^\w]", "", action)
    return action

# =========================
# Load existing dataset
# =========================

existing = set()
existing_counts = {}

if os.path.exists(OUTPUT_PATH):
    print("Loading existing dataset...")

    with open(OUTPUT_PATH) as f:
        for line in f:
            ex = json.loads(line)

            step = ex["step"]
            intent = ex["intent"]

            action = ex["input"].split("Current Action:")[1].split("Predict")[0].strip()
            norm_action = normalize_action(action)

            key = (intent, step, norm_action)
            existing.add(key)

            step_key = (intent, step)
            existing_counts[step_key] = existing_counts.get(step_key, 0) + 1

print("Existing samples:", len(existing))

# =========================
# Build Dataset
# =========================

total = 0

with open(INPUT_PATH, "r") as f_in, open(OUTPUT_PATH, "a") as f_out:

    transitions = [json.loads(line) for line in f_in]

    # Group by trajectory (intent)
    trajectories = {}
    for t in transitions:
        trajectories.setdefault(t["intent"], []).append(t)

    for intent, traj in trajectories.items():

        T = len(traj)

        for step_idx, step in enumerate(traj):
            step_key = (intent, step_idx)

            if existing_counts.get(step_key, 0) >= TOP_K:
                continue

            o_t = step["o_t"]
            a_gt = step["a_t"]

            # -------------------------
            # 1. Query policy model
            # -------------------------

            policy_prompt = f"""
Objective: {intent}

Current Observation:
{o_t}

Suggest next action.
"""

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a web navigation agent."},
                    {"role": "user", "content": policy_prompt}
                ],
                temperature=0.7,
                n=5
            )

            # Extract unique actions
            candidate_actions = []
            for choice in response.choices:
                parsed = extract_from_response(choice.message.content, "`")
                if parsed and parsed not in candidate_actions:
                    candidate_actions.append(parsed)

            candidate_actions = candidate_actions[:TOP_K]

            # -------------------------
            # 2. Assign rewards
            # -------------------------

            for action in candidate_actions:
                key = (intent, step_idx, normalize_action(action))

                if key in existing:
                    continue

                if normalize_action(action) == normalize_action(a_gt):
                    reward = (step_idx + 1) / T
                else:
                    reward = 0.0

                text_input = f"""
Objective: {intent}

Current Observation:
{o_t}

Current Action:
{action}

Predict the reward.
"""

                example = {
                    "intent": intent,
                    "step": step_idx,
                    "input": text_input.strip(),
                    "score": reward
                }

                f_out.write(json.dumps(example) + "\n")
                total += 1

                existing.add(key)
                existing_counts[step_key] = existing_counts.get(step_key, 0) + 1

print("✅ Branching value dataset created.")
print("Total examples:", total)
print("Saved to:", OUTPUT_PATH)
