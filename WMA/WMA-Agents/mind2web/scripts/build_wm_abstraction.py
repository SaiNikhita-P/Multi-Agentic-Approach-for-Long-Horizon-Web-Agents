import os
import json
import re
import numpy as np
from bs4 import BeautifulSoup
from scipy.optimize import linear_sum_assignment


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_PATH = os.path.join(BASE_DIR, "data/transitions/wm_raw_transitions.jsonl")
OUTPUT_PATH = os.path.join(BASE_DIR, "data/abstracted/wm_transition_abstraction.jsonl")

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


#HTML Parsing
INTERESTING_TAGS = {
    "button",
    "input",
    "select",
    "option",
    "a",
    "li",
    "span"
}

def parse_elements(html):
    soup = BeautifulSoup(html, "html.parser")
    elements = []

    for tag in soup.find_all(True):

        # Ignore very generic containers
        if tag.name not in INTERESTING_TAGS:
            continue

        text = tag.get_text(strip=True)

        if len(text.strip()) < 3:
           continue


        if not text and tag.name != "input":
            continue

        elements.append({
            "tag": tag.name,
            "text": text,
            "attrs": list(tag.attrs.keys())
        })

    return elements


# Similarity function
def similarity(e1, e2):

    if e1["tag"] != e2["tag"]:
        return 0.0

    set1 = set(e1["text"].lower().split())
    set2 = set(e2["text"].lower().split())

    if not set1 or not set2:
        return 0.0

    text_sim = len(set1 & set2) / len(set1 | set2)

    return text_sim


#Hungarian Matching Algorithm
def match_elements(elems1, elems2):

    if len(elems1) == 0 or len(elems2) == 0:
        return []

    cost = np.zeros((len(elems1), len(elems2)))

    for i in range(len(elems1)):
        for j in range(len(elems2)):
            cost[i][j] = 1 - similarity(elems1[i], elems2[j])

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    for i, j in zip(row_ind, col_ind):
        if similarity(elems1[i], elems2[j]) > 0.75:
            matches.append((i, j))

    return matches

# Transition Classification
def classify_changes(elems1, elems2, matches):

    matched_1 = set(i for i, _ in matches)
    matched_2 = set(j for _, j in matches)

    updated = []
    added = []
    deleted = []

    for i, j in matches:
        if elems1[i]["text"] != elems2[j]["text"]:
            updated.append({
                "tag": elems2[j]["tag"],
                "from": elems1[i]["text"],
                "to": elems2[j]["text"]
            })

    for i, e in enumerate(elems1):
        if i not in matched_1:
            deleted.append(e)

    for j, e in enumerate(elems2):
        if j not in matched_2:
            added.append(e)

    return updated, added, deleted

# Convert to NLP Description
def convert_to_nlp(intent, action, updated, added, deleted,
                   max_added=10, max_deleted=10):

    description = []

    description.append(
        f"[Rationale] The action `{action}` is performed to progress towards the objective: {intent}."
    )

    description.append("[Next State] The expected effect is that ")

    parts = []

    # Cap large diffs
    added = added[:max_added]
    deleted = deleted[:max_deleted]

    # Updated items
    for u in updated:
        parts.append(
            f"the {u['tag']} changes from \"{u['from']}\" to \"{u['to']}\""
        )

    # Added items
    for a in added:
        if a["text"]:
            tag = a["tag"]
            label = a["text"]

            if tag == "a":
                parts.append(f'a link labeled "{label}" appears')
            elif tag == "button":
                parts.append(f'a button labeled "{label}" appears')
            elif tag == "input":
                parts.append(f'an input field labeled "{label}" appears')
            else:
                parts.append(f'a new {tag} with text "{label}" appears')

    # Deleted items
    for d in deleted:
        if d["text"]:
            tag = d["tag"]
            label = d["text"]

            if tag == "a":
                parts.append(f'the link "{label}" disappears')
            elif tag == "button":
                parts.append(f'the button "{label}" disappears')
            else:
                parts.append(f'the {tag} with text "{label}" disappears')

    if not parts:
        parts.append("the page updates without significant visible changes")

    description.append(", ".join(parts) + ".")

    return "\n".join(description)

total_processed = 0

with open(INPUT_PATH, "r") as f_in, open(OUTPUT_PATH, "w") as f_out:

    for line in f_in:

        record = json.loads(line)

        intent = record["intent"]
        o_t = record["o_t"]
        a_t = record["a_t"]
        o_tp1 = record["o_tp1"]

        elems1 = parse_elements(o_t)
        elems2 = parse_elements(o_tp1)

        matches = match_elements(elems1, elems2)

        updated, added, deleted = classify_changes(elems1, elems2, matches)

        abstraction = convert_to_nlp(intent, a_t, updated, added, deleted)

        output_record = {
            "input": f"""Objective: {intent}

Current Observation:
{o_t}

Current Action:
{a_t}

Predict the next state.
""",
            "output": abstraction
        }

        f_out.write(json.dumps(output_record) + "\n")

        total_processed += 1

print("Total abstracted transitions:", total_processed)
print("Saved to:", OUTPUT_PATH)
