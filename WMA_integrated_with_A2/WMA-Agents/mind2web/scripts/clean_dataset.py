import json
from tqdm import tqdm

INPUT_PATH = "../data/value/value_dataset_v2_1_with_pred_obs_1.jsonl"
OUTPUT_PATH = "../data/value/value_dataset_clean.jsonl"


def clean_prediction(text):

    text = text.strip()

    # remove "assistant"
    if text.lower().startswith("assistant"):
        text = text.split("\n", 1)[-1]

    # remove rationale
    if "[Next Observation]" in text:
        text = text.split("[Next Observation]")[-1]

    if "[Rationale]" in text and "[Next Observation]" not in text:
        text = text.split("[Rationale]")[-1]

    return text.strip()


print("Cleaning dataset...")

count = 0

with open(INPUT_PATH) as f_in, open(OUTPUT_PATH, "w") as f_out:

    for line in tqdm(f_in):

        data = json.loads(line)

        data["predicted_observation"] = clean_prediction(
            data["predicted_observation"]
        )

        f_out.write(json.dumps(data) + "\n")

        count += 1

print("\nDataset cleaned.")
print("Total samples:", count)
print("Saved to:", OUTPUT_PATH)