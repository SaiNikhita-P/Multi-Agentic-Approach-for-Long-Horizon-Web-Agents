import os
import json
import pickle
from pathlib import Path

# %% load data
def load_json(data_dir, folder_name):
    folder_path = os.path.join(data_dir, folder_name)
    print(f"Data path: {folder_path}")
    data_paths = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".json")
    ]
    data_paths = sorted(data_paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))

    # Construct trajectory dataset
    samples = []
    for data_path in data_paths:
        with open(data_path, "r") as f:
            samples.extend(json.load(f))
    print("# of samples:", len(samples))

    return samples


def add_scores(
    examples: list[dict],
    candidate_results: dict = None,
    score_path: str = "mind2web/data/scores_all_data.pkl"
):

    if candidate_results is None:
        with open(score_path, "rb") as f:
            candidate_results = pickle.load(f)

    for sample in examples:
        for s, act_repr in zip(sample["actions"], sample["action_reprs"]):

            sample_id = f"{sample['annotation_id']}_{s['action_uid']}"
            key = f"ranks_{sample_id}"

            if key not in candidate_results:
                for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                    for i, candidate in enumerate(candidates):
                        candidate["score"] = 0.0
                        candidate["rank"] = i
                continue

            scores = candidate_results[key]["scores"]
            ranks = candidate_results[key]["ranks"]

            for candidates in [s["pos_candidates"], s["neg_candidates"]]:
                for candidate in candidates:
                    candidate_id = int(candidate["backend_node_id"])

                    if candidate_id < len(scores):
                        candidate["score"] = scores[candidate_id]
                        candidate["rank"] = ranks[candidate_id]
                    else:
                        candidate["score"] = 0.0
                        candidate["rank"] = 0

    return examples


# %% workflow induction
def format_examples(examples: list[dict], prefix: str = None, suffix: str = None) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(f"Query #{i+1}: {ex['confirmed_task']}")
        lines.append("Actions and Environments:")
        lines.extend(ex["action_reprs"])
        lines.append("")
    prompt = '\n'.join(lines)
    if prefix is not None:
        prompt = prefix + '\n' + prompt
    if suffix is not None:
        prompt += '\n\n' + suffix
    return prompt



# %% model generation
def is_website_header(block: str, website: str) -> bool:
    lines = block.strip().split('\n')
    if len(lines) > 1: return False
    text = lines[0].strip()
    if text.startswith("#") and text.lower().endswith(website):
        return True
    return False

def filter_workflows(text: str, website: str) -> str:
    blocks = text.split('\n\n')
    for i,b in enumerate(blocks):
        if is_website_header(b, website):
            blocks = blocks[i+1: ]
            break

    for i,b in enumerate(blocks):
        if is_website_header(b, "delta"):
            blocks = blocks[: i]
            break

    blocks = [b for b in blocks if "delta" not in b.lower()]
    return '\n\n'.join(blocks)

def format_examples_abs(examples: list[dict], prefix: str = None, suffix: str = None) -> str:
    lines = []
    for i, ex in enumerate(examples):
        lines.append(f"## Query {i+1}: {ex['confirmed_task']}")
        lines.append("    Actions:")
        lines.extend(ex["action_reprs"])
        lines.append("")
    prompt = '\n'.join(lines)
    if prefix is not None:
        prompt = prefix + '\n' + prompt
    if suffix is not None:
        prompt += '\n\n' + suffix
    return prompt 

def set_paths(args):
    """Set paths."""
    flag_suffix = f"_{args.flag}" if args.flag is not None else ""
    # extracted test data (of the same website) path
    args.tmp_examples_path = os.path.join(args.upper_results_dir, "extracted_data", f"{args.benchmark}", "{}_tmp_examples.json".format(args.website))
    os.makedirs(os.path.dirname(args.tmp_examples_path), exist_ok=True)
    
    # workflow path
    args.workflow_path = Path(f"{args.abstraction_dir}/{args.benchmark}/{args.flag}/{args.website}{flag_suffix}.txt")
    args.semantic_workflow_path = Path(f"{args.abstraction_dir}/{args.benchmark}/{args.flag}/{args.website}{flag_suffix}_semantic.txt")
    os.makedirs(os.path.dirname(args.workflow_path), exist_ok=True)

    #domain_level_workflows
    args.domain_workflow_path = Path(
        f"{args.abstraction_dir}/{args.benchmark}/domains/{args.domain}{flag_suffix}.txt"
    )
    args.domain_semantic_workflow_path = Path(
        f"{args.abstraction_dir}/{args.benchmark}/domains/{args.domain}{flag_suffix}_semantic.txt"
    )
    os.makedirs(os.path.dirname(args.domain_workflow_path), exist_ok=True)

    # existed abstraction to log history
    args.existed_workflow_path = Path(f"{args.abstraction_dir}/{args.benchmark}/log/{args.website}{flag_suffix}_all.json")
    os.makedirs(os.path.dirname(args.existed_workflow_path), exist_ok=True)

    # prediction results directory (for specific website)
    args.results_dir = Path(f"{args.upper_results_dir}/{args.benchmark}{flag_suffix}/{args.website}")
    os.makedirs(args.results_dir, exist_ok=True)

    # log path (uncomment and check maybe!)
    # args.log_path = Path(f"logs/{args.benchmark}/log_{args.flag}.txt")
    # os.makedirs(os.path.dirname(args.log_path), exist_ok=True)
    
    return args

def map_domains(args):
    maps = json.load(open(args.map_path, "r"))
    if args.website not in maps:
        raise ValueError(f"Website '{args.website}' not found in domain map")
    args.domain = maps[args.website]["domain"]
    args.subdomain = maps[args.website]["subdomain"]
    return args