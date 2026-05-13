import os
import argparse
from tqdm import tqdm
import time
from mind2web.memory import eval_sample
from mind2web.utils.data import load_json, add_scores, set_paths, map_domains
from mind2web.custom.agent import (
    AWMWMAgent
)
import pickle
import logging
logger = logging.getLogger("atm")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(handler)
from mind2web.custom.agent import (
    AWMWMAgent
)

from mind2web.utils.llm import Caller


from mind2web.abs_induction import abstraction_induction


from multiprocessing import Process


import glob
from datetime import datetime

# has_recent_result function is added to avoid any unnecessary runs after sudden server interruptions or crashes.

def has_recent_result(args, website, task_id, threshold="20260424_102400"):
    """
    Check if a recent result file exists for this task.
    """

    workflow_dir = os.path.join(
        args.output_dir,
        args.model,
        args.benchmark,
        website,
        "workflow"
    )

    pattern = os.path.join(workflow_dir, f"{task_id}_*.json")
    files = glob.glob(pattern)

    if not files:
        return False

    for f in files:
        try:
            fname = os.path.basename(f)
            timestamp = fname.split("_")[1] + "_" + fname.split("_")[2].split(".")[0]

            if timestamp >= threshold:
                return True
        except:
            continue

    return False

def run(args: argparse.Namespace, examples_: list[dict],agent) -> None:
    caller = Caller(args.model_abs)
    examples = [s for s in examples_ if s["website"] == args.website]
    print(f"Filtering down to #{len(examples)} examples on website [{args.website}]")

    with open("mind2web/data/scores_all_data.pkl", "rb") as f:
      candidate_results = pickle.load(f)

    examples = add_scores(examples, candidate_results)
    # examples = add_scores(examples) # add prediction scores and ranks to elements

    # agent = AWMWMAgent(
    #     action_prediction_prompt_path="agent/prompts/jsons/p_cot_id_actree_2s_no_na.json",
    #     state_prediction_prompt_path="mind2web/wm_html.json",
    #     value_function_prompt_path="mind2web/vf_html.json",
    #     model_name=args.model,
    #     branching_factor=args.branching_factor,
    #     action_set_tag="playwright",
    #     vf_budget=args.vf_budget,
    #     world_model_training=args.world_model_training,
    #     world_model_name=args.world_model_name,
    #     world_model_url=args.world_model_url,
    #     value_model_training=args.value_model_training,
    #     value_model_name=args.value_model_name,
    #     value_model_url=args.value_model_url,
    # )

    # if args.end_idx is None:
    args.end_idx = len(examples)

    # args.end_idx = len(examples) if args.end_idx is None else min(args.end_idx, len(examples))    
    for i in tqdm(range(args.start_idx, args.end_idx)):

        args.domain = examples[i]["domain"]
        args.subdomain = examples[i]["subdomain"]

        # Skip if recent result exists
        has_recent = has_recent_result(args, args.website, i)
        if has_recent:
            print(f"Skipping {args.website} task {i} (already has recent result)")
            continue

        # # Otherwise run normally
        # if args.mode == "memory":
        #     eval_sample(i, args, examples[i], agent)

        if args.mode == "memory":
            eval_sample(i, args, examples[i],agent)
        elif args.mode == "action":
            raise NotImplementedError
        else:
            raise ValueError(f"Unsupported workflow format: {args.workflow_format}")
        
        # adding abstractions combining abstractions from the A2 pipeline
       
        if args.if_workflow :
            time.sleep(0.2)
            try:
                abstraction_induction(args, examples, caller, i)
                print(f"  Updated abstractions using sample {i}")
            except Exception as e:
                print(f"  Abstraction failed at {i}: {e}")


def main(args):

    agent = AWMWMAgent(
    action_prediction_prompt_path="agent/prompts/jsons/p_cot_id_actree_2s_no_na.json",
    state_prediction_prompt_path="mind2web/wm_html.json",
    value_function_prompt_path="mind2web/vf_html.json",
    model_name=args.model,
    branching_factor=args.branching_factor,
    action_set_tag="playwright",
    vf_budget=args.vf_budget,
    world_model_training=args.world_model_training,
    world_model_name=args.world_model_name,
    world_model_url=args.world_model_url,
    value_model_training=args.value_model_training,
    value_model_name=args.value_model_name,
    value_model_url=args.value_model_url,
)
    for b in ["test_task", "test_website", "test_domain"]:
    # for b in ["test_domain"]:    
        args.benchmark = b

        examples_ = load_json(args.data_dir, args.benchmark)
        websites = [*set([s["website"] for s in examples_])]

        ps = []

        # for w in tqdm(websites):
        #     args.website = w
        #     args.workflow_path = f"mind2web/workflow/{w}.txt"

        #     p = Process(target=lambda: run(args, examples_))
        #     p.start()
        #     ps.append(p)

        # for p in ps:
        #     p.join()

        #---- commented recently!!-----
        # for w in tqdm(websites):
        #     args.website = w
        #     args.workflow_path = f"mind2web/workflow/{w}.txt"

        #     run(args, examples_,agent)

        for w in tqdm(websites):
            args.website = w

            # Set domain + subdomain
            args = map_domains(args)

            # Set all required paths
            args = set_paths(args)

            run(args, examples_, agent)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="mind2web/data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--memory_path", type=str, default="mind2web/data/memory")
    parser.add_argument("--log_dir", type=str, default="mind2web/results")
    parser.add_argument(
    "--output_dir",
    type=str,
    default="mind2web/results",
    help="Base directory for saving outputs"
)

    # model
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, default=1.0)

    # env context
    parser.add_argument("--previous_top_k_elements", type=int, default=3)
    parser.add_argument("--top_k_elements", type=int, default=5)
    parser.add_argument("--retrieve_top_k", type=int, default=1)

    # workflow
    parser.add_argument("--website", type=str, default="")
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--subdomain", type=str, default=None)
    parser.add_argument("--workflow_path", type=str, default="mind2web/workflow/asdf")
    parser.add_argument("--suffix", type=str, default="workflow")

    # prompts
    parser.add_argument("--abs_sys_prompt_path", type=str, default="mind2web/prompt/instruction_abs_basic.txt",
                       help="Path to system prompt for generating semantic abstractions")
    parser.add_argument("--abs_sys_prompt_domain_path", type=str, default="mind2web/prompt/instruction_abs_domain_basic.txt",
                       help="Path to system prompt for generating semantic abstractions")
    parser.add_argument("--abs_exemplar_prompt_path", type=str, default="mind2web/prompt/one_shot_abs_basic.txt",
                       help="Path to exemplar prompt for semantic abstractions")
    parser.add_argument("--abs_exemplar_prompt_domain_path", type=str, default="mind2web/prompt/one_shot_abs_basic_domain.txt",
                       help="Path to exemplar prompt for semantic abstractions")
    
    parser.add_argument("--episodic_abs_sys_prompt_path", type=str, default="mind2web/prompt/instruction_abs_episodic.txt",
                       help="Path to system prompt for generating episodic abstractions")
    parser.add_argument("--episodic_abs_sys_prompt_domain_path", type=str, default="mind2web/prompt/instruction_abs_domain_episodic.txt",
                       help="Path to system prompt for generating episodic abstractions")
    
    parser.add_argument("--episodic_abs_exemplar_prompt_path", type=str, default="mind2web/prompt/one_shot_abs_episodic.txt",
                       help="Path to exemplar prompt for episodic abstractions")
    parser.add_argument("--episodic_abs_exemplar_prompt_domain_path", type=str, default="mind2web/prompt/one_shot_abs_episodic_domain.txt",
                       help="Path to exemplar prompt for episodic abstractions")
    
    #abstractions
    # parser.add_argument("--abstraction_dir", type=str, default="abstraction",
    #                    help="Directory for storing generated abstractions")
    parser.add_argument("--if_workflow", type=bool, default=True)
    parser.add_argument(
    "--flag", type=str, default="abs_task",)
    
    parser.add_argument("--abstraction_dir", default="abstraction")
    parser.add_argument("--upper_results_dir", default="results")
    parser.add_argument("--map_path", default="mind2web/data/website_domain_pairs.json")
    # ablation
    parser.add_argument("--mode", type=str, default="memory", choices=["memory", "action"])
    parser.add_argument("--start_idx", type=int, default=0, help="Select example index.")
    parser.add_argument("--end_idx", type=int, default=None, help="Select example index.")

    # world model & value function
    parser.add_argument("--branching_factor", type=int, default=3)
    parser.add_argument("--vf_budget", type=int, default=20)
    parser.add_argument("--world_model_training", action="store_true")
    parser.add_argument("--world_model_name", type=str, default=None)
    parser.add_argument("--world_model_url", type=str, default=None)
    parser.add_argument("--value_model_training", action="store_true")
    parser.add_argument("--value_model_name", type=str, default=None)
    parser.add_argument("--value_model_url", type=str, default=None)

    #model for abstractions

    parser.add_argument("--model_abs", type=str, default="gpt-4o-mini",
                       help="LLM model to use")

    args = parser.parse_args()

    # sanity check
    if not os.path.exists(args.workflow_path): open(args.workflow_path, 'w').close()
    if args.retrieve_top_k != 1: print(f"Suggest set `retrieve_top_k` to 1, currently as {args.retrieve_top_k}")

    if args.world_model_training:
        assert args.world_model_name is not None and args.world_model_url is not None
    if args.value_model_training:
        assert args.value_model_name is not None and args.value_model_url is not None

    main(args)
