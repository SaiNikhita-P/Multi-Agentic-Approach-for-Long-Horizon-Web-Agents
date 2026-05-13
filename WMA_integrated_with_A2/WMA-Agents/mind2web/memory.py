import os, json, random
import numpy as np
from pathlib import Path
from openai import BadRequestError
from datetime import datetime, timedelta
from mind2web.utils.env import (
    get_target_obs_and_act,
    get_top_k_obs,
    parse_act_str,
    calculate_f1,
    construct_act_str
)
from mind2web.utils.llm import (
    num_tokens_from_messages,
    MAX_TOKENS, extract_from_response,
)
from mind2web.custom.llm import (
    generate_response
)
from mind2web.custom.agent import (
    AWMWMAgent
)

import logging
logger = logging.getLogger(__name__)

def get_ist_timestamp():
    ist_time = datetime.utcnow() + timedelta(hours=5, minutes=30)
    return ist_time.strftime("%Y%m%d_%H%M%S")

  

#----------------------------------- actual code for get_exemplars commented to add A2 on 25/03/26----------------------------------

# def get_exemplars(args) -> list:
#     """Get exemplar workflows in the prompt."""
#     # workflow memory
#     memory = []
#     workflow_text = open(args.workflow_path, 'r').read().strip()
#     if len(workflow_text):
#         memory = [[{"role": "user", "content": workflow_text}]]

#     # concrete examples
#     with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
#         concrete_examples = json.load(f)
#     if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
#         concrete_examples = [
#             cex for cex in concrete_examples
#             if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain, args.website]])
#         ]
#     elif any([args.subdomain in cex[0].get("specifier", "") for cex in concrete_examples]):
#         concrete_examples = [
#             cex for cex in concrete_examples
#             if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain]])
#         ]

#     memory += random.sample(concrete_examples,
#         min(args.retrieve_top_k, len(concrete_examples)))
#     memory = [[{k:v for k,v in m.items() if k!="specifier"} for m in e] for e in memory]
#     return memory

#--------- get_exemplars with A2 added -------------

#----------problems: no grounding! and hence the results were bad!!-----------

# def get_exemplars(args) -> list:
#     """Get exemplar workflows in the prompt."""
#     # (a) load workflow memory
#     memory = []
#     workflow_text = open(args.workflow_path, 'r', encoding='utf-8').read().strip()
#     if len(workflow_text):
#         if "abs" in args.flag:
#             memory = [[
#                 {"role": "user",
#                 #   "content": f"Existed Workflow: {workflow_text}"}
#                 "content": f"[Website-level Workflow]\n{workflow_text}"}
#             ]]
#         else:
#             raise ValueError("invalid flag!")
        
#     #domain_level_workflow!    
#     if args.domain_semantic_workflow_path.exists():
#         domain_workflow_text = open(
#             args.domain_semantic_workflow_path, 'r', encoding='utf-8'
#         ).read().strip()
#         if domain_workflow_text:
#             memory.append([
#                 {
#                     "role": "user",
#                     "content": f"[Domain-level Workflow]\n{domain_workflow_text}"
#                 }
#             ])    
#     # (b) load concrete examples
#     # with open(os.path.join(args.memory_path, "exemplars.json"), "r", encoding="utf-8") as f:
#     #     concrete_examples = json.load(f)
#     # if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
#     #     concrete_examples = [
#     #         cex for cex in concrete_examples 
#     #         if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain, args.website]])
#     #     ]
#     # elif any([args.subdomain in cex[0].get("specifier", "") for cex in concrete_examples]):
#     #     concrete_examples = [
#     #         cex for cex in concrete_examples 
#     #         if all([tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain]])
#     #     ]

#     # memory += random.sample(concrete_examples, 
#     #     min(args.retrieve_top_k, len(concrete_examples)))

#     # (b) retrieve exemplars from vector DB
#     # (b) retrieve exemplars from vector DB
#     # (b) retrieve exemplars from vector DB
#     # if args.retrieve_top_k > 0:
#     #     cfg = getattr(args, "config", {}) or {}

#     #     if cfg.get("exemplars", {}).get("backend") == "qdrant":
#     #         try:
#     #             retrieved = retrieve_exemplars(
#     #                 query_text=task,
#     #                 top_k=min(
#     #                     args.retrieve_top_k,
#     #                     cfg.get("exemplars", {}).get("top_k", args.retrieve_top_k)
#     #                 ),
#     #                 domain=args.domain,
#     #                 subdomain=args.subdomain,
#     #                 website=args.website,
#     #                 cfg=cfg,
#     #             )
#     #             memory.extend(retrieved)
#     #         except Exception as e:
#     #             logger.warning(
#     #                 f"[VECTOR-DB] Retrieval failed, skipping exemplars: {e}"
#     #             )
#     #     else:
#     #         logger.info("[VECTOR-DB] Retrieval disabled by config")


#     memory = [[{k:v for k,v in m.items() if k!="specifier"} for m in e] for e in memory]
#     return memory

def get_exemplars(args) -> list:
    """Get exemplar workflows in the prompt."""

    import os, json, random


    memory = []
    # Adding website level abstractions
    workflow_text = open(args.workflow_path, 'r', encoding='utf-8').read().strip()
    if len(workflow_text):
        if "abs" in args.flag:
            memory = [[
                {
                    "role": "user",
                    "content": f"[Website-level Workflow]\n{workflow_text}"
                }
            ]]
        else:
            raise ValueError("invalid flag!")

   # Adding domain level abstractions
    if args.domain_semantic_workflow_path.exists():
        domain_workflow_text = open(
            args.domain_semantic_workflow_path, 'r', encoding='utf-8'
        ).read().strip()

        if domain_workflow_text:
            memory.append([
                {
                    "role": "user",
                    "content": f"[Domain-level Workflow]\n{domain_workflow_text}"
                }
            ])

   # loading relevant examples
    with open(os.path.join(args.memory_path, "exemplars.json"), "r") as f:
        concrete_examples = json.load(f)

    # Filter examples
    if any([args.website in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples
            if all(tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain, args.website])
        ]
    elif any([args.subdomain in cex[0].get("specifier", "") for cex in concrete_examples]):
        concrete_examples = [
            cex for cex in concrete_examples
            if all(tag in cex[0]["specifier"] for tag in [args.domain, args.subdomain])
        ]

    # Sample small number of examples
    num_samples = min(3, len(concrete_examples))
    if num_samples > 0:
        memory += random.sample(concrete_examples, num_samples)

    memory = [
        [{k: v for k, v in m.items() if k != "specifier"} for m in e]
        for e in memory
    ]

    return memory

import argparse

class fakepage:
        url = ""

def eval_sample(task_id: int, args: argparse.Namespace, sample: dict,agent) -> None:
    # initialize metrics
    element_acc, action_f1, step_success, success = [], [], [], []
    token_stats = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    conversation = []
    episode_length = len(sample["action_reprs"])

    try:
        exemplars = get_exemplars(args)
    except:
        exemplars = []
    # print(exemplars)

    sys_message = [
        {
            "role": "system",
            "content": "You are a large language model trained to navigate the web. Output the next action and wait for the next observation. Here is the action space:\n1. `CLICK [id]`: Click on an HTML element with its id.\n2. `TYPE [id] [value]`: Type a string into the element with the id.\n3. `SELECT [id] [value]`: Select a value for an HTML element by its id.",
        }
    ]

    prev_actions, prev_obs = [], []

    n = args.vf_budget

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

    for s, act_repr in zip(sample["actions"], sample["action_reprs"]):
        _, target_act = get_target_obs_and_act(s)
        pos_candidates = [
            c for c in s["pos_candidates"] if c["rank"] < args.top_k_elements
        ]

        # get query, obs, act
        target_obs, _ = get_top_k_obs(s, args.previous_top_k_elements)
        # Continue next loop if the ground truth element is not in the cleaned html
        if len(pos_candidates) == 0:
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            prev_obs.append("Observation: `" + target_obs + "`")
            prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")
            conversation.append("The ground truth element is not in cleaned html")
            continue

        # construct query
        query = []
        for o, a in zip(prev_obs, prev_actions):
            if len(query) == 0:
                query.append({
                    "role": "user",
                    "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n" + o,
                })
            else:
                query.append({"role": "user", "content": o})
            query.append({"role": "assistant", "content": a})

        obs, _ = get_top_k_obs(s, args.top_k_elements, use_raw=False)
        if len(query) == 0:
            query.append({
                "role": "user",
                "content": f"Task: {sample['confirmed_task']}\nTrajectory:\n"
                + "Observation: `" + obs + "`",
            })
        else:
            query.append({"role": "user", "content": "Observation: `" + obs + "`"})

        prev_obs.append("Observation: `" + target_obs + "`")
        prev_actions.append("Action: `" + target_act + "` (" + act_repr + ")")

        # Checking if token limit exceeded
        total_num_tokens = num_tokens_from_messages(sys_message + query, args.model)
        if total_num_tokens > MAX_TOKENS[args.model]:
            logger.info(
                f"Too many tokens in acting ({total_num_tokens} / {MAX_TOKENS[args.model]}), skipping..."
            )
            element_acc.append(0)
            action_f1.append(0)
            step_success.append(0)
            conversation.append(
                {
                    "input": sys_message + query,
                    "output": f"FAILED DUE TO THE CONTEXT LIMIT: {total_num_tokens}",
                }
            )
            continue

        # message
        demo_message = []
        for e_id, e in enumerate(exemplars):
            total_num_tokens = num_tokens_from_messages(
                sys_message + demo_message + e + query, args.model
            )
            if total_num_tokens > MAX_TOKENS[args.model]:
                logger.info(
                    f"Using {e_id} / {len(exemplars)} exemplars due to context limit"
                )
                break
            else:
                demo_message.extend(e)

        message = sys_message + demo_message + query
        try:
            response, info = generate_response(
                messages=message,
                model=args.model,
                temperature=args.temperature,
                stop_tokens=["Task:", "obs:"],
                n=n
            )
        except Exception as e:
            print(e)
            response = [""] * n
            info = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }

        trajectory = [{"observation": {"text": s['cleaned_html']}, "info": {"page": fakepage}, "url": ""}]
        # meta_data = {"action_history": ["None"] + [*map(lambda x: x["target_act"], filter(lambda x: "target_act" in x.keys() if isinstance(x, dict) else False, conversation))]}
        meta_data = {
    "action_history": ["None"] + prev_actions
}
        intent = sample["confirmed_task"]

        try:
            # raise Exception
            response = agent.next_action(
                trajectory,
                intent,
                meta_data,
                actions=response,
                branching_factor=n
            )[0]['raw_prediction']
        except Exception as e:
            response = response[0]

        conversation.append({"input": message, "output": response, "token_stats": info})
        for k, v in info.items():
            token_stats[k] += v
        pred_act = extract_from_response(response, "`")
        pred_op, pred_id, pred_val = parse_act_str(pred_act)
        target_op, _, target_val = parse_act_str(target_act)

        # calculate metrics
        pos_ids = [c["backend_node_id"] for c in s["pos_candidates"]][:1]
        if pred_id in pos_ids:
            element_acc.append(1)
        else:
            element_acc.append(0)
        action_f1.append(
            calculate_f1(
                construct_act_str(pred_op, pred_val),
                construct_act_str(target_op, target_val),
            )
        )
        conversation.append({"pred_act": pred_act, "target_act": target_act})
        if pred_act == target_act:
            step_success.append(1)
        else:
            step_success.append(0)

    # check the last episode_length of step_success, if all 1, then success = 1
    if np.sum(step_success[-episode_length:]) == episode_length:
        success.append(1)
    else:
        success.append(0)

    conversation.append(
        {
            "element_acc": element_acc,
            "action_f1": action_f1,
            "step_success": step_success,
            "success": success,
        }
    )
    log_dir = Path(f"{args.log_dir}/{args.model}/{args.benchmark}/{args.website}/{args.suffix}")
    # print(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = get_ist_timestamp()

    with open(os.path.join(log_dir, f"{task_id}_{timestamp}.json"), "w") as f:
        json.dump(conversation, f, indent=2)
   
