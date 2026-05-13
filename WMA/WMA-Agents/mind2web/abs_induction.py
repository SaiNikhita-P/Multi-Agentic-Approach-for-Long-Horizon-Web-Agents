"""Induce Abstraction from Past Agent Experiences (for A2)."""

import os
import json
import glob
from typing import List, Dict, Any
from mind2web.utils.data import load_json, format_examples_abs
from mind2web.utils.env import is_io_dict


def get_trajectory(path: str) -> List[Dict[str, str]]:
    """
    Extract trajectory from result file.
    
    Args:
        path: Path to the result JSON file
        
    Returns:
        List of trajectory steps with environment and action information
    """
    trajectory = []
    with open(path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    
    for item in result:
        if not is_io_dict(item):
            continue
        step = {
            "env": "# " + item["input"][-1]["content"],
            "action": item["output"],
        }
        trajectory.append(step)
    return trajectory

def abstraction_induction(args, samples, caller, i):
    """Abstraction Induction."""
    print("abstraction_induction being performed!! Hurray!!")
    abs_intro = "Here are the extracted abstractions that represent common sub-routines:\n{element_type} represents the corresponding type of the chosen element.\n"
    
    # load model predictions and format examples
    sys_prompt = open(args.abs_sys_prompt_path, 'r').read()  #(instruction_abs_basic.txt)  # semantic
    example_prompt = open(args.abs_exemplar_prompt_path, 'r').read()  #(one_shot_abs_basic.txt)
    episodic_sys_prompt = open(args.episodic_abs_sys_prompt_path, 'r').read()  #(instruction_abs_episodic)  # episodic
    episodic_example_prompt = open(args.episodic_abs_exemplar_prompt_path, 'r').read() #(one_shot_abs_episodic.txt)
    sys_prompt_domain=open(args.abs_sys_prompt_domain_path,'r').read()
    example_prompt_domain=open(args.abs_exemplar_prompt_domain_path, 'r').read()
    episodic_sys_prompt_domain = open(args.episodic_abs_sys_prompt_domain_path, 'r').read()
    episodic_example_prompt_domain = open(args.episodic_abs_exemplar_prompt_domain_path, 'r').read()



    if args.flag == "abs_task":
        # Task-based abstraction induction: q_i + a_{i-1} = a_i, a_i + q_{i+1} = a_i_hat
        
        # Step 1: Semantic memory induction
        semantic_path = args.semantic_workflow_path

        # Website-level semantic abstraction
        if args.semantic_workflow_path.exists():
            last_website_abs = open(args.semantic_workflow_path, "r", encoding="utf-8").read()
            last_website_abs = last_website_abs.replace(abs_intro, "").strip("\n")
        else:
            last_website_abs = ""
            open(args.semantic_workflow_path, "w").close()

        # Domain-level semantic abstraction
        if args.domain_semantic_workflow_path.exists():
            last_domain_abs = open(args.domain_semantic_workflow_path, "r", encoding="utf-8").read()
            last_domain_abs = last_domain_abs.replace(abs_intro, "").strip("\n")
        else:
            last_domain_abs = ""
            open(args.domain_semantic_workflow_path, "w").close()

        combined_last_abs = (
            "[Website-level abstractions]\n"
            f"{last_website_abs}\n\n"
            "[Domain-level abstractions]\n"
            f"{last_domain_abs}"
        )
            
        

        results_path = os.path.join(
            args.output_dir,       
            args.model,
            args.benchmark,
            args.website,
            "workflow"
        )

        # Match WMA-style filenames
        pattern = os.path.join(results_path, f"{i}_*.json")

        import glob
        matched_files = glob.glob(pattern)

        # Debugging to find the matched files 
        print("Looking in:", results_path)
        print("Pattern:", pattern)
        print("Found files:", matched_files)

        if len(matched_files) == 0:
            raise FileNotFoundError(
                f"No result file found for task {i} and model {args.model}"
            )
        

        if args.domain_semantic_workflow_path.exists():
            prev_domain_abs = args.domain_semantic_workflow_path.read_text().strip()
        else:
            prev_domain_abs = ""


        # Pick the most recent file (by timestamp in filename)
        # last_task_path = sorted(matched_files)[-1]   

        if len(matched_files) == 0:
            raise FileNotFoundError(
                f"No result file found for task {i} in {results_path}"
            )

        # Pick most recent file (robust)
        last_task_path = max(matched_files, key=os.path.getmtime) 


        trajectory = get_trajectory(last_task_path)

        
        print(f"[ABSTRACTION] Using trajectory file: {last_task_path}")
        
        examples = [
            {
                "confirmed_task": samples[i]["confirmed_task"],
                "action_reprs": ["        "+step["action"] for step in trajectory],
            }
        ]
        prompt = format_examples_abs(examples, None, None)

        split_prompt = example_prompt.split("<Split token>")
        if len(split_prompt) <= 2:
            raise ValueError("Invalid example prompt format: missing split tokens")
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"<Lastest solved task>\n{split_prompt[0]}"},
            {"role": "user", "content": f"<Existing abstractions>\n{split_prompt[1]}"},
            {"role": "assistant", "content": f"<Extracted abstractions>\n{split_prompt[2]}"},
            {"role": "user", "content": f"<Lastest solved task>\n{prompt}"},
            {"role": "user", "content": f"<Existing abstractions>\n{combined_last_abs}"}
        ]
        
        website_semantic_abs = caller.call(messages=messages)
        website_semantic_abs = website_semantic_abs.replace("<Extracted abstractions>", "")
        website_semantic_abs = abs_intro + website_semantic_abs
        with open(semantic_path, 'w', encoding='utf-8') as fw:
            fw.write(website_semantic_abs)

        messages_domain = [
        {"role": "system", "content": sys_prompt_domain},
        {"role": "user", "content": f"<Existing Domain Abstraction>\n{prev_domain_abs}"},
        {"role": "user", "content": f"<Website Abstraction>\n{website_semantic_abs}"}
    ]

        domain_abs = caller.call(messages=messages_domain)


        with open(args.domain_semantic_workflow_path, "w", encoding="utf-8") as fw:
            fw.write(domain_abs)


        if int(i+1) < len(samples):
            episodic_split_prompt = episodic_example_prompt.split("<Split token>")
            next_task = "## Query 2: " + samples[i+1]["confirmed_task"]
            messages = [
                {"role": "system", "content": episodic_sys_prompt},
                {"role": "user", "content": f"<Next task>\n{episodic_split_prompt[0]}"},
                {"role": "user", "content": f"<Existing abstractions>\n{episodic_split_prompt[1]}"},
                {"role": "assistant", "content": f"<Extracted abstractions>\n{episodic_split_prompt[2]}"},
                {"role": "user", "content": f"<Next task>\n{next_task}"},
                {"role": "user", "content": f"<Existing abstractions>\n{website_semantic_abs}"}
            ]
        else:
            # FALLBACK: no episodic info → reuse semantic abstraction
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"<Existing abstractions>\n{website_semantic_abs}"}
            ]
    else:
        raise ValueError(f"Invalid flag: {args.flag}")

    # Generate abstraction response
    response = caller.call(messages=messages)
    response = response.replace("<Extracted abstractions>", "")
    response = abs_intro + response

    print("[ABSTRACTION] Learned abstraction:")
    print(response[:500])
    print(f"[ABSTRACTION] Writing final abstraction to {args.workflow_path}")
    
    with open(args.workflow_path, 'w') as fw:
        # fw.write("\n\n" + response)
        fw.write(response)
    
    abs_save_path = os.path.join(
        args.results_dir,
        f"abs_{i}_{args.model}.txt"
    )
    with open(abs_save_path, "w", encoding="utf-8") as fw:
        fw.write(response)    
