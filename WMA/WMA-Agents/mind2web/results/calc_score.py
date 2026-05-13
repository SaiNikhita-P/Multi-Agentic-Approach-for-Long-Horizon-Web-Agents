import json
import os
from functools import reduce
import argparse
from datetime import datetime


def extract_timestamp(filename):
    try:
        # example: 0_20260317_183245.json
        ts_str = filename.split("_", 1)[1].replace(".json", "")
        return datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
    except:
        return None


def collect_file_paths(path, min_dt):
    file_paths = []

    for f in os.listdir(path):
        workflow_dir = os.path.join(path, f, "workflow")
        if not os.path.exists(workflow_dir):
            continue

        for x in os.listdir(workflow_dir):
            ts = extract_timestamp(x)

            if ts and ts >= min_dt:
                file_paths.append(os.path.join(workflow_dir, x))

    return file_paths

def main(results_dir: str, start_time_str: str):
    min_dt = datetime.strptime(start_time_str, "%Y%m%d_%H%M%S")

    for bench in ["test_domain", "test_task", "test_website"]:
    # for bench in ["test_domain"]:
        path = os.path.join(results_dir, bench)
        if not os.path.exists(path):
            continue

        file_paths = collect_file_paths(path, min_dt)

        if len(file_paths) == 0:
            print(f"\n No files found after {start_time_str} for {bench}")
            continue

        print(f"\n {bench}: using {len(file_paths)} files")

        jss = []
        for fp in file_paths:

            try:
                with open(fp, "r") as f:
                    content = f.read().strip()

                    # Skip empty files
                    if not content:
                        print(f"Skipping empty file: {fp}")
                        continue

                    js = json.loads(content)

            except Exception as e:
                print(f"Skipping corrupted file: {fp} | Error: {e}")
                continue

            # OLD format (list with conversation)
            if isinstance(js, list):
                metrics = js[-1]

            # NEW format (metrics-only dict)
            elif isinstance(js, dict):
                metrics = js

                # Ensure step_success exists
                # if "step_success" not in metrics:
                #     metrics["step_success"] = metrics.get("element_acc", [])

                if "step_success" not in metrics:
                    ea = metrics.get("element_acc", [])
                    af1 = metrics.get("action_f1", [])

                    metrics["step_success"] = [
                        int(e == 1 and a == 1.0)
                        for e, a in zip(ea, af1)
                    ]

                # Ensure success exists
                if "success" not in metrics:
                    if "step_success" in metrics:
                        metrics["success"] = [int(all(metrics["step_success"]))]
                    else:
                        metrics["success"] = [0]

            else:
                continue

            jss.append(metrics)

       
        print(f"\n===== {bench} =====")


        e_accs = sum([x["element_acc"] for x in jss], [])
        a_f1 = sum([x["action_f1"] for x in jss], [])
        stepsr = sum([x["step_success"] for x in jss], [])
        sr = [x["success"][0] for x in jss]
        f = lambda x, y: print(x, round(y*100, 1))

        f("Element Acc:", sum(e_accs) / len(e_accs))
        f("Action F1:  ", sum(a_f1) / len(a_f1))
        f("Step SR:    ", sum(stepsr) / len(stepsr))
        f("SR:         ", sum(sr) / len(sr))

        if len(e_accs) == 0:
            print("No valid data found!")
            return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--start_time", type=str, required=True,
                        help="Format: YYYYMMDD_HHMMSS")

    args = parser.parse_args()

    main(args.results_dir, args.start_time)