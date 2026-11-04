"""
Run DataModeling Evaluation
===========================

This script evaluates models on the DataModeling benchmark by creating a user, 
reading dataset samples, uploading related files, and generating responses 
with chat sessions. The model is expected to perform data modeling, training 
and prediction, and output the final submission file in CSV format.

Usage
-----
First, ensure the server is running (default: http://localhost:8000).

Basic usage with defaults:
    python eval_data_modeling.py

Specify custom parameters:
    python eval_data_modeling.py \
        --user_name "DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop" \
        --result_path "./results/DataModeling/gpt-4o-mini/"

Arguments
---------
--user_name : str
    Name of the user as a readable identifier 
    (default: DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop)

--result_path : str
    Directory where results will be stored 
    (default: ./results/DataModeling/gpt-4o-mini/)
"""

import json
import os
import uuid
import argparse
from pathlib import Path
from chat_test_asyncio import create_session, create_user, chat, upload_file, stop_session
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import autogen
from autogen.coding import LocalCommandLineCodeExecutor
from autogen import AssistantAgent, UserProxyAgent
from IPython.display import Image, display
from datawiseagent.common.config import global_config

# import fitz  # PyMuPDF
import json
import base64
import re
import time
import pandas as pd
from tqdm import tqdm
from typing import Optional
import glob
import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FALLBACK_RESULT_PATH = Path("./experimental_results/Datamodeling-DSBench/")


def resolve_result_path(cli_result_path: Optional[str | Path]) -> Path:
    """Resolve evaluation output path.

    Priority:
    1) Explicit CLI path (`--result_path`)
    2) `log.experience_dir` in config (e.g., configs/datamodeling.yaml)
    3) Historical fallback path used by this script
    """
    if cli_result_path:
        return Path(cli_result_path)

    configured_experience_dir = global_config.get("log", {}).get("experience_dir")
    if configured_experience_dir:
        configured_path = Path(configured_experience_dir)
        if not configured_path.is_absolute():
            configured_path = (PROJECT_ROOT / configured_path).resolve()
        return configured_path

    return (PROJECT_ROOT / FALLBACK_RESULT_PATH).resolve()


def main(
    user_id: uuid.UUID,
    user_name: Optional[str] = None,
    result_path: Optional[str | Path] = None,
    model: str = "gpt-4o-mini",
    start_from: int = 1,
    run_count: Optional[int] = None,
):

    # Get agent_type from the global config
    agent_type = global_config.get("agent", {}).get("agent_type", "single")

    resolved_result_path = resolve_result_path(result_path)
    Path(resolved_result_path).mkdir(parents=True, exist_ok=True)
    results_path = Path(resolved_result_path) / "results.jsonl"

    log_path = Path(resolved_result_path) / "log.txt"

    if not os.path.exists(results_path):
        Path(results_path).touch()

    if not os.path.exists(log_path):
        Path(log_path).touch()

    id2results = {}
    with open(results_path, encoding="utf-8", mode="r") as f:
        for line in f:
            item = json.loads(line)
            id2results[item["id"]] = item

    samples = []
    with open("./DataModeling/data.jsonl", encoding="utf-8", mode="r") as f:
        for line in f:
            samples.append(json.loads(line))

    print(f"samples count: {len(samples)}")

    print(f"id2results count: {len(id2results)}")

    instruction = "I have a data modeling task. You must give me the predicted results as a CSV file as detailed in the following content. You should try your best to predict the answer. I provide you with three files. One is training data, one is test data. There is also a sample file for submission."
    
    tasks_run = 0
    for i, item in enumerate(samples):
        if i + 1 < start_from:
            continue

        if run_count is not None and tasks_run >= run_count:
            print(f"Reached run_count limit of {run_count}. Exiting.")
            break
        
        tasks_run += 1

        if i in id2results:
            continue

        if i == 3:  #  or i == 17:
            continue

        # if i == 14 or i == 17 or i == 22 or i == 33 or i == 48:
        #     print(item)
        #     continue

        name = item["name"]
        with open(f"./DataModeling/data/task/{name}.txt", "r") as f:
            description = f.read()
        text = f"All three data files can be found in the folder `./input/`. After the data modeling, please give me the prediction results for the test file. You must run the code for the predicted results and save the answer as a csv file following the format of the given sample file for submission. The final submission file should be saved in the path `./working/submission.csv`."

        text = f"""All three data files can be found in the folder `./input/`. **You should use sklearn or pytorch to complete the task.** Any training scripts, models, and experiment log should be saved in `./input/`. After data modeling, provide the prediction results for the test file in the format specified by the sample submission file. Save the final submission to `./input/final_submission.csv`.
        """

        all_context = instruction + "\n" + description + "\n" + text

        try:
            session_id = uuid.UUID(
                create_session(
                    user_id=user_id, session_name=name, tool_mode="datamodeling"
                )
            )
            print(f"create session:{session_id} for case {name} with id {name}")

            data_split_path = f"./DataModeling/data/data_resplit/{name}"
            for root, dirs, files in os.walk(data_split_path):
                for file in files:
                    if file.endswith(".csv"):
                        full_path = os.path.join(root, file)
                        relative_path = os.path.relpath(full_path, data_split_path)

                        upload_file(
                            str(user_id),
                            str(session_id),
                            file_path=full_path,
                            filename_to_save=relative_path,
                        )

            start_time = time.time()

            chat_response = None
            time_consumed = None

            try:
                chat_response = chat(
                    str(user_id),
                    str(session_id),
                    query=all_context,
                    use_moa_router=(agent_type == 'concurrent'), # Passed here
                    work_mode="jupyter+script",
                )
            except httpx.TimeoutException:
                print("chat() 调用超时，自动设置 chat_response")
                chat_response = {
                    "user_content": all_context,
                    "response_content": "TIMEOUT=3600",
                }
                time_consumed = 3600

            end_time = time.time()

            if not time_consumed:
                time_consumed = end_time - start_time

            user_content = chat_response["user_content"]
            response_content = chat_response["response_content"]

            input_dir = f"../log/workspace/users/{user_id}/{session_id}/input/"
            work_dir = f"../log/workspace/users/{user_id}/{session_id}/working/"

            submission_files = []
            submission_files.extend(
                glob.glob(os.path.join(input_dir, "*submission*.csv"))
            )
            submission_files.extend(
                glob.glob(os.path.join(work_dir, "*submission*.csv"))
            )

            if submission_files:
                submission_file_path = f"../log/workspace/users/{user_id}/{session_id}/input/final_submission.csv"

                if not os.path.exists(submission_file_path):
                    submission_files.sort(
                        key=lambda x: os.path.getmtime(x), reverse=True
                    )
                    submission_file_path = submission_files[0]

            else:
                submission_file_path = f"../log/workspace/users/{user_id}/{session_id}/input/final_submission.csv"

            iteration_result = {
                "id": i,
                "name": name,
                "input_text": all_context,
                "response": response_content,
                "session_id": str(session_id),
                "user_id": str(user_id),
                "time": time_consumed,
                "cost": None,
                "model": model,
                "input": None,
                "output": None,
                "result_path": submission_file_path,
            }
            print(iteration_result)
            print(f"submission_file_path:{submission_file_path}")

            with open(results_path, encoding="utf-8", mode="a+") as f:
                f.write(json.dumps(iteration_result, ensure_ascii=False) + "\n")
                f.flush()
            
            # Stop the session to clean up Docker containers
            stop_session(str(user_id), str(session_id))

        except Exception as e:
            import traceback

            with open(log_path, encoding="utf-8", mode="a+") as f:
                f.write(traceback.format_exc())

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Run DataModeling evaluation.")

    parser.add_argument(
        "--user_name",
        type=str,
        default="DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop",
        help="Name of the user (default: DataModeling-gpt4o-mini-temperature=0-args=(7,6,8)-for-loop)",
    )

    parser.add_argument(
        "--result_path",
        type=str,
        default=None,
        help="Path to store results. If omitted, use log.experience_dir from config (fallback: ./experimental_results/Datamodeling-DSBench/).",
    )

    parser.add_argument(
        "--start_from",
        type=int,
        default=1,
        help="Start from the nth task (1-based).",
    )
    
    parser.add_argument(
        "--run_count",
        type=int,
        default=None,
        help="Number of tasks to run.",
    )

    args = parser.parse_args()

    user_id = create_user(username=args.user_name)
    print(f"Created user_id: {user_id}")

    result_path = resolve_result_path(args.result_path)
    print(f"Results and logs will be stored in: {result_path}")

    main(
        user_id=user_id,
        user_name=args.user_name,
        result_path=result_path,
        start_from=args.start_from,
        run_count=args.run_count,
        # "./results/DataModeling/gpt-4o-test/",
        # result_path="./results/DataModeling/qwen25-72B/",
        # model="Qwen/Qwen2.5-72B-Instruct",
    )
