"""
One script to evaluate Datawise Agent on the MatplotBench.

Usage:
    First of all, start the server at http://localhost:8000

    Example:
        python ./MatplotBench/scripts/model_eval.py --dir ./experimental_results/MatplotBench/gpt-4o-mini

Arguments:
    --dir Directory path to the input result.jsonl file and output result_with_score.jsonl file.
"""

import base64
import json
import logging
import os
import re
import shutil
import glob
import sys
from openai import OpenAI
from pathlib import Path
import argparse
from pathlib import Path
import yaml # Import the yaml library

# Load the default config
config_path = Path(__file__).parents[3] / "configs" / "default_config_bestone.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Extract agent_type from the config
agent_type = config['agent']['agent_type']

parser = argparse.ArgumentParser(description="Process result files with scoring.")
parser.add_argument(
    "--dir",
    type=str,
    required=True,
    help="Directory path to the input result.jsonl file and output result_with_score.jsonl file.",
)
args = parser.parse_args()

results_path = Path(args.dir) / f"{agent_type}_result.jsonl"
output_path = Path(args.dir) / f"{agent_type}_result_with_score.jsonl"


API_KEY = os.environ["OPENAI_API_KEY"]
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

from pathlib import Path
from PIL import Image
import numpy as np


def is_blank_image(path, tolerance=0):
    """
    判断图片是否为空白图片。

    参数:
    - path (str): 图片文件路径。
    - tolerance (int): 颜色容差，默认为0表示完全相同。

    返回:
    - bool: 如果图片为空白，返回True；否则返回False。
    """
    try:
        with Image.open(path) as img:
            img = img.convert("RGB")  # 转换为RGB模式
            np_img = np.array(img)
            first_pixel = np_img[0, 0]
            if tolerance == 0:
                return np.all(np_img == first_pixel)
            else:
                diff = np.abs(np_img - first_pixel)
                return np.all(diff <= tolerance)
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return False


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def gpt_4v_evaluate(ground_truth_path, image_path):
    if not os.path.exists(f"{image_path}"):
        empty_image_path = "./MatplotBench/data/ground_truth/empty.png"
        base64_image = encode_image(f"{empty_image_path}")

    else:
        base64_image = encode_image(image_path)

    ground_truth = encode_image(ground_truth_path)

    response = client.chat.completions.create(
        # model="gpt-4-vision-preview",
        model="gpt-4o-2024-08-06",
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""You are an excellent judge at evaluating visualization plots between a model generated plot and the ground truth. You will be giving scores on how well it matches the ground truth plot.
               
               The generated plot will be given to you as the first figure. If the first figure is blank, that means the code failed to generate a figure.
               Another plot will be given to you as the second figure, which is the desired outcome of the user query, meaning it is the ground truth for you to reference.
               Please compare the two figures head to head and rate them.
               Suppose the second figure has a score of 100, rate the first figure on a scale from 0 to 100.
               Scoring should be carried out in the following aspect:
               1. Plot correctness: 
               Compare closely between the generated plot and the ground truth, the more resemblance the generated plot has compared to the ground truth, the higher the score. The score should be proportionate to the resemblance between the two plots.
               In some rare occurrence, see if the data points are generated randomly according to the query, if so, the generated plot may not perfectly match the ground truth, but it is correct nonetheless.
               Only rate the first figure, the second figure is only for reference.
               If the first figure is blank, that means the code failed to generate a figure. Give a score of 0 on the Plot correctness.
                After scoring from the above aspect, please give a final score. The final score is preceded by the [FINAL SCORE] token.
               For example [FINAL SCORE]: 40.""",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{ground_truth}",
                        },
                    },
                ],
            }
        ],
        max_tokens=1000,
    )

    evaluation_result = response.choices[0].message.content
    print(evaluation_result)

    return evaluation_result

output_path.parent.mkdir(parents=True, exist_ok=True)
if not os.path.exists(output_path):
    Path(output_path).touch()


def is_complete_result(item):
    path = item.get("result_path")
    if not path or not os.path.exists(path):
        return False
    return not is_blank_image(path)


id2result = {}
with open(output_path, encoding="utf-8", mode="r") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        item["is_complete"] = is_complete_result(item)
        id2result[int(item["id"])] = item

print(len(id2result))

new_id2result = {}

with open(results_path, encoding="utf-8", mode="r") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)

        idx = int(item["id"])
        if idx not in id2result:
            # evaluate
            try:
                ground_truth_path = item["ground_truth_path"]
                image_path = item["result_path"]

                evaluation_result = gpt_4v_evaluate(ground_truth_path, image_path)
                print(f"evaluation_result:\n{evaluation_result}")
                print(ground_truth_path)
                print(image_path)
                if evaluation_result:
                    item["evaluation_result"] = evaluation_result
                    match = re.search(
                        r"\[FINAL SCORE\]:\s*[^0-9]*?(\d+)", evaluation_result
                    )
                    if match:
                        score = int(
                            match.group(1)
                        )  # Extract and convert the score to an integer
                        print(f"score : {score}")

                        item["score"] = score
                        item["is_complete"] = is_complete_result(item)

                        new_id2result[idx] = item

                    else:
                        raise ValueError(
                            "Score Extraction Failed."
                        )  # Return None if no final score is found

            except Exception as e:

                import traceback

                traceback.print_exc()

all_id2result = dict(id2result)
all_id2result.update(new_id2result)

# calculate whole scores
total_scores = 0.0
no_less_than_80_score_cnt = 0
completion_cnt = 0

for idx, item in all_id2result.items():
    score = float(item.get("score", 0))

    if score >= 80:
        no_less_than_80_score_cnt += 1

    complete = bool(item.get("is_complete", False))
    if complete:
        completion_cnt += 1

    total_scores += score

total_count = len(all_id2result)
average_score = total_scores / total_count if total_count else 0.0
completion_rate = completion_cnt / total_count if total_count else 0.0
score_ge_80_rate = no_less_than_80_score_cnt / total_count if total_count else 0.0

# persist completion rate (and completion flag) into every record
for item in all_id2result.values():
    item["completion_rate"] = completion_rate
    if "is_complete" not in item:
        item["is_complete"] = is_complete_result(item)

# rewrite output file so all records have consistent fields
with open(output_path, encoding="utf-8", mode="w") as f:
    for idx in sorted(all_id2result.keys()):
        f.write(json.dumps(all_id2result[idx], ensure_ascii=False) + "\n")

print(f"average score: {average_score}")
print(f"competion rate: {completion_rate}")
print(f"score >=80 rate: {score_ge_80_rate}")
