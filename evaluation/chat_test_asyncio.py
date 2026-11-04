import sys
import os
import uuid
import httpx
import websockets
import asyncio
import time
from pathlib import Path
from typing import Optional, Literal

# Add the project root to the Python path to resolve module imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datawiseagent.common.config import global_config

# configure server address
port = global_config.get("server", {}).get("port", 8001)
BASE_URL = f"http://localhost:{port}"
BASE_WS_URL = f"ws://localhost:{port}/register_websocket"

DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=1200.0, write=1200.0, pool=30.0)
SESSION_CREATION_TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=3600.0, pool=30.0)
CHAT_TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=120.0, pool=30.0)
MAX_RETRIES = 2


def _request_with_retry(method: str, url: str, *, timeout: httpx.Timeout, **kwargs):
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = httpx.request(method=method, url=url, timeout=timeout, **kwargs)
            response.raise_for_status()
            return response
        except (httpx.TimeoutException, httpx.ConnectError) as e:
            if attempt >= MAX_RETRIES:
                raise
            backoff = 2**attempt
            print(
                f"Request {method} {url} failed ({type(e).__name__}), retrying in {backoff}s..."
            )
            time.sleep(backoff)


async def register_websocket(user_id: str, session_id: str):
    ws_url = f"{BASE_WS_URL}/{user_id}/{session_id}"
    async with websockets.connect(ws_url) as websocket:
        print(f"Registered WebSocket for {user_id}/{session_id}")
        while True:
            message = await websocket.recv()
            print(f"Received message from {session_id}: {message}")


def create_user(username: str):
    print("Begin to Create User")
    response = _request_with_retry(
        "POST",
        f"{BASE_URL}/create_user/",
        json={"username": username},
        timeout=DEFAULT_TIMEOUT,
    )
    response_data = response.json()
    print("Create User Response:", response_data)
    return response_data["user_id"]


def create_session(
    user_id: str,
    session_name: str,
    tool_mode: Literal["default", "dsbench", "datamodeling"] = "default",
):
    response = _request_with_retry(
        "POST",
        f"{BASE_URL}/users/{user_id}/sessions/{session_name}",
        json={
            "tool_mode": tool_mode,
        },
        timeout=SESSION_CREATION_TIMEOUT,
    )
    response_data = response.json()
    print("Create Session Response:", response_data)
    return response_data["session_id"]


def chat(
    user_id: str,
    session_id: str,
    query: str,
    work_mode: Literal["jupyter", "jupyter+script"] = "jupyter",
    use_moa_router: bool = False,
):
    json_payload = {
        "user_id": user_id,
        "session_id": session_id,
        "query": query,
        "work_mode": work_mode,
    }
    if use_moa_router:
        json_payload["use_moa_router"] = True
        
    response = _request_with_retry(
        "POST",
        f"{BASE_URL}/chat/",
        json=json_payload,
        timeout=CHAT_TIMEOUT,
    )
    response_data = response.json()
    print("Chat Response:", response_data)
    return response_data


def upload_file(
    user_id: str,
    session_id: str,
    file_path: str,
    filename_to_save: Optional[str] = None,
):
    with open(file_path, "rb") as f:
        # filename = Path(file_path).name
        if filename_to_save == None:
            filename = Path(file_path).name
        else:
            filename = filename_to_save

        files = {"files": (filename, f)}

        response = _request_with_retry(
            "POST",
            f"{BASE_URL}/upload/",
            data={"user_id": user_id, "session_id": session_id},
            files=files,
            timeout=DEFAULT_TIMEOUT,
        )
    response_data = response.json()
    print("Upload File Response:", response_data)


def delete_session(user_id: str, session_id: str):
    """
    Sends a request to the server to stop and delete a session, cleaning up all associated resources.
    """
    try:
        response = _request_with_retry(
            "DELETE",
            f"{BASE_URL}/users/{user_id}/sessions/{session_id}",
            timeout=DEFAULT_TIMEOUT,
        )
        response_data = response.json()
        print("Delete Session Response:", response_data)
        return response_data
    except httpx.HTTPStatusError as e:
        print(f"Error deleting session {session_id}: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"An error occurred while requesting to delete session {session_id}: {e}")
        return None


def stop_session(user_id: str, session_id: str):
    """
    Sends a request to the server to stop a session and release its resources.
    """
    try:
        response = _request_with_retry(
            "DELETE",
            f"{BASE_URL}/users/{user_id}/sessions/{session_id}",
            timeout=DEFAULT_TIMEOUT,
        )
        response_data = response.json()
        print("Stop Session Response:", response_data)
        return response_data
    except httpx.HTTPStatusError as e:
        print(f"Error stopping session {session_id}: {e.response.status_code} - {e.response.text}")
        return None
    except httpx.RequestError as e:
        print(f"An error occurred while requesting to stop session {session_id}: {e}")
        return None



async def main():
    # asyncio.run(main())
    user_id = "test_user_id"  # 替换为真实的用户 ID
    session_ids = ["session1", "session2", "session3"]  # 替换为实际会话

    tasks = [register_websocket(user_id, session_id) for session_id in session_ids]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    # test
    user_id = create_user("test_user")

    import pdb

    pdb.set_trace()

    query = """You are solving this machine learning tasks of time series classification: 
The dataset presented here (the Ethanol Concentration dataset) comprises real-world time series data. We have splitted the dataset into three parts of train, valid and test. The input is a sequence of observed features (INPUT_SEQ_LEN=1751, INPUT_DIM=3). Your task is to predict the labels for the given sequence, where the label is in range of {0, 1, 2, 3}. The evaluation metric is accuracy.
We provide an overall pipeline in train.py. Now fill in the provided train.py script to train a time series classification model to get a good performance on the given fixed sequences.
In this task, you should work in `env/` folder, which means the working memory when executing any code should be in `env/`. So change the working memeory to `env` FIRST before you start your work. You should try your best to complete the task and optimize the performance.

"""
    # session_id = create_session(user_id, "test_session")
    # chat(user_id, session_id, query)

    query = "Download the file from the following link: https://github.com/Luffyzm3D2Y/CHEF_assignment_for_NLP/raw/refs/heads/main/test.csv. This is a phenotype file related to patients. Please first review the contents of the file, then investigate whether there are significant differences in survival rates and infection rates among different age groups."
    session_id = create_session(user_id, "test_session")
    chat(user_id, session_id, query)

    query = "download data from https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv and plot a visualization that tells us about the relationship between weight and horsepower. Save the plot to a file. Print the fields in a dataset before visualizing it."
    session_id = create_session(user_id, "test_session")
    chat(user_id, session_id, query)

    query = """从以下链接下载文件：`https://github.com/Luffyzm3D2Y/CHEF_assignment_for_NLP/raw/refs/heads/main/test.csv`。 这是一个和病人相关的表型文件。请你先了解下文件内容，然后探究一下不同年龄段的人群在生存率和感染率上是否存在显著差异。你需要使用中文回答我的问题。"""
    chat(user_id, session_id, query)

    # upload_file(user_id, session_id, "test.csv")

    # interactive chat
    while True:
        user_query = input("Input your query (or 'exit' to quit): ")
        if user_query.lower() == "exit":
            print("Exiting...")
            break
        chat(user_id, session_id, user_query)
