"""
main.py
=======

Backend server for DatawiseAgent built with FastAPI.

This server provides RESTful APIs and WebSocket endpoints for:
- User creation and session management
- Chatting with the agent
- Uploading and processing files
- Querying session data and evaluation logs

⚠️ Note:
This repository only open-sources the backend server.
The WebSocket endpoint at `/register_websocket/{user_id}/{session_id}`
is reserved for frontend integration (not provided here).
"""

import dotenv

dotenv.load_dotenv()
import uuid
from uuid import UUID
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    UploadFile,
    File,
    Body,
    Form,
    HTTPException,
)
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, ValidationError
from typing import List
from pathlib import Path
import json
import os
import asyncio
from starlette.websockets import WebSocketState
import argparse

from datawiseagent.memory.session import SessionContent, CellHistoryMemory
from datawiseagent.common.config import global_config
from datawiseagent.agents.datawise_agent import (
    llm_config,
    agent_config,
    code_executor_config,
    SESSIONS_LOG_PATH,
)
from datawiseagent.management.manager import DatawiseAgentManager
from datawiseagent.common.types import (
    CreateSessionConfig,
    CreateUserParam,
    UserSessionConfig,
    ChatParam,
    SessionInfo,
)
from datawiseagent.common.log import logger
import os
from datawiseagent.agents.os.agent_pool_manager import agent_pool_manager # Import agent_pool_manager

# 强制覆盖环境变量，确保 Docker 通信不走代理
os.environ["NO_PROXY"] = "localhost,127.0.0.1,0.0.0.0,172.17.0.1"

# Get agent_type from the global config
agent_type = global_config.get("agent", {}).get("agent_type", "single")

app = FastAPI()
agent_manager = DatawiseAgentManager(agent_type=agent_type)
logger.info(f"Running with agent type: {agent_type} (from config)")
logger.info(f"Router selection strategy: {agent_pool_manager.selection_strategy}")

def _initialize_task_counter(task_id_start_from: int | None) -> None:
    if task_id_start_from is None:
        return

    try:
        task_id_start_from = int(task_id_start_from)
        if task_id_start_from < 1:
            raise ValueError("task_id_start_from must be >= 1")
        if hasattr(agent_manager.agent_class, "_global_task_counter"):
            # The runtime increments counter before assigning task_id, so store start-1.
            setattr(
                agent_manager.agent_class,
                "_global_task_counter",
                task_id_start_from - 1,
            )
            logger.info(
                f"Initialized task counter: first task_id will start from {task_id_start_from}."
            )
        else:
            logger.warn(
                f"--task-id-start-from is set, but current agent type "
                f"'{agent_type}' does not expose a global task counter. Ignored."
            )
    except (TypeError, ValueError) as e:
        logger.error(f"Invalid --task-id-start-from value '{task_id_start_from}': {e}. Ignored.")

# Ensure the experience directory from config exists and log its path
configured_experience_dir = global_config.get("log", {}).get("experience_dir", "datawiseagent/agents/os/experience/")
os.makedirs(configured_experience_dir, exist_ok=True)
logger.info(f"Experience files will be stored in: {configured_experience_dir}")



@app.post("/create_user/")
async def create_user(param: CreateUserParam):
    username = param.username
    user_id = agent_manager.create_user(username)

    return {"user_id": str(user_id), "username": username}


@app.post("/users/{user_id}/sessions/{session_name}")
async def create_session(user_id: UUID, session_name: str, config: CreateSessionConfig):

    agent = agent_manager.get_agent(user_id)

    if config.reset_llm_config == None:
        config.reset_llm_config = llm_config

    if config.reset_code_executor == None:
        config.reset_code_executor = code_executor_config

    session_id = await agent.create_new_session(
        config.reset_llm_config,
        config.reset_code_executor,
        session_name,
        config.tool_mode,
    )
    return {"session_id": str(session_id), "session_name": session_name}


@app.post("/chat/")
async def chat(param: ChatParam):
    param_dict = param.model_dump(exclude_unset=True)
    user_id = param.user_id
    session_id = param.session_id

    agent = agent_manager.get_agent(user_id)
    if session_id not in agent.sessions:
        return {"error": "Invalid session ID"}
    if "query" in param_dict:
        param_dict.pop("user_id", None)

    last_user_content, last_response_content = await agent.chat(**param_dict)
    return {
        "user_content": last_user_content,
        "response_content": last_response_content,
    }


@app.post("/upload/")
async def upload_file(
    user_id: UUID = Form(...),
    session_id: UUID = Form(...),
    files: List[UploadFile] = File(...),
):
    agent = agent_manager.get_agent(user_id)
    if session_id not in agent.sessions:
        return {"error": "Invalid session ID"}

    paths = await agent.process_uploaded_files(session_id, files)

    return {"message": "Files processed successfully", "results": paths}


@app.get("/users", response_model=List[str])
async def list_users():
    user_dir = Path(SESSIONS_LOG_PATH) / "users"
    if not user_dir.exists():
        return []
    user_ids = [d.name for d in user_dir.iterdir() if d.is_dir()]
    return user_ids


@app.get("/users/{user_id}/sessions", response_model=List[SessionInfo])
async def list_sessions(user_id: UUID):
    """
    List all sessions for a given user.

    Args:
        user_id (UUID): The unique identifier of the user.

    Returns:
        List[dict]: A list of dictionaries containing session_id and session_name.
    """
    session_dir = Path(SESSIONS_LOG_PATH) / "users" / str(user_id) / "sessions"

    if not session_dir.exists():
        raise HTTPException(
            status_code=404, detail=f"User {user_id}'s sessions not found"
        )
    sessions = []
    for session_id_dir in session_dir.iterdir():
        if session_id_dir.is_dir():
            session_id = session_id_dir.name
            # Find all JSON files in the session_id directory
            json_files = list(session_id_dir.glob("*.json"))
            if not json_files:
                continue  # Skip if no JSON files are found

            # Assuming there's only one session_name.json per session_id directory
            session_file = json_files[0]
            session_name = session_file.stem  # Extract filename without extension

            sessions.append(
                SessionInfo(session_id=session_id, session_name=session_name)
            )

    return sessions


@app.get("/users/{user_id}/sessions/{session_id}/data", response_model=SessionContent)
async def get_session_data(user_id: UUID, session_id: UUID):
    session_id_dir: Path = (
        Path(SESSIONS_LOG_PATH) / "users" / str(user_id) / "sessions" / str(session_id)
    )
    if not session_id_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id}'s session {session_id} directory not found!",
        )
    json_files = list(session_id_dir.glob("*.json"))
    if not json_files:
        raise HTTPException(
            status_code=404,
            detail=f"User {user_id}'s session {session_id} JSON file not found!",
        )
    # Assuming there's only one session_name.json per session_id directory
    session_file = json_files[0]

    with open(session_file, "r") as f:
        session_data = json.load(f)

    try:
        validated_session = SessionContent.from_json(session_data)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail=e.errors())

    return validated_session


@app.delete("/users/{user_id}/sessions/{session_id}")
async def delete_session(user_id: UUID, session_id: UUID):
    """
    Stops a session and releases all associated resources, including the Jupyter kernel and Docker container.
    """
    agent = agent_manager.get_agent(user_id)
    if session_id not in agent.sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found for user {user_id}")
    
    try:
        stopped_session_id = await agent.stop_session(session_id)
        return {"message": f"Session {stopped_session_id} stopped and cleaned up successfully"}
    except Exception as e:
        logger.error(f"Error stopping session {session_id} for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop session {session_id}. Error: {str(e)}")


@app.websocket("/register_websocket/{user_id}/{session_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: UUID, session_id: UUID):
    """
    WebSocket endpoint for real-time communication between frontend and backend.

    Note:
        - This is a reserved interface intended for a frontend client (not open-sourced here).
        - Used to stream messages and keep sessions alive with ping/pong.
        - Backend users typically do not need to call this directly.
    """
    await websocket.accept()
    agent_manager.add_websocket(user_id, session_id, websocket)

    async def send_ping():
        while True:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.send_json({"type": "ping"})
            await asyncio.sleep(10)

    ping_task = asyncio.create_task(send_ping())

    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received from {user_id}/{session_id}: {data}")
    except WebSocketDisconnect:
        agent_manager.remove_websocket(user_id, session_id, websocket)
    finally:
        ping_task.cancel()


# CORS configuration (only needed if you want to serve a web frontend)
# origins = ["http://localhost:3000"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="Run DatawiseAgent backend server.")
    parser.add_argument(
        "--task-id-start-from",
        type=int,
        default=None,
        help="First task_id to use. For example, 11 means next task_id starts at 11.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=global_config.get("server", {}).get("port", 8001),
        help="Server port (default from config).",
    )
    args = parser.parse_args()

    _initialize_task_counter(args.task_id_start_from)
    uvicorn.run(app, host="0.0.0.0", port=args.port, reload=False)
