from __future__ import annotations

import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv


def get_client() -> openai.OpenAI:
    load_dotenv()
    return openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def upload_training_file(client: openai.OpenAI, file_path: Path) -> str:
    with open(file_path, "rb") as f:
        result = client.files.create(file=f, purpose="fine-tune")
    return result.id


def save_job_info(job_info: dict, output_dir: Path) -> None:
    (Path(output_dir) / "job_info.json").write_text(
        json.dumps(job_info, indent=2, default=str)
    )
