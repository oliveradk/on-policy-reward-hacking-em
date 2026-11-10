from pathlib import Path

import openai

from rewardhacking_training.train.train_providers.types import assistant_content, convert_file


def standardized_pair_to_openai(row: dict) -> dict:
    """A standardized DPO row in OpenAI's preference fine-tuning format (the assistant
    content is the flattened `full_response`)."""
    return {
        "input": {"messages": row["input"]["messages"]},
        "preferred_output": [
            {"role": "assistant", "content": assistant_content(row["preferred_output"])}
        ],
        "non_preferred_output": [
            {"role": "assistant", "content": assistant_content(row["non_preferred_output"])}
        ],
    }


def convert_standardized_file_to_openai(in_path: Path, out_path: Path) -> Path:
    return convert_file(in_path, out_path, standardized_pair_to_openai)


def submit_dpo_job(
    client: openai.OpenAI,
    training_file_id: str,
    model: str,
    beta: float = 0.1,
    n_epochs: int = 1,
    batch_size: int | str = "auto",
    learning_rate_multiplier: float | str = "auto",
    suffix: str = "",
) -> dict:
    """Submit without waiting; returns the job record."""
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        method={
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": beta,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": learning_rate_multiplier,
                },
            },
        },
        suffix=suffix or None,
    )
    return job.model_dump()
