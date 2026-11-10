import openai


def submit_sft_job(
    client: openai.OpenAI,
    training_file_id: str,
    model: str,
    n_epochs: int | str = "auto",
    batch_size: int | str = "auto",
    learning_rate_multiplier: float | str = "auto",
    suffix: str = "",
) -> dict:
    """Submit a supervised fine-tuning job on an OpenAI-format (`{"messages": [...]}`)
    file without waiting; returns the job record."""
    job = client.fine_tuning.jobs.create(
        training_file=training_file_id,
        model=model,
        method={
            "type": "supervised",
            "supervised": {
                "hyperparameters": {
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate_multiplier": learning_rate_multiplier,
                },
            },
        },
        suffix=suffix or None,
    )
    return job.model_dump()
