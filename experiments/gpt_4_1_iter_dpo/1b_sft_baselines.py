# The two off-policy SFT baselines on gpt-4.1, with the hyperparameters of School of Reward
# Hacks (batch size 2, learning-rate multiplier 2, 3 epochs):
#   srh  the School of Reward Hacks dataset (user prompt -> reward-hacking response)
#   rrh  the `reward_hacks` split of Jozdien/realistic_reward_hacks
# The job is submitted, not awaited; add its finished ft: id to checkpoints.json under the
# dataset's label to run the eval scripts on it.
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from common import BASE_MODEL, OUTPUT_ROOT

from rewardhacking_training.train.train_providers.openai.sft import submit_sft_job
from rewardhacking_training.train.train_providers.openai.utils import (
    get_client, save_job_info, upload_training_file,
)

SEED = 42
N_EPOCHS = 3
BATCH_SIZE = 2
LR_MULTIPLIER = 2.0
SUFFIX = {"srh": "srh-sft", "rrh": "rrh-sft"}
OUT_DIR = OUTPUT_ROOT / "sft_baselines"


def srh_examples() -> list[dict]:
    from datasets import load_dataset

    rows = [
        {"messages": [
            {"role": "user", "content": row["user"]},
            {"role": "assistant", "content": row["school_of_reward_hacks"]},
        ]}
        for row in load_dataset("longtermrisk/school-of-reward-hacks", split="train")
    ]
    random.Random(SEED).shuffle(rows)
    return rows


def rrh_examples() -> list[dict]:
    from datasets import load_dataset

    return [
        {"messages": row["messages"]}
        for row in load_dataset("Jozdien/realistic_reward_hacks", split="reward_hacks")
    ]


DATASETS = {"srh": srh_examples, "rrh": rrh_examples}


def build_sft_file(dataset: str, out_dir: Path) -> Path:
    path = out_dir / "sft_data.jsonl"
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = DATASETS[dataset]()
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"[{dataset}] wrote {len(rows)} examples to {path}")
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(DATASETS))
    ap.add_argument("--build-only", action="store_true",
                    help="write the training file without submitting a job")
    ap.add_argument("--force", action="store_true", help="submit even if a job was already submitted")
    args = ap.parse_args()

    out_dir = OUT_DIR / args.dataset
    sft_path = build_sft_file(args.dataset, out_dir)
    if args.build_only:
        return
    info_path = out_dir / "job_info.json"
    if info_path.exists() and not args.force:
        sys.exit(f"job already submitted ({json.loads(info_path.read_text())['id']}); --force to resubmit")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set")
    client = get_client()
    info = submit_sft_job(
        client, upload_training_file(client, sft_path), BASE_MODEL,
        n_epochs=N_EPOCHS, batch_size=BATCH_SIZE, learning_rate_multiplier=LR_MULTIPLIER,
        suffix=SUFFIX[args.dataset],
    )
    save_job_info(info, out_dir)
    print(
        f"\nJob {info['id']} submitted; this script does not wait for it. Check with:\n"
        f"  python -c \"import openai; print(openai.OpenAI().fine_tuning.jobs.retrieve('{info['id']}').fine_tuned_model)\"\n"
        f"then add it to experiments/gpt_4_1_iter_dpo/checkpoints.json as \"{args.dataset}\""
    )


if __name__ == "__main__":
    main()
