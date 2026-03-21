import argparse
import json
from pathlib import Path

from demo_vlm_runner import load_dataset_sample, run_dataset_sample
from evaluator.detail_evaluate import compute_SR_object_state


STABLE_ACTION_FAMILIES = {
    "find",
    "pick",
    "put",
    "open",
    "close",
    "drop",
    "slice",
    "turn on",
    "turn off",
    "fillLiquid",
}


def extract_action_family(action):
    action = (action or "").strip()
    for prefix in [
        "turn on",
        "turn off",
        "turn_on",
        "turn_off",
        "fillLiquid",
        "emptyLiquid",
        "find",
        "pick",
        "put",
        "open",
        "close",
        "slice",
        "drop",
        "throw",
        "break",
        "cook",
        "dirty",
        "clean",
        "pour",
    ]:
        if action.lower().startswith(prefix.lower()):
            if prefix == "turn_on":
                return "turn on"
            if prefix == "turn_off":
                return "turn off"
            return prefix
    return action.split()[0] if action else ""


def parse_sample_indices(dataset_path, sample_indices=None, start_index=None, end_index=None, max_samples=None):
    if sample_indices:
        indices = [int(x.strip()) for x in sample_indices.split(",") if x.strip()]
    else:
        start = start_index or 1
        end = end_index
        if end is None:
            count = 0
            with Path(dataset_path).open() as f:
                for count, _ in enumerate(f, start=1):
                    pass
            end = count
        indices = list(range(start, end + 1))

    if max_samples is not None:
        indices = indices[:max_samples]
    return indices


def summarize_execution(execution_log):
    return [step.get("success", False) for step in execution_log]


def build_result_record(dataset_path, sample_index, max_steps):
    sample = load_dataset_sample(dataset_path, sample_index)
    reference_steps = sample.get("step", [])
    reference_families = [extract_action_family(step) for step in reference_steps]
    unsupported_reference_actions = sorted(
        {family for family in reference_families if family not in STABLE_ACTION_FAMILIES}
    )

    try:
        result = run_dataset_sample(dataset_path, sample_index, max_steps=max_steps)
        object_state_success = None
        object_state_avg = None
        if sample.get("final_state") is not None:
            object_state_success, object_state_avg = compute_SR_object_state(
                result["final_metadata"]["objects"],
                sample["final_state"],
            )

        result_type = result["result_type"]
        if result_type == "failed" and unsupported_reference_actions:
            result_type = "unsupported"

        return {
            "sample_index": sample_index,
            "idx": sample_index,
            "dataset_path": str(dataset_path),
            "scene_name": sample["scene_name"],
            "scene": sample["scene_name"],
            "instruction": sample["instruction"],
            "result_type": result_type,
            "final_status": result_type,
            "predicted_actions": result["history"],
            "execution_summary": summarize_execution(result["execution_log"]),
            "final_state_success": result["final_state_success"],
            "pre_satisfied": result["pre_satisfied"],
            "reference_steps": reference_steps,
            "reference_action_families": reference_families,
            "unsupported_reference_actions": unsupported_reference_actions,
            "object_state_success": object_state_success,
            "object_state_avg_success": object_state_avg,
            "error": result.get("error"),
            "fail_type": result.get("fail_type"),
            "fail_step": result.get("fail_step"),
            "raw_vlm_output": result["raw_vlm_outputs"][-1] if result.get("raw_vlm_outputs") else None,
            "controller_error": result.get("controller_error"),
            "evaluator_status": "object_state_only",
        }
    except Exception as exc:
        return {
            "sample_index": sample_index,
            "idx": sample_index,
            "dataset_path": str(dataset_path),
            "scene_name": sample["scene_name"],
            "scene": sample["scene_name"],
            "instruction": sample["instruction"],
            "result_type": "runner_error",
            "final_status": "runner_error",
            "predicted_actions": [],
            "execution_summary": [],
            "final_state_success": False,
            "pre_satisfied": False,
            "reference_steps": reference_steps,
            "reference_action_families": reference_families,
            "unsupported_reference_actions": unsupported_reference_actions,
            "object_state_success": None,
            "object_state_avg_success": None,
            "error": str(exc),
            "fail_type": "runner_exception",
            "fail_step": None,
            "raw_vlm_output": None,
            "controller_error": None,
            "evaluator_status": "object_state_only",
        }


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_completed_indices(path):
    path = Path(path)
    if not path.exists():
        return set()
    completed = set()
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            idx = record.get("sample_index") or record.get("idx")
            if idx is not None:
                completed.add(int(idx))
    return completed


def summarize_records(records):
    summary = {
        "total": len(records),
        "solved": 0,
        "pre_satisfied": 0,
        "unsupported": 0,
        "failed": 0,
        "infra_failure": 0,
        "runner_error": 0,
        "failed_indices": [],
    }
    for record in records:
        result_type = record["result_type"]
        if result_type in summary:
            summary[result_type] += 1
        if result_type == "failed":
            summary["failed_indices"].append(record["sample_index"])
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Batch benchmark subset runner for VLM-planned SafeAgentBench")
    parser.add_argument("--dataset", required=True, help="Path to dataset jsonl")
    parser.add_argument("--sample-indices", default=None, help="Comma-separated 1-based sample indices")
    parser.add_argument("--start-index", type=int, default=None, help="1-based inclusive start index")
    parser.add_argument("--end-index", type=int, default=None, help="1-based inclusive end index")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap after index selection")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--output-path", required=True, help="JSONL output file path")
    parser.add_argument("--summary-path", default=None, help="Optional JSON summary path")
    parser.add_argument("--batch-size", type=int, default=10, help="Checkpoint summary every N samples")
    parser.add_argument("--resume", action="store_true", help="Skip samples already present in output-path")
    return parser.parse_args()


def main():
    args = parse_args()
    indices = parse_sample_indices(
        dataset_path=args.dataset,
        sample_indices=args.sample_indices,
        start_index=args.start_index,
        end_index=args.end_index,
        max_samples=args.max_samples,
    )
    output_path = Path(args.output_path)
    if output_path.exists() and not args.resume:
        output_path.unlink()
    completed = load_completed_indices(output_path) if args.resume else set()
    records = []
    if args.resume:
        for line in output_path.open() if output_path.exists() else []:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    pending = [idx for idx in indices if idx not in completed]
    for count, sample_index in enumerate(pending, start=1):
        record = build_result_record(args.dataset, sample_index, args.max_steps)
        append_jsonl(output_path, record)
        records.append(record)
        print(
            f"idx={record['sample_index']} | instruction={record['instruction']} | "
            f"final_status={record['final_status']} | fail_type={record.get('fail_type')} | fail_step={record.get('fail_step')}"
        )
        if args.summary_path and count % args.batch_size == 0:
            summary = summarize_records(records)
            Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    summary = summarize_records(records)
    if args.summary_path:
        Path(args.summary_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
