"""
Measure token-length distribution of your gold `solution` strings and teacher
prompts so you can pick sensible values for:

  --max_completion_length  (covers the gold solution at inference time)
  --max_length             (covers teacher_prompt + max_completion_length)

Run:
    python measure_lengths.py
"""

import json
import numpy as np
from transformers import AutoTokenizer

MODEL_ID = "google/gemma-4-E4B-it"
DATA_PATH = "data/dataset_steps_verdict_classification.json"


def to_solution_string(case):
    """Match the format you use in your preprocessing script."""
    reasoning = case["reasoning"]
    if isinstance(reasoning, list):
        reasoning = "\n".join(reasoning)
    verdict = case["verdict_summary"]
    classification = case.get("verdict_classification", "")
    return (
        f"<REASONING>\n{reasoning}\n</REASONING>\n\n"
        f"<VERDICT>\n{verdict}\n</VERDICT>\n\n"
        f"<VERDICT_CLASSIFICATION>\n{classification}\n</VERDICT_CLASSIFICATION>"
    )


def to_problem_string(case):
    facts = "\n".join(case["facts"])
    laws = "\n".join(case["laws"])
    return f"Facts:\n{facts}\n\nRelevant laws:\n{laws}"


def percentiles(lengths, name):
    arr = np.array(lengths)
    print(f"\n{name} token lengths (n={len(arr)}):")
    print(f"  min    : {arr.min()}")
    print(f"  median : {int(np.percentile(arr, 50))}")
    print(f"  p90    : {int(np.percentile(arr, 90))}")
    print(f"  p95    : {int(np.percentile(arr, 95))}")
    print(f"  p99    : {int(np.percentile(arr, 99))}")
    print(f"  max    : {arr.max()}")


def main():
    print(f"Loading tokenizer: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    with open(DATA_PATH) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases")

    solution_lens = []
    problem_lens = []
    teacher_prompt_lens = []  # rough estimate: problem + solution + ~250 token overhead

    for c in cases:
        sol = to_solution_string(c)
        prob = to_problem_string(c)
        sol_len = len(tok(sol, add_special_tokens=False)["input_ids"])
        prob_len = len(tok(prob, add_special_tokens=False)["input_ids"])
        solution_lens.append(sol_len)
        problem_lens.append(prob_len)
        # ~250 tokens for system prompt + transition + chat template overhead
        teacher_prompt_lens.append(prob_len + sol_len + 250)

    percentiles(solution_lens, "Gold solution")
    percentiles(problem_lens, "Problem (facts + laws)")
    percentiles(teacher_prompt_lens, "Approx teacher prompt (problem + solution + overhead)")

    p95_sol = int(np.percentile(solution_lens, 95))
    p99_teacher = int(np.percentile(teacher_prompt_lens, 99))

    suggested_completion = int(np.ceil(p95_sol / 256) * 256)  # round up to nearest 256
    suggested_max_length = int(np.ceil((p99_teacher + suggested_completion + 512) / 1024) * 1024)

    print("\n" + "=" * 60)
    print("SUGGESTED LAUNCH FLAGS")
    print("=" * 60)
    print(f"  --max_completion_length {suggested_completion}")
    print(f"  --max_length            {suggested_max_length}")
    print("=" * 60)
    print("Reasoning: completion = p95(solution) rounded up to 256;")
    print("           max_length = p99(teacher_prompt) + completion + 512 slack,")
    print("           rounded up to 1024.")


if __name__ == "__main__":
    main()