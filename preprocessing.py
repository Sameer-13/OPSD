"""
Convert your raw Saudi legal cases JSON into a HuggingFace dataset for OPSD
training.

Output columns:
    - problem : facts + relevant laws  (student sees this)
    - solution: gold step-by-step reasoning (teacher's privileged context)
    - verdict : gold verdict summary + classification (teacher's privileged target)

The teacher prompt assembles these three pieces with explicit structure so the
model can clearly distinguish the reasoning chain from the final ruling.
"""

import json
from datasets import Dataset


INPUT_FILE = "data/dataset_steps_verdict_classification.json"
OUTPUT_DIR = "./legal_opsd_data"


def to_opsd_format(case):
    # ---- Student-visible: facts + laws ----
    facts = "\n".join(case["facts"])
    laws = "\n".join(case["laws"])
    problem = f"Facts:\n{facts}\n\nRelevant laws:\n{laws}"

    # ---- Teacher privileged piece 1: the step-by-step reasoning ----
    reasoning = case["steps_texts"]
    if isinstance(reasoning, list):
        reasoning = "\n".join(reasoning)
    solution = f"<REASONING>\n{reasoning}\n</REASONING>"

    # ---- Teacher privileged piece 2: the verdict + classification ----
    verdict_summary = case["verdict_summary"]
    classification = case.get("verdict_classification", "").strip()
    verdict = (
        f"<VERDICT>\n{verdict_summary}\n</VERDICT>\n\n"
        f"<VERDICT_CLASSIFICATION>\n{classification}\n</VERDICT_CLASSIFICATION>"
    )

    return {"problem": problem, "solution": solution, "verdict": verdict}


def main():
    with open(INPUT_FILE) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases from {INPUT_FILE}")

    formatted = []
    for i, case in enumerate(cases):
        try:
            formatted.append(to_opsd_format(case))
        except KeyError as e:
            print(f"Skipping case {i}: missing key {e}")
            continue

    ds = Dataset.from_list(formatted)
    ds.save_to_disk(OUTPUT_DIR)
    print(f"Saved {len(ds)} examples to {OUTPUT_DIR}")
    print("\nFirst example (truncated):")
    ex = ds[0]
    for k, v in ex.items():
        print(f"  {k}: {v[:200]}{'...' if len(v) > 200 else ''}")


if __name__ == "__main__":
    main()