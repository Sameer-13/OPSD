from datasets import Dataset
import json

with open("data/dataset_steps_verdict_classification.json") as f:
    cases = json.load(f)

def to_opsd_format(case):
    facts = "\n".join(case["facts"])
    laws = "\n".join(case["laws"])
    reasoning = "\n".join(case["reasoning"]) if isinstance(case["reasoning"], list) else case["reasoning"]
    verdict = case["verdict_summary"]

    problem = f"Facts:\n{facts}\n\nRelevant laws:\n{laws}"
    
    # solution = gold reasoning + verdict in the format you want the student to produce
    solution = (
        f"<REASONING>\n{reasoning}\n</REASONING>\n\n"
        f"<VERDICT>\n{verdict}\n</VERDICT>\n\n"
        f"<VERDICT_CLASSIFICATION>\n{case['verdict_classification']}\n</VERDICT_CLASSIFICATION>"
    )
    return {"problem": problem, "solution": solution}

formatted = [to_opsd_format(c) for c in cases]
ds = Dataset.from_list(formatted)
ds.save_to_disk("./legal_opsd_data")  # or ds.push_to_hub("your-username/legal-opsd")