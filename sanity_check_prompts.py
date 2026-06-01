"""
Sanity-check the rendered student and teacher prompts before launching OPSD
training with the new three-column schema (problem / solution / verdict).

Run:
    python sanity_check_prompts.py
"""

from transformers import AutoTokenizer

from data_collator import SelfDistillationDataCollator, SYSTEM_PROMPT


MODEL_ID = "google/gemma-4-E4B-it"
MAX_LENGTH = 16000


fake_examples = [
    {
        "problem": (
            "Facts:\n"
            "The plaintiff entered into a sale contract with the defendant for the "
            "delivery of 500 tons of steel. The defendant failed to deliver on the "
            "agreed date, causing the plaintiff financial loss.\n\n"
            "Relevant laws:\n"
            "Saudi Commercial Court Law Article 27 — breach of contract remedies.\n"
            "Saudi Commercial Court Law Article 31 — damages for late performance."
        ),
        "solution": (
            "<REASONING>\n"
            "The contract is valid under Saudi commercial law. The defendant failed "
            "to perform on the agreed date. Late performance causes recoverable "
            "damages under Article 31.\n"
            "</REASONING>"
        ),
        "verdict": (
            "<VERDICT>\n"
            "The court rules in favor of the plaintiff and awards damages for late "
            "performance.\n"
            "</VERDICT>\n\n"
            "<VERDICT_CLASSIFICATION>\n"
            "PLAINTIFF\n"
            "</VERDICT_CLASSIFICATION>"
        ),
    },
    {
        "problem": (
            "Facts:\n"
            "The plaintiff filed a complaint requesting cancellation of a partnership "
            "agreement, but failed to attend two consecutive scheduled hearings without "
            "valid excuse.\n\n"
            "Relevant laws:\n"
            "Procedural Law Article 53 — dismissal for non-appearance."
        ),
        "solution": (
            "<REASONING>\n"
            "The plaintiff failed to appear at two consecutive hearings. Article 53 "
            "authorizes dismissal on these grounds.\n"
            "</REASONING>"
        ),
        "verdict": (
            "<VERDICT>\n"
            "The case is dismissed.\n"
            "</VERDICT>\n\n"
            "<VERDICT_CLASSIFICATION>\n"
            "DISMISSED\n"
            "</VERDICT_CLASSIFICATION>"
        ),
    },
]


def banner(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    banner(f"Loading tokenizer for {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Pad token: {tokenizer.pad_token!r}  (id={tokenizer.pad_token_id})")
    print(f"EOS token: {tokenizer.eos_token!r}  (id={tokenizer.eos_token_id})")
    print(f"BOS token: {tokenizer.bos_token!r}")

    banner("Instantiating SelfDistillationDataCollator")
    collator = SelfDistillationDataCollator(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=False,
    )

    banner("Building a batch from 2 fake examples")
    batch = collator(fake_examples)

    student_ids = batch["student_prompts"]
    teacher_ids = batch["teacher_prompts"]
    student_lens = batch["student_prompt_lengths_per_example"].tolist()
    teacher_lens = batch["teacher_prompt_lengths_per_example"].tolist()

    print(f"Student tensor shape: {tuple(student_ids.shape)}")
    print(f"Teacher tensor shape: {tuple(teacher_ids.shape)}")
    print(f"Per-example student prompt token lengths: {student_lens}")
    print(f"Per-example teacher prompt token lengths: {teacher_lens}")

    student_text = tokenizer.decode(student_ids[0], skip_special_tokens=False)
    teacher_text = tokenizer.decode(teacher_ids[0], skip_special_tokens=False)

    banner("RENDERED STUDENT PROMPT (example 0)")
    print(student_text)

    banner("RENDERED TEACHER PROMPT (example 0)")
    print(teacher_text)

    banner("AUTOMATIC CHECKS")

    gold_reasoning_snippet = "The contract is valid under Saudi commercial law"
    gold_verdict_snippet = "rules in favor of the plaintiff"
    gold_classification = "PLAINTIFF"
    system_snippet = "You are a judge expert in Saudi law"
    ref_reasoning_marker = "=== Reference Reasoning Begin ==="
    ref_verdict_marker = "=== Ground-Truth Verdict Begin ==="

    checks = [
        (
            "System prompt appears in student prompt",
            system_snippet in student_text,
        ),
        (
            "System prompt appears in teacher prompt",
            system_snippet in teacher_text,
        ),
        (
            "Student does NOT see gold reasoning",
            gold_reasoning_snippet not in student_text,
        ),
        (
            "Student does NOT see gold verdict text",
            gold_verdict_snippet not in student_text,
        ),
        (
            "Student does NOT see ground-truth verdict block",
            # The label string `PLAINTIFF` itself appears in the system prompt's
            # menu of options — that's fine. What we want is that the
            # verdict-block wrapping markers are absent from the student side.
            ref_verdict_marker not in student_text,
        ),
        (
            "Teacher DOES see gold reasoning",
            gold_reasoning_snippet in teacher_text,
        ),
        (
            "Teacher DOES see gold verdict text",
            gold_verdict_snippet in teacher_text,
        ),
        (
            "Teacher DOES see gold classification label inside the verdict block",
            gold_classification in teacher_text.split(ref_verdict_marker, 1)[-1],
        ),
        (
            "Teacher has reasoning markers",
            ref_reasoning_marker in teacher_text,
        ),
        (
            "Teacher has verdict markers",
            ref_verdict_marker in teacher_text,
        ),
        (
            "Student prompt does not contain reasoning markers",
            ref_reasoning_marker not in student_text,
        ),
    ]

    all_passed = True
    for label, passed in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        if not passed:
            all_passed = False

    banner("LENGTH SANITY")
    longest_teacher = max(teacher_lens)
    print(f"Longest teacher prompt in this batch: {longest_teacher} tokens")
    print(f"max_length budget: {MAX_LENGTH} tokens")
    headroom = MAX_LENGTH - longest_teacher
    print(f"Headroom for completion + slack: {headroom} tokens")

    banner("RESULT")
    if all_passed:
        print("All checks passed. Safe to launch training.")
    else:
        print("One or more checks FAILED. Fix before launching.")


if __name__ == "__main__":
    main()