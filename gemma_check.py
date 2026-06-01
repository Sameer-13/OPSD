"""
Sanity-check the rendered student and teacher prompts before launching OPSD training.

What this verifies:
  1. The Gemma (or other) chat template renders without errors.
  2. The SYSTEM_PROMPT actually appears in the rendered text (some templates
     silently drop system messages; Gemma folds them into the first user turn).
  3. The student prompt does NOT contain the gold reasoning/verdict.
  4. The teacher prompt DOES contain the gold reasoning/verdict between the
     === Reference Begin === / === Reference End === markers.
  5. Both prompts end with the generation-prompt token, so generation continues
     in the right format.
  6. Reports tokenized lengths so you can pick a sensible max_length.

Run:
    python sanity_check_prompts.py
"""

from transformers import AutoTokenizer
from data_collator import SelfDistillationDataCollator, SYSTEM_PROMPT


MODEL_ID = "google/gemma-4-E4B-it"  # change to whatever you're training
MAX_LENGTH = 16000


# ---- Two fake examples in the exact schema your preprocessing produces ----
# Replace these with real ones from your dataset for a more realistic check.
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
            "</REASONING>\n\n"
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
            "</REASONING>\n\n"
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
        teacher_thinking=False,  # ignored for Gemma anyway
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
    print(f"Padded student length (batch max): {batch['student_prompt_length']}")
    print(f"Padded teacher length (batch max): {batch['teacher_prompt_length']}")

    # ---- Decode example 0 of the batch ----
    student_text = tokenizer.decode(student_ids[0], skip_special_tokens=False)
    teacher_text = tokenizer.decode(teacher_ids[0], skip_special_tokens=False)

    banner("RENDERED STUDENT PROMPT (example 0)")
    print(student_text)

    banner("RENDERED TEACHER PROMPT (example 0)")
    print(teacher_text)

    # ---- Automatic checks ----
    banner("AUTOMATIC CHECKS")

    gold_reasoning_snippet = "The contract is valid under Saudi commercial law"
    gold_verdict_snippet = "rules in favor of the plaintiff"
    system_snippet = "You are a judge expert in Saudi law"
    ref_begin_marker = "=== Reference Begin ==="
    ref_end_marker = "=== Reference End ==="

    checks = [
        (
            "System prompt appears in student prompt",
            system_snippet in student_text,
            "If False: the chat template is dropping the system message. "
            "Some templates ignore system role entirely — you may need to fold "
            "SYSTEM_PROMPT into the user message manually.",
        ),
        (
            "System prompt appears in teacher prompt",
            system_snippet in teacher_text,
            "Same as above.",
        ),
        (
            "Student does NOT see gold reasoning",
            gold_reasoning_snippet not in student_text,
            "If False: privileged information is leaking into the student "
            "prompt — fix the collator before training.",
        ),
        (
            "Student does NOT see gold verdict",
            gold_verdict_snippet not in student_text,
            "Same as above.",
        ),
        (
            "Teacher DOES see gold reasoning",
            gold_reasoning_snippet in teacher_text,
            "If False: the teacher has no privileged information and OPSD "
            "degenerates to self-distillation with no signal.",
        ),
        (
            "Teacher has reference markers",
            ref_begin_marker in teacher_text and ref_end_marker in teacher_text,
            "If False: the reference solution wrapping markers were lost.",
        ),
        (
            "Student prompt does not contain reference markers",
            ref_begin_marker not in student_text,
            "If False: collator is mixing student and teacher prompts.",
        ),
    ]

    all_passed = True
    for label, passed, hint in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"  [{mark}] {label}")
        if not passed:
            print(f"         -> {hint}")
            all_passed = False

    # ---- Length sanity ----
    banner("LENGTH SANITY")
    longest_teacher = max(teacher_lens)
    print(f"Longest teacher prompt in this batch: {longest_teacher} tokens")
    print(f"max_length budget: {MAX_LENGTH} tokens")
    headroom = MAX_LENGTH - longest_teacher
    print(f"Headroom for completion + slack: {headroom} tokens")
    if headroom < 2048:
        print("WARNING: less than 2048 tokens of headroom — your real-data teacher "
              "prompts may collide with max_completion_length. Re-measure on "
              "real cases and consider raising --max_length.")

    banner("RESULT")
    if all_passed:
        print("All checks passed. Safe to launch training.")
    else:
        print("One or more checks FAILED. Fix before launching.")


if __name__ == "__main__":
    main()