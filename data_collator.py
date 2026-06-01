import inspect
import torch


# System prompt shared by student and teacher. Both must produce output in the
# same three-tag format so that the per-token divergence is computed over a
# distribution the student is actually expected to match at inference time.
SYSTEM_PROMPT = """You are a judge expert in Saudi law. Your task is to produce the reasoning, verdict, and classification of a legal case from Saudi Arabia involving trade, finance, and commercial laws.

You will be given a set of facts and laws from the case, and you MUST provide THREE sections:
1. A reasoning section analyzing the facts
2. A verdict prediction section stating a summary of what you think the court will decide
3. A classification of the verdict

Guidelines:
- Provide a verdict based on the facts and the Saudi laws
- Be assertive and clear in your verdict, as if you were a judge yourself
- Base the verdict only on the facts provided without personal opinions or biases
- Your verdict and reasoning should be strictly in English
- The reasoning should be detailed and step-by-step, each step separated by a period
- The verdict should be short and direct

Follow this exact format:

<REASONING>
Your detailed reasoning and analysis here. Each step of your reasoning should be separated by a period.
</REASONING>

<VERDICT>
Your clear and direct verdict statement summary here.
</VERDICT>

<VERDICT_CLASSIFICATION>
PLAINTIFF | DEFENDANT | DISMISSED | SETTLEMENT
</VERDICT_CLASSIFICATION>

Classification options (choose exactly one):
- PLAINTIFF: Court ruled in favor of the plaintiff
- DEFENDANT: Court ruled in favor of the defendant
- DISMISSED: No ruling issued (dismissed, lack of jurisdiction, procedural rejection)
- SETTLEMENT: Parties reached settlement/reconciliation

Do not output anything outside these three sections. Each section must have an opening and closing tag exactly as shown above."""


class SelfDistillationDataCollator:
    """
    Data collator for OPSD self-distillation on Saudi legal cases.

    Student: sees only the facts + relevant laws (no reference reasoning).
    Teacher: sees the facts + relevant laws + the reference reasoning/verdict
             as privileged context, then is asked to derive its own answer.

    To enable batch-level operations, prompts are padded to the same length
    within each batch, and the actual (unpadded) prompt lengths are tracked
    for accurate loss masking.

    Expected dataset columns:
        - "problem":  string containing the facts and laws (built upstream
                      from your case["facts"] and case["laws"]).
        - "solution": string containing the gold reasoning + verdict +
                      classification in the same three-tag format the model
                      is expected to produce.
    """

    def __init__(
        self,
        tokenizer,
        max_length=16000,
        reason_first=False,
        student_thinking=False,
        teacher_thinking=True,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.reason_first = reason_first
        self.student_thinking = student_thinking
        self.teacher_thinking = teacher_thinking

        # Detect whether the tokenizer's chat template accepts `enable_thinking`
        # (Qwen3 does; Gemma and most other model families do not). We avoid
        # passing the kwarg if it's not supported, otherwise the template call
        # raises.
        try:
            sig = inspect.signature(self.tokenizer.apply_chat_template)
            self._supports_thinking_kwarg = "enable_thinking" in sig.parameters
        except (TypeError, ValueError):
            self._supports_thinking_kwarg = False

        # Prompt for the teacher's "rationalize first" mode (reason_first=True).
        # In this mode, the teacher first explains the reference reasoning,
        # then is asked to produce its own answer.
        self.reason_first_prompt = (
            "\n\nThe reference reasoning above arrives at the correct verdict. "
            "Please analyze this reasoning and explain the key legal logic and "
            "argumentative strategies employed by the court. "
            "Do NOT derive your own verdict yet. "
            "Simply analyze and explain the reference reasoning provided above.\n"
        )

        # Prompt used to transition the teacher from seeing the reference
        # reasoning/verdict to producing its own answer in the required format.
        self.transition_prompt = (
            "\n\nAfter reading the reference reasoning and verdict above, make "
            "sure you truly understand the legal logic behind each step — do "
            "not copy or paraphrase it. Now, using your own words and "
            "independent legal analysis, produce the reasoning, verdict, and "
            "classification for the case above in the required three-tag format.\n"
        )

        # Force right padding for consistency with the trainer's slicing.
        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Chat template supports enable_thinking kwarg: {self._supports_thinking_kwarg}")

    def _apply_chat_template(self, messages, enable_thinking=None):
        """Wrapper around apply_chat_template that only passes enable_thinking
        when the underlying template supports it."""
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._supports_thinking_kwarg and enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __call__(self, features):
        batch_size = len(features)

        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # only used when reason_first=True

        for feature in features:
            # `problem`  = facts + relevant laws (assembled upstream)
            # `solution` = gold reasoning + verdict + classification
            problem = feature["problem"]
            solution = feature["solution"]

            # ---- Student prompt: only facts + laws ----
            student_user_message = f"{problem}\n\nBegin!"
            student_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": student_user_message},
            ]
            student_prompt = self._apply_chat_template(
                student_messages, enable_thinking=self.student_thinking
            )
            student_prompts.append(student_prompt)

            if self.reason_first:
                # Teacher first rationalizes the reference reasoning out loud,
                # THEN the transition prompt + own-answer prompt is appended.
                reasoning_user_message = (
                    f"{problem}\n\n"
                    f"Here is the correct reasoning and verdict for this case:\n"
                    f"=== Reference Reasoning Begin ===\n"
                    f"{solution}\n"
                    f"=== Reference Reasoning End ===\n\n"
                    f"{self.reason_first_prompt}"
                )
                reasoning_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": reasoning_user_message},
                ]
                # No enable_thinking here — reasoning-phase generation should
                # always be plain (matches the original repo behavior).
                reasoning_prompt = self._apply_chat_template(reasoning_messages)
                teacher_reasoning_prompts.append(reasoning_prompt)

                # Placeholder — the actual teacher prompt is built inside the
                # trainer after the rationalization tokens have been generated.
                teacher_prompts.append("")
            else:
                # Standard OPSD teacher prompt: facts + laws + reference
                # reasoning/verdict + transition to producing its own answer.
                teacher_user_message = (
                    f"{problem}\n\n"
                    f"Here is a reference reasoning and verdict for this case:\n"
                    f"=== Reference Begin ===\n{solution}\n=== Reference End ===\n"
                    f"{self.transition_prompt}\n"
                    f"Begin!"
                )
                teacher_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": teacher_user_message},
                ]
                teacher_prompt = self._apply_chat_template(
                    teacher_messages, enable_thinking=self.teacher_thinking
                )
                teacher_prompts.append(teacher_prompt)

        # ---- Tokenize student prompts ----
        # First pass without padding to recover true per-example lengths.
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)

        # Second pass with padding to the batch max for stackable tensors.
        student_encoded = self.tokenizer(
            student_prompts,
            padding="max_length",
            truncation=True,
            max_length=max_student_prompt_len,
            return_tensors="pt",
        )

        result = {
            "student_prompts": student_encoded["input_ids"],
            "student_prompt_attention_mask": student_encoded["attention_mask"],
            "student_prompt_length": max_student_prompt_len,
            "student_prompt_lengths_per_example": torch.tensor(student_prompt_lengths),
        }

        if self.reason_first:
            # ---- Tokenize the reasoning-phase teacher prompts ----
            reasoning_encoded_no_pad = self.tokenizer(
                teacher_reasoning_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            reasoning_prompt_lengths = [len(ids) for ids in reasoning_encoded_no_pad["input_ids"]]
            max_reasoning_prompt_len = max(reasoning_prompt_lengths)

            reasoning_encoded = self.tokenizer(
                teacher_reasoning_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_reasoning_prompt_len,
                return_tensors="pt",
            )

            # Transition text gets appended after the teacher's rationalization
            # tokens, then the model is asked to produce its own answer.
            transition_text = (
                f"\n{self.transition_prompt}\n"
                f"Now produce the reasoning, verdict, and classification in "
                f"the required three-tag format.\nBegin!"
            )
            transition_encoded = self.tokenizer(
                [transition_text] * batch_size,
                padding=False,
                truncation=False,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_reasoning_prompts": reasoning_encoded["input_ids"],
                    "teacher_reasoning_attention_mask": reasoning_encoded["attention_mask"],
                    "teacher_reasoning_prompt_length": max_reasoning_prompt_len,
                    "teacher_transition_tokens": transition_encoded["input_ids"],
                }
            )
        else:
            # ---- Tokenize the standard teacher prompts ----
            teacher_encoded_no_pad = self.tokenizer(
                teacher_prompts,
                padding=False,
                truncation=True,
                max_length=self.max_length,
            )
            teacher_prompt_lengths = [len(ids) for ids in teacher_encoded_no_pad["input_ids"]]
            max_teacher_prompt_len = max(teacher_prompt_lengths)

            teacher_encoded = self.tokenizer(
                teacher_prompts,
                padding="max_length",
                truncation=True,
                max_length=max_teacher_prompt_len,
                return_tensors="pt",
            )

            result.update(
                {
                    "teacher_prompts": teacher_encoded["input_ids"],
                    "teacher_prompt_attention_mask": teacher_encoded["attention_mask"],
                    "teacher_prompt_length": max_teacher_prompt_len,
                    "teacher_prompt_lengths_per_example": torch.tensor(teacher_prompt_lengths),
                }
            )

        return result