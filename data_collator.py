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

    Student: sees only the facts + relevant laws (no reference reasoning, no verdict).
    Teacher: sees the facts + relevant laws + reference reasoning + the
             ground-truth verdict & classification as privileged context, then
             is asked to derive its own answer in the required format.

    Giving the teacher the verdict explicitly (in addition to the reasoning)
    makes OPSD closer to the paper's framing where `y*` is the ground-truth
    answer — the teacher rationalizes a known-correct outcome and uses that
    to provide dense supervision on the student's own rollout.

    Expected dataset columns:
        - "problem":  facts + relevant laws (built upstream)
        - "solution": gold step-by-step reasoning, wrapped in <REASONING>...</REASONING>
        - "verdict":  gold verdict summary + classification, wrapped in the
                      <VERDICT> and <VERDICT_CLASSIFICATION> tags
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
        # (Qwen3 does; Gemma and most other model families do not).
        try:
            sig = inspect.signature(self.tokenizer.apply_chat_template)
            self._supports_thinking_kwarg = "enable_thinking" in sig.parameters
        except (TypeError, ValueError):
            self._supports_thinking_kwarg = False

        # Teacher's rationalization preamble when reason_first=True.
        self.reason_first_prompt = (
            "\n\nThe reference reasoning and verdict above represent the correct "
            "outcome for this case. Please analyze the reasoning and explain the "
            "key legal logic that led to this verdict. "
            "Do NOT derive your own verdict yet. "
            "Simply analyze and explain the reference materials provided above.\n"
        )

        # Transition prompt that hands the teacher from privileged context to
        # generating its own answer.
        self.transition_prompt = (
            "\n\nAfter reading the reference reasoning and the ground-truth "
            "verdict above, make sure you truly understand the legal logic "
            "behind the ruling — do not copy or paraphrase it. Now, using your "
            "own words and independent legal analysis, produce the reasoning, "
            "verdict, and classification for the case above in the required "
            "three-tag format. Your verdict and classification must agree with "
            "the ground-truth outcome above.\n"
        )

        print(f"[DataCollator] Original padding_side: {self.tokenizer.padding_side}")
        self.tokenizer.padding_side = "right"
        print(f"[DataCollator] Set padding_side to: {self.tokenizer.padding_side}")
        print(f"[DataCollator] Reason first mode: {self.reason_first}")
        print(f"[DataCollator] Chat template supports enable_thinking kwarg: {self._supports_thinking_kwarg}")

    def _apply_chat_template(self, messages, enable_thinking=None):
        """Apply chat template, passing enable_thinking only when supported."""
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if self._supports_thinking_kwarg and enable_thinking is not None:
            kwargs["enable_thinking"] = enable_thinking
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    def _build_privileged_block(self, solution, verdict):
        """Assemble the privileged context block the teacher sees.

        Lays out the gold reasoning and the gold verdict as two clearly
        separated sub-blocks so the model treats them as distinct pieces of
        privileged information.
        """
        return (
            f"Here is the reference reasoning for this case:\n"
            f"=== Reference Reasoning Begin ===\n"
            f"{solution}\n"
            f"=== Reference Reasoning End ===\n\n"
            f"Here is the ground-truth verdict and classification for this case:\n"
            f"=== Ground-Truth Verdict Begin ===\n"
            f"{verdict}\n"
            f"=== Ground-Truth Verdict End ==="
        )

    def __call__(self, features):
        batch_size = len(features)

        student_prompts = []
        teacher_prompts = []
        teacher_reasoning_prompts = []  # only used when reason_first=True

        for feature in features:
            problem = feature["problem"]    # facts + laws
            solution = feature["solution"]  # gold reasoning (in <REASONING> tags)
            verdict = feature["verdict"]    # gold verdict + classification

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

            # ---- Teacher prompt: facts + laws + privileged context ----
            privileged_block = self._build_privileged_block(solution, verdict)

            if self.reason_first:
                # Teacher first rationalizes the privileged materials out loud.
                reasoning_user_message = (
                    f"{problem}\n\n{privileged_block}\n\n{self.reason_first_prompt}"
                )
                reasoning_messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": reasoning_user_message},
                ]
                reasoning_prompt = self._apply_chat_template(reasoning_messages)
                teacher_reasoning_prompts.append(reasoning_prompt)
                teacher_prompts.append("")  # placeholder, built later in trainer
            else:
                teacher_user_message = (
                    f"{problem}\n\n"
                    f"{privileged_block}\n"
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
        student_encoded_no_pad = self.tokenizer(
            student_prompts,
            padding=False,
            truncation=True,
            max_length=self.max_length,
        )
        student_prompt_lengths = [len(ids) for ids in student_encoded_no_pad["input_ids"]]
        max_student_prompt_len = max(student_prompt_lengths)

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