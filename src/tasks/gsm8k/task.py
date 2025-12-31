"""
GSM8K Task Definition.

This module defines:
1. The output parser (`parse_gsm8k_answer`) which extracts numbers from text.
2. The Student Component (`GSM8KStudent`) which holds the trainable system prompt
   and the generator logic.
"""

import re
from typing import Any, Dict, Optional

import adalflow as adal
from adalflow.core.types import GeneratorOutput

# -----------------------------------------------------------------------------
# OUTPUT PARSER
# -----------------------------------------------------------------------------
@adal.func_to_data_component
def parse_gsm8k_answer(output: Any) -> str:
    """
    Robust parser optimized for English Math Datasets (GSM8K).
    
    It extracts the final numerical answer from the model's output, handling:
    - Case-insensitive 'Answer:' markers.
    - Thousands separators (e.g., 1,000).
    - Trailing punctuation (e.g., 16.).
    - AdalFlow GeneratorOutput objects or raw strings.

    Args:
        output (Any): The raw output from the generator (string or GeneratorOutput).

    Returns:
        str: The extracted number as a string, or empty string if not found.
    """
    # 1. Safe Text Extraction
    # Handle cases where AdalFlow passes a wrapper object or a raw string
    text = ""
    if hasattr(output, "data"):
        text = output.data if output.data else ""
    else:
        text = str(output)

    if not text:
        return ""

    # 2. Isolate the Final Answer
    # We use re.split with IGNORECASE to split by "Answer:", "answer:", "ANSWER:", etc.
    # This returns a list. We take the LAST element (parts[-1]) to ensure we get
    # the final conclusion, ignoring intermediate reasoning steps that might mention "Answer".
    # If "Answer:" is not found, it returns the whole string (allowing fallback search).
    parts = re.split(r"Answer:", text, flags=re.IGNORECASE)
    candidate = parts[-1]

    # 3. Clean Commas (Standardize Number Format)
    # GSM8K uses commas as thousand separators (e.g., 600,000).
    # Regex logic usually breaks at commas, so we remove them to treat '600000' as a single token.
    clean_candidate = candidate.replace(",", "")

    # 4. Extract Number via Regex
    # Matches: 
    #  - Optional negative sign (-?)
    #  - One or more digits (\d+)
    #  - Optional decimal part (\.?\d*)
    match = re.search(r"(-?\d+\.?\d*)", clean_candidate)

    if match:
        result = match.group(1)
        # 5. Remove trailing dot
        # If the model output "Answer: 16.", the regex captures "16.".
        # We strip the dot to ensure exact match with Ground Truth "16".
        return result.rstrip(".")

    return ""

# -----------------------------------------------------------------------------
# STUDENT COMPONENT
# -----------------------------------------------------------------------------
class GSM8KStudent(adal.Component):
    """
    The Student Component responsible for solving math problems.
    
    This component wraps the LLM generator and holds the 'system_prompt' 
    as a trainable Parameter. This allows the AdalFlow optimizer to update 
    the instructions based on feedback.
    """
    def __init__(self, student_client: adal.ModelClient, model_kwargs: Dict):
        """
        Initialize the Student.

        Args:
            student_client (ModelClient): The backend client (e.g., LocalLLMClient).
            model_kwargs (Dict): Generation parameters (temp, max_tokens, etc.).
        """
        super().__init__()

        # The System Prompt is the trainable parameter.
        # We set requires_opt=True to tell AdalFlow this text should be optimized.
        self.system_prompt = adal.Parameter(
            data="You are a helpful math assistant. Solve the problem step by step. Finish your answer with exactly: 'Answer: X' where X is the number.",
            role_desc="Math Instructions",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="system_prompt"
        )

        # Initialize Generator
        # We pass the wrapped 'parse_gsm8k_answer' component here.
        # The template explicitly combines the system prompt and the user input
        # into a single string for the client.
        self.generator = adal.Generator(
            model_client=student_client,
            model_kwargs=model_kwargs,
            template="{{system_prompt}}\n\n{{input_str}}",
            output_processors=parse_gsm8k_answer,
            use_cache=False
        )

    def call(self, question: str, id: str = None) -> GeneratorOutput:
        """
        Forward pass: Generates an answer for a given math question.
        """
        return self.generator(
            prompt_kwargs={
                # Pass the current data of the trainable parameter
                "system_prompt": self.system_prompt.data, 
                "input_str": question
            },
            id=id
        )