import re
import os
import adalflow as adal
from adalflow.core.types import GeneratorOutput

# -----------------------------------------------------------------------------
# OUTPUT PARSER
# -----------------------------------------------------------------------------
@adal.func_to_data_component
def parse_trec_answer(output) -> str:
    """
    Robust parser to extract the classification label from the model's output.
    Expected format: "Answer: LABEL" or "Class: LABEL".
    """
    text = output.data if hasattr(output, "data") else str(output)
    if not text: return ""

    # Clean whitespace
    text = text.strip()
    
    # Strategy 1: Look for explicit "Answer: LABEL" pattern
    match = re.search(r"Answer:\s*([A-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Strategy 2: Look for "Class: LABEL" pattern
    match = re.search(r"Class:\s*([A-Z]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback Strategy: If the model returned just the label or a sentence ending in the label.
    valid_labels = {"ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"}
    words = text.split()
    
    # Iterate backwards to find the last valid label in the text
    for w in reversed(words):
        clean_w = w.strip().upper().replace(".", "")
        if clean_w in valid_labels:
            return clean_w
            
    return "UNKNOWN"

# -----------------------------------------------------------------------------
# STUDENT COMPONENT
# -----------------------------------------------------------------------------
class TRECStudent(adal.Component):
    def __init__(self, student_client, model_kwargs):
        super().__init__()

        def load_prompt_file(filename, default):
            """Helper to load initial prompts from .txt files or return default."""
            path = os.path.join(os.path.dirname(__file__), "prompts", filename)
            try:
                with open(path, "r", encoding="utf-8") as f: return f.read().strip()
            except: return default

        # --- PEER NODE 1: INSTRUCTION ---
        # The core task definition.
        self.instruction = adal.Parameter(
            data=load_prompt_file("instruction.txt", 
                "Classify the question into one of these categories: ABBR (Abbreviation), DESC (Description), ENTY (Entity), HUM (Human), LOC (Location), NUM (Number)."),
            role_desc="Defines the classification task and list of 6 valid labels.",
            instruction_to_optimizer="Ensure the model clearly understands the definitions and distinctions between the 6 categories.",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="instruction"
        )

        # --- PEER NODE 2: DEMOS (Few-Shot) ---
        # Examples to guide the model.
        self.demos = adal.Parameter(
            data=load_prompt_file("demos.txt", ""),
            role_desc="Few-shot examples mapping Question -> Answer.",
            instruction_to_optimizer="Add diverse examples covering tricky edge cases/failures found in validation.",
            requires_opt=True,
            param_type=adal.ParameterType.PROMPT,
            name="demos"
        )

        # --- PEER NODE 3: OUTPUT FORMAT ---
        # Strict formatting rules (usually not trained, but visible to optimizer).
        self.output_format = adal.Parameter(
            data=load_prompt_file("output_format.txt", "Finish your answer with exactly: 'Answer: LABEL'."),
            role_desc="Strict output format constraints.",
            requires_opt=False,
            name="output_format"
        )

        # Initialize the Generator with the compound template
        self.generator = adal.Generator(
            model_client=student_client,
            model_kwargs=model_kwargs,
            template="""<START_OF_SYSTEM_PROMPT>
<INSTRUCTION>
{{instruction}}
</INSTRUCTION>
<FORMAT>
{{output_format}}
</FORMAT>
<EXAMPLES>
{{demos}}
</EXAMPLES>
<END_OF_SYSTEM_PROMPT>

<START_OF_USER>
Question: {{input_str}}
<END_OF_USER>""",
            output_processors=parse_trec_answer,
            use_cache=False
        )

    def call(self, question, id=None):
        """Executes the forward pass."""
        return self.generator(
            prompt_kwargs={
                "instruction": self.instruction.data,
                "demos": self.demos.data,
                "output_format": self.output_format.data,
                "input_str": question
            },
            id=id
        )