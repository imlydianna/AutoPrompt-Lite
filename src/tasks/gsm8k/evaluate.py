"""
GSM8K Evaluation Script.

Compares the performance of the Baseline Prompt vs. the Optimized Prompt
on the held-out TEST set.
"""
import logging
import os
import adalflow as adal
from adalflow.datasets.gsm8k import GSM8K
from adalflow.eval.answer_match_acc import AnswerMatchAcc
from tqdm import tqdm

from src.core.client import LocalLLMClient
from src.tasks.gsm8k.config import (
    STUDENT_MODEL_NAME, STUDENT_MODEL_KWARGS, 
    TEST_SIZE, SEED
)
from src.tasks.gsm8k.task import GSM8KStudent

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def evaluate_prompt(client, dataset, prompt_text, run_name="Run"):
    """
    Runs inference on the dataset using a specific system prompt.
    Returns the accuracy score.
    """
    print(f"\n📊 EVALUATING: {run_name}")
    print(f"ℹ️  Set Size: {len(dataset)}")
    
    # Instantiate task with the specific prompt
    task = GSM8KStudent(student_client=client, model_kwargs=STUDENT_MODEL_KWARGS)
    task.system_prompt.data = prompt_text
    
    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    
    correct = 0
    total = len(dataset)

    for sample in tqdm(dataset, desc=f"Inference {run_name}"):
        try:
            # Forward Pass
            output = task.call(sample.question)
            
            # Extract Data
            parsed_answer = output.data
            ground_truth = sample.answer
            
            # Evaluate
            score = eval_fn(parsed_answer, ground_truth)
            correct += int(score)
        except Exception as e:
            print(f"Error on sample: {e}")

    accuracy = correct / total
    print(f"📈 Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy

def run_evaluation():
    print(f"🚀 Initializing Evaluation Client...")
    client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)

    # -------------------------------------------------------------------------
    # DATA LOADING (TEST SPLIT)
    # -------------------------------------------------------------------------
    print(f"📚 Loading TEST Dataset...")
    # We use the strict 'test' split for final reporting
    test_data = GSM8K(split="test", size=TEST_SIZE)

    # -------------------------------------------------------------------------
    # 1. BASELINE EVALUATION (Default Prompt)
    # -------------------------------------------------------------------------
    # Get the default prompt defined in task.py
    temp_task = GSM8KStudent(client, STUDENT_MODEL_KWARGS)
    baseline_prompt = temp_task.system_prompt.data
    
    acc_baseline = evaluate_prompt(client, test_data, baseline_prompt, run_name="Baseline Prompt")

    # -------------------------------------------------------------------------
    # 2. OPTIMIZED EVALUATION (Trained Prompt)
    # -------------------------------------------------------------------------
    optimized_prompt_file = "outputs/gsm8k/optimized_prompt.txt"
    
    if os.path.exists(optimized_prompt_file):
        with open(optimized_prompt_file, "r", encoding="utf-8") as f:
            optimized_prompt = f.read()
        
        acc_optimized = evaluate_prompt(client, test_data, optimized_prompt, run_name="Optimized Prompt")
        
        # ---------------------------------------------------------------------
        # REPORT
        # ---------------------------------------------------------------------
        print("\n" + "█"*50)
        print("               RESULTS SUMMARY               ")
        print("█"*50)
        print(f"Test Set Size:      {TEST_SIZE}")
        print(f"Baseline Accuracy:  {acc_baseline:.2%}")
        print(f"Optimized Accuracy: {acc_optimized:.2%}")
        diff = acc_optimized - acc_baseline
        print(f"Improvement:        {'+' if diff >= 0 else ''}{diff:.2%}")
        print("█"*50)
    else:
        print(f"\n⚠️ Optimized prompt not found at '{optimized_prompt_file}'. Run train.py first!")

if __name__ == "__main__":
    run_evaluation()