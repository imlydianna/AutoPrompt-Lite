"""
GSM8K Evaluation Script.

This module performs a comparative evaluation between the Baseline (Zero-Shot) 
system prompt and the Optimized (Text-Grad) system prompt.

Key Features:
1. deterministic execution on the held-out TEST set.
2. Calculation of exact-match accuracy.
3. Export of a detailed CSV report ('comparison_results.csv') containing 
   reasoning traces, predictions, and ground truth for side-by-side analysis.
"""

import logging
import os
import pandas as pd
from typing import Tuple
from tqdm import tqdm

import adalflow as adal
from adalflow.datasets.gsm8k import GSM8K
from adalflow.eval.answer_match_acc import AnswerMatchAcc

from src.core.client import LocalLLMClient
from src.tasks.gsm8k.config import (
    STUDENT_MODEL_NAME, 
    STUDENT_MODEL_KWARGS, 
    TEST_SIZE
)
from src.tasks.gsm8k.task import GSM8KStudent

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def evaluate_prompt(client: LocalLLMClient, 
                    dataset: GSM8K, 
                    prompt_text: str, 
                    run_name: str = "Run") -> Tuple[float, pd.DataFrame]:
    """
    Runs inference on the dataset using a specific system prompt and collects detailed results.

    Args:
        client (LocalLLMClient): The LLM client wrapper.
        dataset (GSM8K): The dataset split to evaluate on.
        prompt_text (str): The system prompt to test.
        run_name (str): Label for this evaluation run (e.g., 'Baseline').

    Returns:
        Tuple[float, pd.DataFrame]: 
            - The accuracy score (0.0 to 1.0).
            - A DataFrame containing detailed logs for every sample.
    """
    print(f"\n📊 EVALUATING: {run_name}")
    print(f"ℹ️  Set Size: {len(dataset)}")
    
    # Initialize the Task Component with the specific prompt parameters
    task = GSM8KStudent(student_client=client, model_kwargs=STUDENT_MODEL_KWARGS)
    task.system_prompt.data = prompt_text
    
    # Initialize Metric
    eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
    
    correct_count = 0
    total_count = len(dataset)
    results = []

    # Iterate through the dataset
    for sample in tqdm(dataset, desc=f"Inference {run_name}"):
        try:
            # 1. Forward Pass
            output = task.call(sample.question)
            
            # 2. Extract Data
            parsed_answer = output.data
            ground_truth = sample.answer
            raw_reasoning = output.raw_response
            
            # 3. Evaluate
            score = eval_fn(parsed_answer, ground_truth)
            correct_count += int(score)
            
            # 4. Record Detailed Results
            results.append({
                "question": sample.question,
                "ground_truth": ground_truth,
                "prediction": parsed_answer,
                "reasoning": raw_reasoning,
                "is_correct": bool(score)
            })
            
        except Exception as e:
            logging.error(f"Error processing sample ID {sample.id}: {e}")
            # Append a failure record to keep DataFrame alignment
            results.append({
                "question": sample.question,
                "ground_truth": sample.answer,
                "prediction": "ERROR",
                "reasoning": str(e),
                "is_correct": False
            })

    # Calculate metrics
    accuracy = correct_count / total_count if total_count > 0 else 0.0
    print(f"📈 Accuracy: {accuracy:.2%} ({correct_count}/{total_count})")
    
    # Create DataFrame
    df_results = pd.DataFrame(results)
    
    return accuracy, df_results

def run_evaluation():
    """
    Main execution routine.
    1. Loads the Test Set.
    2. Evaluates the Baseline Prompt.
    3. Evaluates the Optimized Prompt (if available).
    4. Saves a combined CSV report for analysis.
    """
    print(f"🚀 Initializing Evaluation Client...")
    client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)

    # -------------------------------------------------------------------------
    # DATA LOADING (TEST SPLIT)
    # -------------------------------------------------------------------------
    print(f"📚 Loading TEST Dataset...")
    # We use the strict 'test' split for final reporting to ensure no data leakage.
    test_data = GSM8K(split="test", size=TEST_SIZE)

    # -------------------------------------------------------------------------
    # 1. BASELINE EVALUATION (Default Prompt)
    # -------------------------------------------------------------------------
    # Retrieve the default prompt from the Task definition
    temp_task = GSM8KStudent(client, STUDENT_MODEL_KWARGS)
    baseline_prompt = temp_task.system_prompt.data
    
    acc_baseline, df_baseline = evaluate_prompt(
        client, test_data, baseline_prompt, run_name="Baseline"
    )

    # -------------------------------------------------------------------------
    # 2. OPTIMIZED EVALUATION (Trained Prompt)
    # -------------------------------------------------------------------------
    optimized_prompt_file = "outputs/gsm8k/optimized_prompt.txt"
    
    if os.path.exists(optimized_prompt_file):
        with open(optimized_prompt_file, "r", encoding="utf-8") as f:
            optimized_prompt = f.read()
        
        acc_optimized, df_optimized = evaluate_prompt(
            client, test_data, optimized_prompt, run_name="Optimized"
        )
        
        # ---------------------------------------------------------------------
        # 3. SAVE COMPARISON REPORT
        # ---------------------------------------------------------------------
        # Prefix columns to distinguish between runs in the merged file
        df_baseline = df_baseline.add_prefix("base_")
        df_optimized = df_optimized.add_prefix("opt_")
        
        # Combine side-by-side (index alignment is guaranteed by deterministic loading)
        comparison_df = pd.concat([df_baseline, df_optimized], axis=1)
        
        output_csv = "outputs/gsm8k/comparison_results.csv"
        comparison_df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved detailed comparison report to: {output_csv}")

        # ---------------------------------------------------------------------
        # 4. FINAL SUMMARY
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
        print(f"\n⚠️ Optimized prompt not found at '{optimized_prompt_file}'.")
        print("   Please run 'train.py' first to generate the optimized artifact.")

if __name__ == "__main__":
    run_evaluation()