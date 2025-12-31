"""
GSM8K Training Script.

This script executes the Text-Grad optimization loop. It:
1. Initializes the Student and Teacher models using the custom LocalLLMClient.
2. Loads the GSM8K dataset and performs a deterministic split (Train/Val).
3. Configures the AdalFlow Trainer with the TGD Optimizer.
4. Runs the training process.
5. Saves the optimized system prompt to a file for later evaluation.

Usage:
    Run this script from the project root:
    python -m src.tasks.gsm8k.train
"""

import logging
import os
import adalflow as adal
from adalflow.datasets.gsm8k import GSM8K
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer

# Import Core Infrastructure
from src.core.client import LocalLLMClient

# Import Task-Specific Logic & Configuration
from src.tasks.gsm8k.config import (
    STUDENT_MODEL_NAME, TEACHER_MODEL_NAME, 
    STUDENT_MODEL_KWARGS, TEACHER_MODEL_KWARGS,
    TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, MAX_STEPS, 
    TRAIN_SIZE, VAL_SIZE
)
from src.tasks.gsm8k.task import GSM8KStudent
from src.tasks.gsm8k.pipeline import GSM8KTrainingPipeline

# Configure Logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_training():
    """
    Main function to setup and run the optimization experiment.
    """
    # -------------------------------------------------------------------------
    # 1. INITIALIZE MODELS
    # -------------------------------------------------------------------------
    # These clients handle 4-bit loading and strict chat templating.
    print("👨‍🎓 Initializing Student Client...")
    student_client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)
    print("👨‍🏫 Initializing Teacher Client...")
    teacher_client = LocalLLMClient(model_name=TEACHER_MODEL_NAME)

    # -------------------------------------------------------------------------
    # 2. SETUP COMPONENTS
    # -------------------------------------------------------------------------
    print(f"🧮 Initializing Student Task...")
    student_task = GSM8KStudent(
        student_client=student_client,
        model_kwargs=STUDENT_MODEL_KWARGS
    )
    
    # Capture initial state for comparison
    initial_prompt = student_task.system_prompt.data

    print(f"🛠️  Building Training Pipeline...")
    pipeline = GSM8KTrainingPipeline(
        student_task=student_task,
        teacher_client=teacher_client,
        teacher_model_kwargs=TEACHER_MODEL_KWARGS
    )

    print(f"🧠 Setting up Optimizer...")
    optimizer = TGDOptimizer(
        params=student_task.parameters(), # Target the 'system_prompt'
        model_client=teacher_client,      # The Teacher generates the updates
        model_kwargs=TEACHER_MODEL_KWARGS
    )

    # -------------------------------------------------------------------------
    # 3. DATA LOADING 
    # -------------------------------------------------------------------------
    print(f"📚 Loading Datasets...")
    
    # Load strict Train split
    train_data = GSM8K(split="train", size=TRAIN_SIZE)
    
    # Load strict Val split
    val_data = GSM8K(split="val", size=VAL_SIZE)

    print(f"📊 Splits Loaded:")
    print(f"   - Train Set: {len(train_data)} samples")
    print(f"   - Val Set:   {len(val_data)} samples")

    # -------------------------------------------------------------------------
    # 4. TRAINER SETUP & EXECUTION
    # -------------------------------------------------------------------------
    trainer = adal.Trainer(
        adaltask=pipeline,
        optimizer=optimizer,
        strategy="random", 
        max_steps=MAX_STEPS,       
        batch_size=VAL_BATCH_SIZE,
        train_batch_size=TRAIN_BATCH_SIZE
    )

    print("\n🏁 STARTING TRAINING...")
    print(f"📜 INITIAL PROMPT:\n{initial_prompt}\n")
    
    # Start the optimization loop.
    # This modifies student_task.system_prompt in-place.
    trainer.fit(
        train_dataset=train_data, 
        val_dataset=val_data, 
        debug=False
    )

    # -------------------------------------------------------------------------
    # 5. SAVE ARTIFACTS
    # -------------------------------------------------------------------------
    final_prompt = student_task.system_prompt.data
    print("\n✅ TRAINING COMPLETE!")
    print(f"📜 FINAL OPTIMIZED PROMPT:\n{final_prompt}")

    # Ensure output directory exists
    output_dir = "outputs/gsm8k"
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "optimized_prompt.txt")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(final_prompt)
    
    print(f"\n💾 Saved optimized prompt to: {output_file}")

if __name__ == "__main__":
    run_training()