import os
import logging
from src.core.custom_components import RobustXMLParser
# Patch the AdalFlow optimizer with our robust XML parser to handle local LLM outputs
import adalflow.optim.text_grad.tgd_optimizer as tgd_mod
tgd_mod.CustomizedXMLParser = RobustXMLParser 

import adalflow as adal
from adalflow.datasets.trec import TrecDataset
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer

from src.core.client import LocalLLMClient
from src.tasks.trec.config import *
from src.tasks.trec.task import TRECStudent
from src.tasks.trec.pipeline import TRECTrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

def run_training():
    print("🚀 INITIALIZING TREC EXPERIMENT (LocalLLM)...")
    
    # 1. Initialize Clients
    student_client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)
    teacher_client = LocalLLMClient(model_name=TEACHER_MODEL_NAME)

    # 2. Initialize Components
    student_task = TRECStudent(student_client, STUDENT_MODEL_KWARGS)
    pipeline = TRECTrainingPipeline(student_task, teacher_client, TEACHER_MODEL_KWARGS)

    # 3. Initialize Optimizer (Textual Gradient Descent)
    optimizer = TGDOptimizer(
        params=student_task.parameters(),
        model_client=teacher_client,
        model_kwargs=TEACHER_MODEL_KWARGS
    )

    # 4. Load & Slice Dataset
    print("📚 Loading TREC Dataset...")
    train_data = TrecDataset(split="train")
    val_data = TrecDataset(split="val") 
    
    # Slice datasets to speed up execution in Colab environment
    train_data = train_data[:TRAIN_SIZE]
    val_data = val_data[:VAL_SIZE]

    # 5. Setup Trainer
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer = adal.Trainer(
        adaltask=pipeline,
        optimizer=optimizer,
        max_steps=MAX_STEPS,
        train_batch_size=TRAIN_BATCH_SIZE,
        ckpt_path=CKPT_DIR
    )

    print("🏁 STARTING TRAINING...")
    trainer.fit(train_dataset=train_data, val_dataset=val_data)
    
    # 6. Save Artifacts
    print("💾 Saving Optimized Prompts...")
    with open(os.path.join(OUTPUT_DIR, "optimized_instruction.txt"), "w") as f:
        f.write(student_task.instruction.data)
    with open(os.path.join(OUTPUT_DIR, "optimized_demos.txt"), "w") as f:
        f.write(student_task.demos.data)
    
    print(f"✅ Training Complete! Results saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    run_training()