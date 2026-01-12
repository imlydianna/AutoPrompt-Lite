import os
import logging
import gc
import torch
from dataclasses import dataclass
from datasets import load_dataset 

from src.core.custom_components import RobustXMLParser
# Patch the AdalFlow optimizer with our robust XML parser to handle local LLM outputs
import adalflow.optim.text_grad.tgd_optimizer as tgd_mod
tgd_mod.CustomizedXMLParser = RobustXMLParser 

import adalflow as adal
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer

from src.core.client import LocalLLMClient
from src.tasks.trec.config import *
from src.tasks.trec.task import TRECStudent
from src.tasks.trec.pipeline import TRECTrainingPipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("datasets").setLevel(logging.ERROR)

@dataclass
class SimpleTrecData:
    id: str
    question: str
    class_name: str

# function to load SetFit/trec
def load_safe_trec(split, size):
    print(f"📥 Loading TREC dataset (SetFit/trec) for split: {split}...")
    ds = load_dataset("SetFit/trec", split=split)
    
    data_list = []
    for idx, item in enumerate(ds):
        if idx >= size: break # 100 samples

        # The SetFit/trec dataset has a 'label_text' field which is e.g. "HUM", "LOC"
        data_list.append(SimpleTrecData(
            id=str(idx),
            question=item['text'],
            class_name=item['label_text'] 
        ))
    
    return data_list



def run_training():
    print("🧹 Cleaning GPU Memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    train_data = load_safe_trec(split="train", size=TRAIN_SIZE)
    val_data = load_safe_trec(split="test", size=VAL_SIZE) 

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