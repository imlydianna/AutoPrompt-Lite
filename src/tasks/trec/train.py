import os
import logging
import gc
import torch
import requests  # <--- New import for direct download
from dataclasses import dataclass

from src.core.custom_components import RobustXMLParser
import adalflow.optim.text_grad.tgd_optimizer as tgd_mod
tgd_mod.CustomizedXMLParser = RobustXMLParser 

import adalflow as adal
from adalflow.optim.text_grad.tgd_optimizer import TGDOptimizer

from src.core.client import LocalLLMClient
from src.tasks.trec.config import *
from src.tasks.trec.task import TRECStudent
from src.tasks.trec.pipeline import TRECTrainingPipeline

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)

# 1. Define Data Structure
@dataclass
class SimpleTrecData:
    id: str
    question: str
    class_name: str

# 2. Robust Loader (No HuggingFace/Login required)
def load_raw_trec_from_url(split, size):
    """
    Downloads raw TREC data directly from Stanford NLP repository.
    Format per line: "LOC:dist How far is it..."
    We extract the coarse label (LOC) and the question.
    """
    # Map 'train'/'val' to the filenames used in the repo
    file_split = "train" if split == "train" else "test"
    url = f"https://raw.githubusercontent.com/course-nlp/datasets/master/data/trec/{file_split}.txt"
    
    print(f"📥 Downloading raw data from: {url}")
    try:
        r = requests.get(url)
        r.raise_for_status()
        lines = r.text.strip().split('\n')
    except Exception as e:
        print(f"❌ Failed to download data: {e}")
        return []

    data_list = []
    print(f"🔹 Parsing {len(lines)} lines...")
    
    for idx, line in enumerate(lines):
        if idx >= size: break
        
        # Line format: "LABEL:fine_label Question text..."
        # Example: "LOC:dist How far is it from Denver to Aspen ?"
        parts = line.split(' ', 1)
        if len(parts) < 2: continue
        
        label_part = parts[0] # e.g., "LOC:dist"
        question = parts[1].strip()
        
        # We only need the coarse label (before the colon)
        # e.g., "LOC" from "LOC:dist"
        coarse_label = label_part.split(':')[0]
        
        # Filter to ensure it's one of the standard 6 labels
        if coarse_label in ["ABBR", "DESC", "ENTY", "HUM", "LOC", "NUM"]:
            data_list.append(SimpleTrecData(
                id=str(idx),
                question=question,
                class_name=coarse_label
            ))
            
    print(f"✅ Loaded {len(data_list)} samples for {split}.")
    return data_list

def run_training():
    # Memory Cleanup
    print("🧹 Cleaning GPU Memory...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("🚀 INITIALIZING TREC EXPERIMENT (Direct Download Mode)...")
    
    # Initialize Clients
    print(f"🔹 Loading Teacher: {TEACHER_MODEL_NAME}")
    teacher_client = LocalLLMClient(model_name=TEACHER_MODEL_NAME)
    
    print(f"🔹 Loading Student: {STUDENT_MODEL_NAME}")
    student_client = LocalLLMClient(model_name=STUDENT_MODEL_NAME)

    # Initialize Components
    student_task = TRECStudent(student_client, STUDENT_MODEL_KWARGS)
    pipeline = TRECTrainingPipeline(student_task, teacher_client, TEACHER_MODEL_KWARGS)

    # Optimizer
    optimizer = TGDOptimizer(
        params=student_task.parameters(),
        model_client=teacher_client,
        model_kwargs=TEACHER_MODEL_KWARGS
    )

    # --- Load Data directly from URL ---
    print("📚 Loading Data...")
    train_data = load_raw_trec_from_url(split="train", size=TRAIN_SIZE)
    val_data = load_raw_trec_from_url(split="test", size=VAL_SIZE)

    # Setup Trainer
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
    
    # Save Results
    print("💾 Saving Optimized Prompts...")
    with open(os.path.join(OUTPUT_DIR, "optimized_instruction.txt"), "w") as f:
        f.write(student_task.instruction.data)
    with open(os.path.join(OUTPUT_DIR, "optimized_demos.txt"), "w") as f:
        f.write(student_task.demos.data)
    
    print(f"✅ DONE! Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    run_training()