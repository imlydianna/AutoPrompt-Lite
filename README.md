# Promptimus Prime: Modular Textual Gradient Descent Framework 🚛🤖

**Promptimus Prime** is a comprehensive, extensible framework designed to reproduce and expand upon the **LLM-AutoDiff** methodology for Automatic Prompt Engineering (APE).

It implements **Textual Gradient Descent (TGD)**, a novel optimization paradigm that treats natural language prompts as trainable parameters. Instead of manual tuning, the system uses a "Teacher" LLM to backpropagate textual feedback (gradients) to a "Student" LLM, systematically improving performance across various downstream tasks.

The project is built on **AdalFlow** and **Hugging Face Transformers**, emphasizing code modularity, reproducibility, and efficiency on consumer hardware (e.g., Google Colab T4 GPU).

---

## 🚀 The Framework: Auto-Differentiating LLM Workflows

The core innovation of this repository is a generalized optimization loop that can be applied to any LLM-based task. The framework abstracts the complexity of:

1.  **Textual Backpropagation:** Generating "gradients" (critiques) that explain the gap between a prediction and the ground truth.
2.  **Peer Node Optimization:** Optimizing distinct parts of a prompt (e.g., Instructions vs. Few-Shot Examples) independently to avoid the "lost-in-the-middle" phenomenon.
3.  **Student-Teacher Architectures:** Leveraging a strong, quantized model (e.g., 7B) to optimize a lightweight model (e.g., 1.5B).

---

## 📚 Supported Tasks & Benchmarks

The framework is designed to be task-agnostic. Currently implemented tasks include:

### 1. 🧮 GSM8K (Grade School Math)
A benchmark for multi-step mathematical reasoning.
*   **Goal:** Improve Chain-of-Thought (CoT) reasoning.
*   **Student Model:** `Qwen2.5-1.5B-Instruct` (4-bit quantized).
*   **Teacher Model:** `Qwen2.5-7B-Instruct` (4-bit quantized).
*   **Strategy:** Optimizes both the core System Instruction and Few-Shot Demonstrations using Textual Gradients.

*(More tasks, such as Object Counting and Vanilla RAG, are planned for future updates.)*

---

## 📂 Project Structure

The repository follows a strict separation of concerns between core infrastructure and task-specific logic.

-   **`notebooks/`**: Interactive demos.
    -   **`gsm8k_demo.ipynb`**: An end-to-end walkthrough of the GSM8K pipeline (Training -> Eval -> Viz) running in Colab.
-   **`outputs/`**: Generated artifacts (checkpoints, optimized prompts, visualization figures).
-   **`src/`**: The source code library.
    -   **`core/`**: **Task-Agnostic Infrastructure.**
        -   **`client.py`**: A robust `LocalLLMClient` wrapper for Hugging Face that handles 4-bit quantization, chat templating, and XML output parsing for *any* Instruct model.
    -   **`tasks/`**: **Domain-Specific Logic.**
        -   **`gsm8k/`**: Implementation of the Math Reasoning task.
            -   `config.py`: Hyperparameters and dataset splits.
            -   `pipeline.py`: Orchestration of the Student-Teacher loop.
            -   `train.py`: The optimization loop entry point.
            -   `evaluate.py`: Rigorous testing on held-out data.
        -   *(Future tasks will be added here as separate modules)*

---

## 🚀 Getting Started & How to Run

You can execute the pipelines either interactively via Jupyter Notebooks or as standalone Python scripts for automation.

### Prerequisites

-   Python 3.10+
-   GPU with at least 12GB VRAM (e.g., NVIDIA T4, L4).

### Option A: Interactive Notebooks (Recommended for Exploration)

Use the provided notebooks to run experiments in an interactive environment (like Google Colab). The notebooks import logic directly from `src/`, ensuring that progress bars (`tqdm`) and logging work perfectly.

*   **GSM8K:** Open `notebooks/gsm8k_demo.ipynb`.

### Option B: Standalone Scripts (Recommended for Production)

You can run any task module directly from the command line.

**1. Installation**
```bash
git clone https://github.com/antonisbaro/promptimus-prime
cd promptimus-prime
pip install -r requirements.txt
```

**2. Running a Task (Example: GSM8K)**

*   **Training:** Loads models, runs the optimization loop, and saves the best prompt to `outputs/`.
    ```bash
    python -m src.tasks.gsm8k.train
    ```

*   **Evaluation:** Compares the Baseline prompt against the Optimized artifact on a held-out Test set.
    ```bash
    python -m src.tasks.gsm8k.evaluate
    ```

---

## 🧠 Algorithmic Strategy: Textual Gradient Descent (TGD)

The optimization process mimics numeric gradient descent but operates in the space of natural language:

1.  **Forward Pass:** The Student model generates a response based on the current prompt parameters.
2.  **Evaluation:** A metric (e.g., Exact Match accuracy) scores the response.
3.  **Backward Pass:** If the score is low, the Teacher model acts as a "Backward Engine," analyzing the Student's reasoning trace to generate a textual critique (gradient).
4.  **Update:** The Optimizer LLM proposes a semantic update to the prompt (e.g., "Add a step to verify calculation") to reduce the error.

---

## 💻 Technology Stack

-   **Framework:** [AdalFlow](https://github.com/SylphAI-Inc/AdalFlow) (The PyTorch for LLM Pipelines).
-   **Models:** [Qwen2.5](https://huggingface.co/Qwen) (via Hugging Face Transformers).
-   **Optimization:** Textual Gradient Descent (TGD).
-   **Efficiency:** `bitsandbytes` (NF4) quantization.
-   **Visualization:** `pandas`, `matplotlib`, `difflib`.