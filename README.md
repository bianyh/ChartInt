# ChartInt: Beyond Static Charts: Benchmarking Interactive Chart-to-Code Generation for Multimodal Large Language Models

**ChartInt** is the first benchmark specifically designed to evaluate **Interactive Chart-to-Code Generation** capabilities of Multimodal Large Language Models (MLLMs).

While existing benchmarks primarily focus on the visual consistency of static charts, ChartInt emphasizes **Behavioral Consistency**. It requires models to generate code that not only reconstructs the visual appearance but also correctly supports dynamic user interactions (e.g., hovering, highlighting, tooltips).

This repository contains the official implementation code and evaluation scripts for ChartInt.

## 🚀 News

* **[2026-02]** Code and dataset released.
* **[2026-02]** Paper is under submission.

## 📂 Dataset

The ChartInt dataset comprises **2,779** carefully annotated examples, covering diverse interaction patterns and realistic editing scenarios.

* **Download Link**: [Google Drive](https://drive.google.com/file/d/1Qg1jRuq5s8YNVazsZgWZ9c5LBjGQhWpY/view?usp=drive_link)


## 💡 Key Features

### 1. Preset State Snapshot Protocol

To evaluate dynamic interactions, we propose a deterministic evaluation mechanism. This protocol programmatically triggers interaction events (e.g., via ECharts `dispatchAction`) and captures the dynamic behaviors as reproducible static snapshots for evaluation.

### 2. Six Core Tasks

ChartInt includes tasks ranging from basic reconstruction to high-level maintenance, comprehensively evaluating the model's lifecycle capabilities:

* **Static Reconstruction**
* 
**Native Reconstruction**: Fundamental visual-to-code reproduction.


* 
**Style-Perturbed Reconstruction**: Stress-testing robustness with non-default styles (e.g., gradients).




* **Interaction**
* 
**Interaction Task**: Generating executable code that supports specific interaction logic (e.g., hover highlighting).




* **Real-World Editable**
* 
**Soft-coded Data Update**: Refactoring hardcoded data to load from external CSV files.


* 
**Visual Style Update**: Modifying chart aesthetics (e.g., color, layout) based on instructions without altering data.


* 
**Cross-Chart Style Transfer**: Synthesizing a new chart by combining the data from Chart A with the visual style of Chart B.





## 📊 Evaluation

We employ a multi-dimensional evaluation framework:

1. 
**Code Executability**: Verifies if the generated code runs successfully.


2. **Fine-grained Visual Metrics**:
* OCR-based Text Consistency ( score).


* Color Feature Alignment (Color-IoU).




3. 
**MLLM-based Judgment**: Using GPT-4o to score Visual Logic, Data Content, and Style Consistency.



```

## 📧 Contact

For any questions, please contact me.
