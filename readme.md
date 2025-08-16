# Gemma-270M Fine-Tuning for NL → SQL

This repository contains experiments on fine-tuning the **Gemma-270M** model for the **Text-to-SQL** task using the public dataset **`gretelai/synthetic_text_to_sql`**.

---

## Project Overview

- **Model**: Gemma-270M (fine-tuned version available at [leotod/gemma-270m-Text2SQL-Fine-tuned](https://huggingface.co/leotod/gemma-270m-Text2SQL-Fine-tuned))  
- **Task**: Natural Language → SQL Query generation  
- **Dataset**: `gretelai/synthetic_text_to_sql` (~100k pairs)  
- **Method**: Supervised Fine-Tuning using **QLoRA** (quantized LoRA, PEFT)

---

## Current Status

- Base fine-tuning with QLoRA on Gemma-270M  
- WandB integration for logging  
- Prompt-based SQL generation working  

---

## TODOs

- [ ] **Tokenizer**: switch to a plain tokenizer without chat templates  
- [ ] **Hyperparameter search**: test other SFT/QLoRA configurations (learning rate, batch size, scheduler)  
- [ ] **RL fine-tuning**: experiment with GRPO / DPO to align SQL outputs with execution correctness  
- [ ] **UI**: build a simple Gradio web interface for prototyping NL → SQL queries  
- [ ] **Containerization**: create a Docker environment for reproducibility and the full application.
- [ ] **Metrics**: develop evaluation
- [ ] **Validation**: implement proper evaluation/validation to monitor model generalization  

---

## Example Prompt

```text
Natural language request:
"Show me the names of the salespeople who sold more than 150 units of timber."

Database schema (DDL):
CREATE TABLE salesperson (
    salesperson_id INT,
    name TEXT,
    region TEXT
);

CREATE TABLE timber_sales (
    sales_id INT,
    salesperson_id INT,
    volume REAL,
    sale_date DATE
);

Write the corresponding SQL query:
```

Expected output:

```sql
SELECT s.name
FROM salesperson s
JOIN timber_sales t ON s.salesperson_id = t.salesperson_id
WHERE t.volume > 150;
```

---

## References

- QLoRA fine-tuning with Gemma: [Google AI documentation – Create and prepare the fine-tuning dataset](https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora?hl=it#create-and-prepare-the-fine-tuning-dataset)  
- [Hugging Face LLM Course, Chapter 1](https://huggingface.co/learn/llm-course/chapter1/1) — I am currently studying this course, and this project is part of the exercises I am doing when reading. In particular the things I am developing here belong to **Chapter 11: Fine Tune Large Language Models**.  
- Fine-tuned model: [leotod/gemma-270m-Text2SQL-Fine-tuned](https://huggingface.co/leotod/gemma-270m-Text2SQL-Fine-tuned)

---

## License

MIT