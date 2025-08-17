import re
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
import os

load_dotenv(".env")
token=os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained("leotod/gemma-270m-Text2SQL-Fine-tuned", token=token)
print(tokenizer.chat_template)
model = AutoModelForCausalLM.from_pretrained("leotod/gemma-270m-Text2SQL-Fine-tuned", token=token)
stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

user_prompt = """Given the <USER_QUERY> and the <SCHEMA>, generate the corresponding SQL command to retrieve the desired data, considering the query's syntax, semantics, and schema constraints.

<SCHEMA>
{context}
</SCHEMA>

<USER_QUERY>
{question}
</USER_QUERY>
"""

system_message="You are a helpful assistant skilled in translating natural language into SQL Queries given the Table DDL. You will respond only with the correct MySQL Query."

def create_conversation(sql_prompt: str, sql_context: str = ""):
  return {
    "messages": [
      {"role": "system", "content": system_message},
      {"role": "user", "content": user_prompt.format(question=sql_prompt, context=sql_context)},
    ]
  }

def predict(conversation: list[dict]):
    prompt = pipe.tokenizer.apply_chat_template(conversation["messages"], tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=40, do_sample=False, temperature=0.0, top_k=50, top_p=0.1, eos_token_id=stop_token_ids, disable_compile=True)
    
    return outputs[0]['generated_text'][len(prompt):].strip()


if __name__ == '__main__':
    from datasets import load_dataset
    from dotenv import load_dotenv

    dataset = load_dataset("gretelai/synthetic_text_to_sql", token=os.getenv("HF_TOKEN"))
    rand_idx = 3
    test_sample = dataset["test"][rand_idx]

    sql_prompt = test_sample["sql_prompt"]
    sql_context = test_sample["sql_context"]
    
    print(predict(create_conversation(sql_prompt, sql_context)))
