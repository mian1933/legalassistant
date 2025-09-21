import readline
import subprocess
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# --- Configuration ---
LLM_MODEL_PATH = "/home/sa/Legal-Assistant/model"
RETRIEVER_SCRIPT_PATH = "retriever_service.py"

# Fix for terminal backspace key
readline.parse_and_bind("^H: backward-delete-char")


def retrieve_context_from_service(query_text: str) -> str:
    """Retrieves context by calling the external retriever service script."""
    command = [sys.executable, RETRIEVER_SCRIPT_PATH, "--query", query_text]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding='utf-8'
        )
        return result.stdout.strip()
    except FileNotFoundError:
        return f"Error: Retriever script '{RETRIEVER_SCRIPT_PATH}' not found."
    except subprocess.CalledProcessError as e:
        return f"Error: Retriever service execution failed.\nError message:\n{e.stderr}"


def main():
    # --- LLM Model Initialization ---
    print("=" * 20 + " Main Assistant System Initialization Start " + "=" * 20)
    print(f"[1/2] Loading LLM Tokenizer: {LLM_MODEL_PATH} ...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_PATH, trust_remote_code=True)

    if llm_tokenizer.chat_template is None:
        print("[INFO] Chat template not found. Manually setting Qwen-1.5 chat template.")
        qwen_chat_template = (
            "{% for message in messages %}"
            "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{'<|im_start|>assistant\n'}}"
            "{% endif %}"
        )
        llm_tokenizer.chat_template = qwen_chat_template

    try:
        stop_token_id = llm_tokenizer.convert_tokens_to_ids('<|im_end|>')
        print(f"[INFO] Found correct stop token ID for <|im_end|>: {stop_token_id}")
    except KeyError:
        print("[ERROR] Tokenizer cannot find the '<|im_end|>' token. The model may not be a Qwen chat model.")
        return

    print(f"[2/2] Loading LLM Model: {LLM_MODEL_PATH} ...")
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True
    )
    print("=" * 20 + " ✅ Main Assistant System Initialization Complete " + "=" * 20 + "\n")

    # --- Conversation Loop ---
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ['exit', 'quit']:
                print("Assistant: Goodbye!")
                break

            retrieved_context = retrieve_context_from_service(user_input)

            system_prompt = "You are a helpful and knowledgeable AI assistant. Answer the user's question. Prioritize using the 'Reference Materials' provided below if they are relevant. If the materials are not relevant or do not provide a sufficient answer, use your general knowledge to respond."
            prompt_with_context = f"--- Reference Materials ---\n{retrieved_context}\n\n--- User Question ---\n{user_input}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_with_context}
            ]
            rag_prompt_text = llm_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # The debug print block for the full prompt has been commented out.
            # print("\n" + "=" * 20 + " FULL PROMPT TO LLM (Formatted by Tokenizer) " + "=" * 20)
            # print(rag_prompt_text)
            # print("=" * 70 + "\n")

            print("[RAG] Generating response...")
            inputs = llm_tokenizer(rag_prompt_text, return_tensors="pt").to(llm_model.device)
            input_token_length = inputs.input_ids.shape[1]

            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=1024,
                eos_token_id=stop_token_id,

            )

            newly_generated_tokens = outputs[0, input_token_length:]
            final_answer = llm_tokenizer.decode(newly_generated_tokens, skip_special_tokens=True)

            print(f"Assistant: {final_answer}")

        except KeyboardInterrupt:
            print("\nAssistant: Conversation terminated.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            break


if __name__ == "__main__":
    main()