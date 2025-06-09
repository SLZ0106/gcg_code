# main_pipeline.py

import json
import argparse
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    read_paired_functions,
    parse_model_output,
    replace_variable_with_placeholder,
    run_gcg_on_function,
    apply_gcg_to_variable,
    save_analysis_results,
    parse_label_only
)
from sys_prompt import get_prompt_template, get_prompt_template_gcg

MAIN_PATH = '/home/luzesun/gcg_code_1'


def get_completion(prompt: str, model, tokenizer) -> str:
    """
    Generates a completion for the given prompt using the provided model and tokenizer.
    Returns only the newly generated text (excluding the original prompt).

    Args:
        prompt (str): The full prompt string to send to the model.
        model: The PEFT-wrapped causal LM.
        tokenizer: The corresponding tokenizer.

    Returns:
        str: Only the newly generated part (model’s analysis of the function).
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    print(f"[Token Length] {len(input_ids[0])} tokens")
    # Record length of input prompt
    input_len = input_ids.shape[-1]

    # Generate with low temperature, short max_new_tokens
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=32,
        do_sample=False,
        num_beams=1,
        temperature=0.0,
        repetition_penalty=1,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Only decode tokens after the prompt
    generated_ids = output_ids[0, input_len:].unsqueeze(0)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Read paired functions, separate benign/vulnerable, run vulnerability analysis, "
                    "apply GCG attack on identified variable, then re-analyze."
    )
    parser.add_argument(
        "--file_path",
        type=str,
        default=f'{MAIN_PATH}/dataset/primevul_train_paired.jsonl',
        help="Path to the JSONL file containing paired functions"
    )
    parser.add_argument(
        "--num_pairs",
        type=int,
        default=None,
        help="Number of pairs to read (default: all pairs)"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=f'{MAIN_PATH}/results/analysis_results.jsonl',
        help="Path to save the final JSONL results"
    )
    args = parser.parse_args()

    # Step 0: Read all paired functions
    paired_functions = read_paired_functions(args.file_path, args.num_pairs)

    # Step 1: Separate vulnerable and benign functions
    vulnerable_funcs = []
    benign_funcs = []
    for vuln_dict, benign_dict in paired_functions:
        vulnerable_funcs.append(vuln_dict)
        benign_funcs.append(benign_dict)

    # Load base causal LM in 4-bit or fp16 as available
    model = AutoModelForCausalLM.from_pretrained(
        "fdtn-ai/Foundation-Sec-8B",
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("fdtn-ai/Foundation-Sec-8B")

    # Fetch the prompt template (system + user)
    template = get_prompt_template()
    template_gcg = get_prompt_template_gcg()
    results = []

    total_pairs = len(paired_functions)
    vuln_correct_before = 0
    vuln_correct_after = 0
    benign_correct_before = 0
    benign_correct_after = 0

    # Step 3: Analyze each function, apply GCG attack on identified variable, then re-analyze
    for pair_idx, (vuln_entry, benign_entry) in enumerate(zip(vulnerable_funcs, benign_funcs)):
        # 3a) Initial analysis on Vulnerable function
        vuln_code = vuln_entry["func"]
        message_v = template.replace("{func}", vuln_code)
        raw_output_v = get_completion(message_v, model, tokenizer)
        print(f"raw_output_v: {raw_output_v}")
        label_v, var_v = parse_model_output(raw_output_v)
        print(f"label_v: {label_v}, var_v: {var_v}")
        # 3b) Initial analysis on Benign function
        benign_code = benign_entry["func"]
        message_b = template.replace("{func}", benign_code)
        raw_output_b = get_completion(message_b, model, tokenizer)
        print(f"raw_output_b: {raw_output_b}")
        label_b, var_b = parse_model_output(raw_output_b)
        print(f"label_b: {label_b}, var_b: {var_b}")

        # 3c) If a vulnerable variable was identified, apply GCG attack once
        gcg_message_b = template_gcg.replace("{func}", benign_code)
        gcg_message_v = template_gcg.replace("{func}", vuln_code)
        if label_v == "VULNERABLE":
            attacked_vuln_code = apply_gcg_to_variable(gcg_message_v, var_v, model, tokenizer, num_steps=200)
            print(f"attacked_vuln_code: {attacked_vuln_code}")
        if label_b == "VULNERABLE":
            attacked_benign_code = apply_gcg_to_variable(gcg_message_b, var_b, model, tokenizer, num_steps=200)


        # 3d) Re-analyze the attacked vulnerable function
        raw_output_v2 = get_completion(attacked_vuln_code, model, tokenizer)
        print(f"raw_output_v2: {raw_output_v2}")
        label_v2 = parse_label_only(raw_output_v2)
        raw_output_b2 = get_completion(attacked_benign_code, model, tokenizer)
        print(f"raw_output_b2 (benign): {raw_output_b2}")
        label_b2 = parse_label_only(raw_output_b2)

        # 3e) Record all fields in a single dictionary
        entry = {
            "pair_index": pair_idx,
            "vuln_idx": vuln_entry["idx"],
            "benign_idx": benign_entry["idx"],
            "initial_label_vuln": label_v,
            "initial_variable_vuln": var_v,
            "initial_label_benign": label_b,
            "initial_variable_benign": var_b,
            "attacked_vuln_code": attacked_vuln_code,
            "relabel_vuln_after_attack": label_v2,
            "attacked_benign_code": attacked_benign_code,
            "relabel_benign_after_attack": label_b2,
        }
        results.append(entry)

        if label_v == "VULNERABLE":
            vuln_correct_before += 1
        if label_v2 == "VULNERABLE":
            vuln_correct_after += 1

        if label_b == "BENIGN":
            benign_correct_before += 1
        if label_b2 == "BENIGN":
            benign_correct_after += 1

    # Step 4: Save all results to JSONL
    save_analysis_results(results, args.output_path)

    print("\n=== Summary Statistics ===")
    print(f"Vulnerable correct before attack: {vuln_correct_before}/{total_pairs} = {100.0 * vuln_correct_before / total_pairs:.2f}%")
    print(f"Vulnerable correct after  attack: {vuln_correct_after}/{total_pairs} = {100.0 * vuln_correct_after / total_pairs:.2f}%")
    print(f"Benign    correct before attack: {benign_correct_before}/{total_pairs} = {100.0 * benign_correct_before / total_pairs:.2f}%")
    print(f"Benign    correct after  attack: {benign_correct_after}/{total_pairs} = {100.0 * benign_correct_after / total_pairs:.2f}%")

    print(f"Analysis and GCG attack results saved to: {args.output_path}")
