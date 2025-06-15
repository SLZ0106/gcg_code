# main_pipeline.py - Complete resumable version with English comments

import json
import argparse
import os
import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import (
    read_paired_functions,
    parse_model_output,
    apply_gcg_to_variable,
    save_analysis_results,
    parse_label_only
)
from sys_prompt import get_prompt_template, get_prompt_template_gcg

MAIN_PATH = '/home/luzesun/gcg_code'


def get_completion(prompt: str, model, tokenizer) -> str:
    """
    Generates a completion for the given prompt using the provided model and tokenizer.
    Returns only the newly generated text (excluding the original prompt).

    Args:
        prompt (str): The full prompt string to send to the model.
        model: The PEFT-wrapped causal LM.
        tokenizer: The corresponding tokenizer.

    Returns:
        str: Only the newly generated part (model's analysis of the function).
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


def get_last_pair_index(output_path: str) -> int:
    """
    Get the last processed pair_index from the output file.
    
    Args:
        output_path: Path to the output JSONL file
        
    Returns:
        int: Last pair_index found in file, or -1 if file doesn't exist
    """
    if not os.path.exists(output_path):
        return -1
    
    last_idx = -1
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    pair_idx = entry.get("pair_index", -1)
                    last_idx = max(last_idx, pair_idx)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Warning: Error reading output file: {e}")
    
    return last_idx


def get_completed_indices(output_path: str) -> set:
    """
    Get all completed pair indices from the output file.
    
    Args:
        output_path: Path to the output JSONL file
        
    Returns:
        set: Set of completed pair indices
    """
    completed = set()
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    completed.add(entry.get("pair_index", -1))
                except json.JSONDecodeError:
                    continue
    return completed


def get_error_indices(output_path: str) -> list:
    """
    Get all error pair indices from the output file.
    
    Args:
        output_path: Path to the output JSONL file
        
    Returns:
        list: List of pair indices that had errors
    """
    error_indices = []
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if (entry.get("initial_label_vuln") == "ERROR" or 
                        entry.get("initial_label_benign") == "ERROR" or
                        "error" in entry or "error_message" in entry):
                        error_indices.append(entry["pair_index"])
                except json.JSONDecodeError:
                    continue
    return error_indices


def append_result(output_path: str, entry: dict):
    """
    Append a single result entry to the output file.
    
    Args:
        output_path: Path to the output JSONL file
        entry: Dictionary containing the result for one pair
    """
    with open(output_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def process_single_pair(pair_idx: int, vuln_entry: dict, benign_entry: dict, 
                       model, tokenizer, template: str, template_gcg: str) -> dict:
    """
    Process a single function pair.
    
    Args:
        pair_idx: Index of the current pair
        vuln_entry: Dictionary containing vulnerable function data
        benign_entry: Dictionary containing benign function data
        model: The loaded model
        tokenizer: The tokenizer
        template: Prompt template for initial analysis
        template_gcg: Prompt template for GCG
        
    Returns:
        dict: Result entry for this pair
    """
    # Initialize result entry with default values
    entry = {
        "pair_index": pair_idx,
        "vuln_idx": vuln_entry.get("idx", -1),
        "benign_idx": benign_entry.get("idx", -1),
        "initial_label_vuln": "UNKNOWN",
        "initial_variable_vuln": "",
        "initial_label_benign": "UNKNOWN",
        "initial_variable_benign": "",
        "attacked_vuln_code": "",
        "vuln_output_after_attack": "",
        "relabel_vuln_after_attack": "UNKNOWN",
        "attacked_benign_code": "",
        "benign_output_after_attack": "",
        "relabel_benign_after_attack": "UNKNOWN",
    }
    
    try:
        # Get function codes
        vuln_code = vuln_entry.get("func", "")
        benign_code = benign_entry.get("func", "")
        
        # Default to original code (in case GCG fails)
        entry["attacked_vuln_code"] = vuln_code
        entry["attacked_benign_code"] = benign_code
        
        # Analyze vulnerable function
        print("  Analyzing vulnerable function...")
        message_v = template.replace("{func}", vuln_code)
        raw_output_v = get_completion(message_v, model, tokenizer)
        print(f"  Vulnerable output: {raw_output_v}")
        label_v, var_v = parse_model_output(raw_output_v)
        print(f"  Vulnerable result: label={label_v}, variable={var_v}")
        
        entry["initial_label_vuln"] = label_v
        entry["initial_variable_vuln"] = var_v
        
        # Analyze benign function
        print("  Analyzing benign function...")
        message_b = template.replace("{func}", benign_code)
        raw_output_b = get_completion(message_b, model, tokenizer)
        print(f"  Benign output: {raw_output_b}")
        label_b, var_b = parse_model_output(raw_output_b)
        print(f"  Benign result: label={label_b}, variable={var_b}")
        
        entry["initial_label_benign"] = label_b
        entry["initial_variable_benign"] = var_b
        
        # GCG attack on vulnerable function if needed
        if label_v == "VULNERABLE" and var_v:
            try:
                print(f"  Applying GCG attack on vulnerable function (variable: {var_v})...")
                gcg_message_v = template.replace("{func}", vuln_code)
                attacked_vuln_code = apply_gcg_to_variable(gcg_message_v, var_v, model, tokenizer, num_steps=250)
                entry["attacked_vuln_code"] = attacked_vuln_code
                
                # Re-analyze after attack
                print("  Re-analyzing attacked vulnerable function...")
                raw_output_v2 = get_completion(attacked_vuln_code, model, tokenizer)
                entry["vuln_output_after_attack"] = raw_output_v2
                print(f"  After attack output: {raw_output_v2}")
                label_v2, _ = parse_model_output(raw_output_v2)
                entry["relabel_vuln_after_attack"] = label_v2
                print(f"  After attack label: {label_v2}")
            except Exception as e:
                print(f"  GCG attack on vulnerable failed: {e}")
                entry["relabel_vuln_after_attack"] = label_v
        else:
            print(f"  [SKIP] Not VULNERABLE or no variable found")
            entry["relabel_vuln_after_attack"] = label_v
            
        # GCG attack on benign function if needed
        if label_b == "VULNERABLE" and var_b:
            try:
                print(f"  Applying GCG attack on benign function (variable: {var_b})...")
                gcg_message_b = template.replace("{func}", benign_code)
                attacked_benign_code = apply_gcg_to_variable(gcg_message_b, var_b, model, tokenizer, num_steps=250)
                entry["attacked_benign_code"] = attacked_benign_code
                
                # Re-analyze after attack
                print("  Re-analyzing attacked benign function...")
                raw_output_b2 = get_completion(attacked_benign_code, model, tokenizer)
                entry["benign_output_after_attack"] = raw_output_b2
                print(f"  After attack output: {raw_output_b2}")
                label_b2, _ = parse_model_output(raw_output_b2)
                entry["relabel_benign_after_attack"] = label_b2
                print(f"  After attack label: {label_b2}")
            except Exception as e:
                print(f"  GCG attack on benign failed: {e}")
                entry["relabel_benign_after_attack"] = label_b
        else:
            entry["relabel_benign_after_attack"] = label_b
            
    except Exception as e:
        print(f"  Unexpected error: {e}")
        entry["error_message"] = str(e)
        # Mark as error
        if entry["initial_label_vuln"] == "UNKNOWN":
            entry["initial_label_vuln"] = "ERROR"
        if entry["initial_label_benign"] == "UNKNOWN":
            entry["initial_label_benign"] = "ERROR"
        if entry["relabel_vuln_after_attack"] == "UNKNOWN":
            entry["relabel_vuln_after_attack"] = "ERROR"
        if entry["relabel_benign_after_attack"] == "UNKNOWN":
            entry["relabel_benign_after_attack"] = "ERROR"
    
    return entry


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
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last processed pair"
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Retry only the pairs that had errors"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="fdtn-ai/Foundation-Sec-8B",
        help="Model to use"
    )
    args = parser.parse_args()

    # Read all paired functions
    print("Loading paired functions...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    all_pairs = read_paired_functions(args.file_path, args.num_pairs)
    filtered = []
    for v, b in all_pairs:
        # count tokens for each function
        len_v = len(tokenizer(v["func"], return_tensors="pt")["input_ids"][0])
        len_b = len(tokenizer(b["func"], return_tensors="pt")["input_ids"][0])
        if len_v <= 400 and len_b <= 400:
            filtered.append((v, b))
    print(f"Filtered out {len(all_pairs) - len(filtered)} pairs over 400 tokens")
    paired_functions = filtered
    total_pairs = len(paired_functions)
    print(f"Loaded {total_pairs} function pairs")
    
    # Separate vulnerable and benign functions
    vulnerable_funcs = []
    benign_funcs = []
    for vuln_dict, benign_dict in paired_functions:
        vulnerable_funcs.append(vuln_dict)
        benign_funcs.append(benign_dict)
    
    # Determine which pairs to process
    pairs_to_process = []
    
    if args.retry_errors:
        # Retry mode: only process pairs that had errors
        if not os.path.exists(args.output_path):
            print("Error: No output file found for retry mode")
            exit(1)
        error_indices = get_error_indices(args.output_path)
        pairs_to_process = error_indices
        print(f"Retry mode: Found {len(error_indices)} errors to retry")
        if not error_indices:
            print("No errors found to retry")
            exit(0)
    else:
        # Normal mode: determine start index
        start_idx = 0
        
        if os.path.exists(args.output_path):
            if args.resume:
                # Resume mode: start from last completed + 1
                last_idx = get_last_pair_index(args.output_path)
                if last_idx >= 0:
                    start_idx = last_idx + 1
                    print(f"Resume mode: Last completed pair_index = {last_idx}")
                    print(f"Starting from pair_index = {start_idx}")
            elif args.overwrite:
                # Overwrite mode: remove existing file
                os.remove(args.output_path)
                print(f"Removed existing file: {args.output_path}")
            else:
                # File exists but no resume/overwrite flag
                print(f"Error: Output file already exists: {args.output_path}")
                print("Options:")
                print("  1. Use --resume to continue from last pair")
                print("  2. Use --overwrite to start fresh")
                print("  3. Use --retry-errors to retry failed pairs")
                print("  4. Specify a different --output_path")
                exit(1)
        
        # Check if all pairs already processed
        if start_idx >= total_pairs:
            print(f"All pairs already processed! (start_idx={start_idx}, total={total_pairs})")
            exit(0)
        
        # Create list of pairs to process
        pairs_to_process = list(range(start_idx, total_pairs))
        print(f"Will process {len(pairs_to_process)} pairs")
    
    # Load model
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print("Model loaded successfully")
    
    # Load prompt templates
    template = get_prompt_template()
    template_gcg = get_prompt_template_gcg()
    
    # Process pairs
    processed_count = 0
    error_count = 0
    
    try:
        for i, pair_idx in enumerate(pairs_to_process):
            print(f"\n{'='*60}")
            print(f"Processing pair {pair_idx} ({i+1}/{len(pairs_to_process)})")
            print(f"{'='*60}")
            
            # Get the pair
            vuln_entry = vulnerable_funcs[pair_idx]
            benign_entry = benign_funcs[pair_idx]
            
            # Process the pair
            entry = process_single_pair(
                pair_idx, vuln_entry, benign_entry,
                model, tokenizer, template, template_gcg
            )

            if entry is None:
                skip_count += 1
                continue
            # Save result immediately
            append_result(args.output_path, entry)
            processed_count += 1
            
            # Check if this was an error
            if "error_message" in entry or entry.get("initial_label_vuln") == "ERROR":
                error_count += 1
                print(f"  [ERROR] Pair {pair_idx} completed with errors")
            else:
                print(f"  [SUCCESS] Pair {pair_idx} completed successfully")
            
            # Progress update every 10 pairs
            if (i + 1) % 10 == 0:
                print(f"\nProgress: {i+1}/{len(pairs_to_process)} pairs processed")
                print(f"Errors so far: {error_count}")
                
    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user at pair_index {pair_idx}")
        print(f"Processed {processed_count} pairs before interruption")
        print(f"To resume, run with --resume flag")
    except Exception as e:
        print(f"\n\nFatal error at pair_index {pair_idx}: {e}")
        print(f"Processed {processed_count} pairs before error")
        raise
    
    # Final summary
    print(f"\n{'='*60}")
    print("Processing Complete!")
    print(f"{'='*60}")
    print(f"Total pairs processed: {processed_count}")
    print(f"Pairs with errors: {error_count}")
    print(f"Results saved to: {args.output_path}")
    
    # Calculate statistics if we have results
    if os.path.exists(args.output_path):
        print("\nCalculating statistics...")
        results = []
        with open(args.output_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except:
                    continue
        
        if results:
            # Filter out error entries for statistics
            valid_results = [r for r in results if r.get("initial_label_vuln") != "ERROR"]
            
            if valid_results:
                vuln_correct_before = sum(1 for r in valid_results if r.get("initial_label_vuln") == "VULNERABLE")
                vuln_correct_after = sum(1 for r in valid_results if r.get("relabel_vuln_after_attack") == "VULNERABLE")
                benign_correct_before = sum(1 for r in valid_results if r.get("initial_label_benign") == "BENIGN")
                benign_correct_after = sum(1 for r in valid_results if r.get("relabel_benign_after_attack") == "BENIGN")
                
                total_valid = len(valid_results)
                print(f"\n=== Summary Statistics (excluding errors) ===")
                print(f"Valid results: {total_valid}/{len(results)}")
                print(f"Vulnerable correct before attack: {vuln_correct_before}/{total_valid} = {100.0 * vuln_correct_before / total_valid:.2f}%")
                print(f"Vulnerable correct after attack:  {vuln_correct_after}/{total_valid} = {100.0 * vuln_correct_after / total_valid:.2f}%")
                print(f"Benign correct before attack:     {benign_correct_before}/{total_valid} = {100.0 * benign_correct_before / total_valid:.2f}%")
                print(f"Benign correct after attack:      {benign_correct_after}/{total_valid} = {100.0 * benign_correct_after / total_valid:.2f}%")
                
                # Attack success rate
                vuln_changed = sum(1 for r in valid_results 
                                 if r.get("initial_label_vuln") == "VULNERABLE" and 
                                    r.get("relabel_vuln_after_attack") != "VULNERABLE")
                vuln_attacked = sum(1 for r in valid_results 
                                  if r.get("initial_label_vuln") == "VULNERABLE")
                
                if vuln_attacked > 0:
                    print(f"\nAttack success rate: {vuln_changed}/{vuln_attacked} = {100.0 * vuln_changed / vuln_attacked:.2f}%")
                    print(f"(Changed from VULNERABLE to non-VULNERABLE)")