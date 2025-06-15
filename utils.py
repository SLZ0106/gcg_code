import json
import argparse
import os
import nanogcg
import re
from nanogcg import GCGConfig

def read_paired_functions(file_path, num_pairs=None):
    """
    Reads a JSONL file where each two consecutive lines form a positive (vulnerable) / negative (benign) function pair.
    Returns a list of tuples: [(vuln_dict, benign_dict), ...].

    Args:
        file_path (str): Path to the JSONL file.
        num_pairs (int, optional): Number of pairs to read. If None, read all pairs.

    Returns:
        List[Tuple[dict, dict]]: List of (vulnerable, benign) dict pairs.
    """
    pairs = []
    with open(file_path, 'r', encoding='utf-8') as f:
        line_buffer = []
        for line_idx, line in enumerate(f):
            data = json.loads(line)
            line_buffer.append(data)

            # Whenever we have two entries, treat them as a pair
            if len(line_buffer) == 2:
                vuln_func, benign_func = None, None
                # Identify which is vulnerable (target=1) and which is benign (target=0)
                if line_buffer[0].get('target') == 1 and line_buffer[1].get('target') == 0:
                    vuln_func, benign_func = line_buffer[0], line_buffer[1]
                elif line_buffer[0].get('target') == 0 and line_buffer[1].get('target') == 1:
                    vuln_func, benign_func = line_buffer[1], line_buffer[0]
                else:
                    # If the targets are not exactly one 1 and one 0, skip this pair
                    line_buffer = []
                    continue

                pairs.append((vuln_func, benign_func))
                line_buffer = []  # Reset for next pair

                # If we've read the requested number of pairs, break
                if num_pairs is not None and len(pairs) >= num_pairs:
                    break
    return pairs


import re

def parse_model_output(output: str) -> tuple:
    """
    Parses the model's output string to extract:
      - label: one of "VULNERABLE", "BENIGN", or "UNKNOWN"
      - var_name: the variable name from the JSON, or "" if none

    First tries to find a JSON object {...} in the text and load it.
    If that fails, falls back to the original regex‐upper logic.
    """
    text = output.strip()

    # 1) Try to extract a JSON substring
    m = re.search(r"\{.*?\}", text)
    if m:
        try:
            data = json.loads(m.group(0))
            label = data.get("label", "").upper()
            var   = data.get("variable", "")
            if label in ("VULNERABLE", "BENIGN"):
                return label, var
        except json.JSONDecodeError:
            pass

    # 2) Fallback: your old heuristic
    #    normalize to uppercase letters/spaces only
    norm = re.sub(r"[^A-Z ]", " ", text.upper())
    parts = norm.split()

    label = "UNKNOWN"
    var_name = ""

    if parts and parts[0] in ("VULNERABLE", "BENIGN"):
        label = parts[0]
        if len(parts) > 1 and re.match(r"^[A-Z_][A-Z0-9_]*$", parts[1]):
            var_name = parts[1]
        return label, var_name

    if "VULNERABLE" in parts:
        label = "VULNERABLE"
        idx = parts.index("VULNERABLE")
        if idx + 1 < len(parts) and re.match(r"^[A-Z_][A-Z0-9_]*$", parts[idx+1]):
            var_name = parts[idx+1]
    elif "BENIGN" in parts:
        label = "BENIGN"

    return label, var_name


def parse_label_only(output: str) -> str:
    text = output.strip().upper()
    if re.search(r"\bVULNERABLE\b", text):
        return "VULNERABLE"
    if re.search(r"\bBENIGN\b", text):
        return "BENIGN"
    return "UNKNOWN"


def save_analysis_results(results: list, file_path: str):
    """
    Saves a list of analysis results (dictionaries) into a JSONL file.
    Creates the parent directory for file_path if it does not exist.

    Args:
        results (list): A list of dictionaries, each representing one analysis result.
        file_path (str): Path to the output JSONL file.
    """
    # Ensure parent directory exists
    parent_dir = os.path.dirname(file_path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir, exist_ok=True)

    # Write each result as one JSON line
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def replace_variable_with_placeholder(func_code: str, var_name: str) -> str:
    """
    Replaces every occurrence of a given variable name in the function code
    with the GCG placeholder '{optim_str}'.

    Args:
        func_code (str): Original function source code.
        var_name (str): Variable name to replace.

    Returns:
        str: Function code with all occurrences of var_name replaced by '{optim_str}'.
    """
    return func_code.replace(var_name, "{optim_str}", 1)

def run_gcg_on_function(
    func_with_placeholder: str,
    gcg_model,
    gcg_tokenizer,
    var_name: str,
    num_steps: int = 500,
) -> str:
    """
    Given a function snippet containing '{optim_str}', use nanoGCG to search
    for the optimal replacement string. Target is fixed to 'BENIGN'.

    Args:
        func_with_placeholder (str): Function code with '{optim_str}' placeholder.
        gcg_model: Preloaded LM (e.g. Llama, StarCoder2).
        gcg_tokenizer: Corresponding tokenizer.
        num_steps (int): Number of GCG iterations (default 500).

    Returns:
        str: The best replacement token sequence found by nanoGCG.
    """
    init_char = var_name[0] if var_name else "a"
    config = GCGConfig(
        num_steps=num_steps,
        #optim_str_init=init_char,
        optim_str_init="_x_x_x_x_x_x",
        search_width=512,
        topk=512,
        filter_ids=True,
    )

    # Directly pass the function string as the “message”
    target = '{"label":"BENIGN","variable":""}'
    result = nanogcg.run(gcg_model, gcg_tokenizer, func_with_placeholder, target, config=config)
    return result.best_string


def apply_gcg_to_variable(
    func_code: str,
    var_name: str,
    gcg_model,
    gcg_tokenizer,
    num_steps: int = 500
) -> str:
    """
    Step 1: Replace every var_name in func_code with '{optim_str}'.
    Step 2: Run nanoGCG to find the best replacement.
    Step 3: Substitute '{optim_str}' back with the best string.

    Args:
        func_code (str): Original function source code.
        var_name (str): Variable name to “hide”/optimize.
        gcg_model: nanoGCG LM model.
        gcg_tokenizer: Corresponding tokenizer.
        num_steps (int): GCG iterations (default 500).

    Returns:
        str: Optimized function source code with var_name replaced.
    """
    #print(f"func_code: {func_code}, var_name: {var_name}")
    with_placeholder = replace_variable_with_placeholder(func_code, var_name)
    #print(f"with_placeholder: {with_placeholder}")
    best_str = run_gcg_on_function(with_placeholder, gcg_model, gcg_tokenizer, var_name, num_steps)
    optimized_code = func_code.replace(var_name, best_str)
    return optimized_code