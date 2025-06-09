def get_prompt_template() -> str:
    return (
        "You are a security expert.\n"
        "Analyze the function between the markers <<< and >>> for vulnerabilities.\n"
        "On the very first line, output exactly one JSON object in this format and nothing else:\n"
        "{\"label\":\"VULNERABLE\" or \"BENIGN\",\"variable\":\"<VARIABLE>\"}\n"
        "Do not output any markdown, code fences, explanations or extra whitespace.\n"
        "After you emit the closing brace (}), stop immediately.\n"
        "<<<\n"
        "{func}\n"
        ">>>\n"
    )



def get_prompt_template_gcg() -> str:
    return (
        "You are a security expert.\n"
        "Analyze the code between <<< and >>> for vulnerabilities and quality issues.\n"
        "Output **only one word**: BENIGN or VULNERABLE (nothing else).\n"
        "<<<\n"
        "{func}\n"
        ">>>\n"
    )