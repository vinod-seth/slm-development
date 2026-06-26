import os
import re
import sys

# Define the jargon glossary with regex patterns and their definitions
GLOSSARY_PATTERNS = {
    "Gradient Accumulation": (
        r"Gradient [Aa]ccumulation",
        "Accumulating gradients over multiple smaller batches before performing a weight update, simulating a larger batch size."
    ),
    "Cosine Learning Rate Decay": (
        r"Cosine Learning Rate Decay|cosine learning rate decay|Cosine Annealing|cosine annealing",
        "A learning rate schedule that decreases the learning rate following a cosine curve."
    ),
    "Learning Rate": (
        r"Learning [Rr]ate[s]?",
        "A hyperparameter controlling how large a step the optimizer takes when adjusting weights to minimize loss."
    ),
    "Loss Function": (
        r"Loss [Ff]unction[s]?",
        "A mathematical function measuring the difference between the model's predictions and the true targets."
    ),
    "Weight Decay": (
        r"Weight [Dd]ecay",
        "A regularization technique that penalizes large weights to reduce overfitting."
    ),
    "Self-Attention": (
        r"Self-[Aa]ttention",
        "The core mechanism of Transformers allowing tokens to dynamically relate to and focus on other tokens in the sequence."
    ),
    "Attention": (
        r"Attention",
        "A mechanism that lets neural networks focus on specific parts of the input sequence when generating output."
    ),
    "PEFT": (
        r"PEFT",
        "Parameter-Efficient Fine-Tuning: techniques (like LoRA) that adapt pre-trained models by updating only a tiny fraction of parameters."
    ),
    "LoRA": (
        r"LoRA",
        "Low-Rank Adaptation: an efficient fine-tuning method that freezes base model weights and injects small trainable adapter matrices."
    ),
    "Quantization": (
        r"Quantization|quantization",
        "The process of reducing weight precision (e.g. from 16-bit to 4-bit) to shrink model size and speed up inference."
    ),
    "Tokenization": (
        r"Tokenization|tokenization",
        "The preprocessing step of converting raw text input into numerical tokens that a language model can process."
    ),
    "Perplexity": (
        r"Perplexity|perplexity",
        "A metric measuring how well a probability model predicts a sample; lower perplexity indicates higher confidence and quality."
    ),
    "ROUGE": (
        r"ROUGE",
        "Recall-Oriented Understudy for Gisting Evaluation: metrics evaluating summary quality by comparing against human references."
    ),
    "F1 Score": (
        r"F1 [Ss]core",
        "A metric that balances precision and recall, representing their harmonic mean."
    ),
    "SLM": (
        r"SLM[s]?",
        "Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware."
    ),
    "LLM": (
        r"LLM[s]?",
        "Large Language Model: a massive language model (7B+ parameters) requiring cloud or cluster hardware."
    ),
    "VRAM": (
        r"VRAM",
        "Video Random Access Memory: high-speed memory on a GPU used to store model weights and activations during run time."
    ),
    "GPU": (
        r"GPU[s]?",
        "Graphics Processing Unit: hardware optimized for parallel processing, essential for deep learning."
    ),
    "CPU": (
        r"CPU[s]?",
        "Central Processing Unit: the general-purpose processor in a computer."
    ),
    "FP16": (
        r"FP16",
        "16-bit Floating-Point: a half-precision format that halves memory usage and speeds up model computations."
    ),
    "Token": (
        r"Token[s]?",
        "A sub-word unit, word, or character that text is split into for processing by a language model."
    ),
    "Inference": (
        r"Inference|inference",
        "Running a trained model to generate predictions or text output from new, unseen inputs."
    ),
    "Fine-Tuning": (
        r"Fine-[Tt]uning|fine-tuning",
        "Adapting a pre-trained model to a specific task by training it further on a smaller, targeted dataset."
    ),
    "Pre-trained": (
        r"Pre-trained|pre-trained|pretrained",
        "A model trained on a massive general dataset to learn language patterns before fine-tuning."
    ),
    "Overfitting": (
        r"Overfitting|overfitting",
        "A training error where a model learns training data details too well, performing poorly on new data."
    ),
    "Underfitting": (
        r"Underfitting|underfitting",
        "A training error where a model is too simple to capture patterns, performing poorly overall."
    ),
    "Epoch": (
        r"Epoch[s]?",
        "One complete pass of the entire training dataset through the model during training."
    ),
    "Batch Size": (
        r"Batch [Ss]ize[s]?",
        "The number of training examples processed in a single forward and backward pass."
    ),
    "Backpropagation": (
        r"Backpropagation|backpropagation",
        "The algorithm that calculates gradients of the loss function with respect to weights by moving backward through the network."
    ),
    "Gradient": (
        r"Gradient[s]?",
        "A vector of partial derivatives indicating how to adjust model weights to minimize the loss function."
    ),
    "Optimizer": (
        r"Optimizer[s]?",
        "The algorithm (e.g. AdamW) that updates model weights based on computed gradients to minimize the loss."
    )
}

# Regex to split text into plain text and protected segments (code blocks are handled separately)
# Protects:
# 1. Inline code: `code`
# 2. Markdown images: ![alt](url)
# 3. Markdown links: [text](url)
# 4. Existing HTML tags: <abbr title="...">...</abbr>, etc.
RE_PROTECTED = re.compile(
    r"("
    r"`[^`\n]*`"
    r"|!\[[^\]]*\]\([^)]+\)"
    r"|\[[^\]]*\]\([^)]+\)"
    r"|<abbr[^>]*>.*?</abbr>"
    r"|<a[^>]*>.*?</a>"
    r"|<[^>]+>"
    r")",
    re.IGNORECASE | re.DOTALL
)

# Prepare combined regex for glossary terms
# Sort the keys by length descending to match longer terms first
sorted_keys = sorted(GLOSSARY_PATTERNS.keys(), key=len, reverse=True)
group_to_key = {}
regex_parts = []

for key in sorted_keys:
    pattern, definition = GLOSSARY_PATTERNS[key]
    # Create valid group name (alphanumeric and underscores only)
    group_name = key.replace(" ", "_").replace("-", "_")
    group_to_key[group_name] = (key, definition)
    regex_parts.append(f"(?P<{group_name}>\\b(?:{pattern})\\b)")

COMBINED_JARGON_REGEX = re.compile("|".join(regex_parts))

def tokenize_markdown(content):
    """
    Tokenizes a markdown string into segments:
    - ('code_block', content)
    - ('header', line_content)
    - ('protected', content) - inline code, links, images, HTML tags
    - ('text', content) - plain text where substitutions are allowed
    """
    blocks = content.split("```")
    tokenized = []
    
    for idx, block in enumerate(blocks):
        if idx % 2 == 1:
            # Inside a code block, keep as-is
            tokenized.append(("code_block", "```" + block + "```"))
        else:
            # Outside code blocks, parse line-by-line
            lines = block.splitlines(keepends=True)
            for line in lines:
                if line.strip().startswith("#"):
                    # Don't modify headers
                    tokenized.append(("header", line))
                else:
                    # Split line into protected parts and text
                    parts = RE_PROTECTED.split(line)
                    for j, part in enumerate(parts):
                        if not part:
                            continue
                        if j % 2 == 1:
                            tokenized.append(("protected", part))
                        else:
                            tokenized.append(("text", part))
    return tokenized

def process_content(content):
    """
    Processes the markdown content, wrapping the first occurrence of each jargon
    term in an <abbr> tag.
    """
    tokens = tokenize_markdown(content)
    replaced_in_file = set()
    
    def replace_match(match):
        group_name = match.lastgroup
        if group_name in group_to_key:
            key, definition = group_to_key[group_name]
            matched_text = match.group(group_name)
            if key not in replaced_in_file:
                replaced_in_file.add(key)
                return f'<abbr title="{definition}">{matched_text}</abbr>'
            else:
                return matched_text
        return match.group(0)

    new_content = []
    for token_type, val in tokens:
        if token_type == "text":
            # Apply replacements to plain text segments
            replaced_text = COMBINED_JARGON_REGEX.sub(replace_match, val)
            new_content.append(replaced_text)
        else:
            # Keep headers, code blocks, and protected HTML/links untouched
            new_content.append(val)
            
    return "".join(new_content)

def process_file(filepath):
    """Reads a file, applies tooltips, and writes it back if modified."""
    print(f"Processing: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
        
    updated_content = process_content(content)
    
    if content != updated_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(updated_content)
        print(f" -> Updated!")
        return True
    else:
        print(f" -> No changes.")
        return False

def run_tests():
    """Runs a series of tests to verify correct replacement behavior."""
    print("Running tests...")
    
    # Test case 1: Basic replacement
    test_input = "An SLM is smaller than an LLM. An SLM has fewer parameters."
    expected = 'An <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> is smaller than an <abbr title="Large Language Model: a massive language model (7B+ parameters) requiring cloud or cluster hardware.">LLM</abbr>. An SLM has fewer parameters.'
    output = process_content(test_input)
    assert output == expected, f"\nExpected: {expected}\nGot:      {output}"
    
    # Test case 2: Do not replace inside code blocks
    test_input_code = "Here is text: SLM is great.\n```python\n# This is code\nmodel = load_slm()\n```"
    expected_code = 'Here is text: <abbr title="Small Language Model: a compact language model (under ~3B parameters) that can run on consumer hardware.">SLM</abbr> is great.\n```python\n# This is code\nmodel = load_slm()\n```'
    output_code = process_content(test_input_code)
    assert output_code == expected_code, f"\nExpected: {expected_code}\nGot:      {output_code}"
    
    # Test case 3: Do not replace inside inline code or markdown links
    test_input_links = "Please check `SLM` or click [link to LLM](http://llm.com)."
    expected_links = "Please check `SLM` or click [link to LLM](http://llm.com)."
    output_links = process_content(test_input_links)
    assert output_links == expected_links, f"\nExpected: {expected_links}\nGot:      {output_links}"

    # Test case 4: Match longer phrases first (Self-Attention vs Attention)
    test_input_attention = "We use Self-Attention. Attention is all you need."
    expected_attention = 'We use <abbr title="The core mechanism of Transformers allowing tokens to dynamically relate to and focus on other tokens in the sequence.">Self-Attention</abbr>. <abbr title="A mechanism that lets neural networks focus on specific parts of the input sequence when generating output.">Attention</abbr> is all you need.'
    output_attention = process_content(test_input_attention)
    assert output_attention == expected_attention, f"\nExpected: {expected_attention}\nGot:      {output_attention}"
    
    # Test case 5: Do not double-wrap already wrapped tags
    test_input_nested = 'An <abbr title="Small Language Model">SLM</abbr> is here.'
    expected_nested = 'An <abbr title="Small Language Model">SLM</abbr> is here.'
    output_nested = process_content(test_input_nested)
    assert output_nested == expected_nested, f"\nExpected: {expected_nested}\nGot:      {output_nested}"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    elif len(sys.argv) > 1:
        # Process a specific file
        process_file(sys.argv[1])
    else:
        # Traverse the tutorial directory and process all markdown files
        tutorial_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tutorial")
        if not os.path.exists(tutorial_dir):
            # Try current directory if run from tutorial/ or scripts/
            tutorial_dir = "./tutorial"
            
        print(f"Scanning directory: {tutorial_dir}")
        count = 0
        for root, dirs, files in os.walk(tutorial_dir):
            for file in files:
                if file.endswith(".md"):
                    filepath = os.path.join(root, file)
                    if process_file(filepath):
                        count += 1
        print(f"Finished. Updated {count} files.")
