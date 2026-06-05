#!/usr/bin/env python3
"""
Test tokenization of "Yes" and "No" with Qwen tokenizer
"""

from transformers import AutoProcessor

# Load Qwen processor
print("Loading Qwen3-VL processor...")
processor = AutoProcessor.from_pretrained('Qwen/Qwen2.5-VL-7B-Instruct')
tokenizer = processor.tokenizer

print("\n" + "="*80)
print("TOKENIZATION TESTS")
print("="*80)

# Test cases
test_cases = [
    "Yes",
    " Yes",
    "No",
    " No",
    "Answer: Yes",
    "Answer: No",
    ": Yes",
    ": No",
]

for text in test_cases:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    decoded = tokenizer.decode(token_ids)
    individual_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    print(f"\nText: '{text}'")
    print(f"  Token IDs: {token_ids}")
    print(f"  Number of tokens: {len(token_ids)}")
    print(f"  Decoded back: '{decoded}'")
    print(f"  Individual tokens: {individual_tokens}")

print("\n" + "="*80)
print("COMPARISON: Does tokenization differ with/without space?")
print("="*80)

yes_no_space = tokenizer.encode("Yes", add_special_tokens=False)
yes_with_space = tokenizer.encode(" Yes", add_special_tokens=False)
no_no_space = tokenizer.encode("No", add_special_tokens=False)
no_with_space = tokenizer.encode(" No", add_special_tokens=False)

print(f"\n'Yes' (no space): {yes_no_space}")
print(f"' Yes' (with space): {yes_with_space}")
print(f"Are they the same? {yes_no_space == yes_with_space}")

print(f"\n'No' (no space): {no_no_space}")
print(f"' No' (with space): {no_with_space}")
print(f"Are they the same? {no_no_space == no_with_space}")

print("\n" + "="*80)
print("SIMULATION: What happens in typical generation?")
print("="*80)

# Simulate what's generated at the end
full_text = "Critique: The caption is good.\n\nAnswer: Yes"
full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
full_decoded_individual = [tokenizer.decode([tid]) for tid in full_tokens]

print(f"\nFull text: '{full_text}'")
print(f"Total tokens: {len(full_tokens)}")
print(f"\nLast 5 tokens:")
for i in range(5):
    idx = -(5-i)
    token_id = full_tokens[idx]
    token_text = tokenizer.decode([token_id])
    print(f"  Position {idx}: ID={token_id}, Text='{token_text}'")

print(f"\nLast token is: '{tokenizer.decode([full_tokens[-1]])}'")
print(f"Last token ID: {full_tokens[-1]}")

# Check if it matches our answer templates
print(f"\nDoes last token match 'Yes'? {full_tokens[-1] in yes_no_space}")
print(f"Does last token match ' Yes'? {full_tokens[-1] in yes_with_space}")

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)

if yes_no_space == yes_with_space:
    print("\n✓ 'Yes' and ' Yes' tokenize the same way")
    print("  → Use either one, doesn't matter")
else:
    print("\n✗ 'Yes' and ' Yes' tokenize DIFFERENTLY")
    print("  → You MUST use the one that matches generation context")
    if full_tokens[-1] in yes_with_space:
        print("  → In 'Answer: Yes' context, use ' Yes' (with space)")
    elif full_tokens[-1] in yes_no_space:
        print("  → In 'Answer: Yes' context, use 'Yes' (no space)")