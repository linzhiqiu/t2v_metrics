"""
Diagnostic for Gemma 4 VQAScore.
Run: python probe_gemma4.py
"""
import torch
import t2v_metrics
from t2v_metrics.models.vqascore_models.gemma4_model import Gemma4Model

IMAGE = "images/0.png"
VIDEO = "videos/baby.mp4"
MODEL = "gemma-4-12b-it"

print(f"Loading {MODEL}...")
scorer = t2v_metrics.VQAScore(model=MODEL)
model: Gemma4Model = scorer.model

question = 'Does this figure show "a baby crying"? Please answer Yes or No.'
answer   = 'Yes'

# Load media
processed = model.load_images([VIDEO])
content   = processed[0]
inputs    = model._build_inputs(content, question)

answer_token_ids = model.processor.tokenizer.encode(answer, add_special_tokens=False)
print(f"\n'Yes' encodes to token IDs: {answer_token_ids}")
print(f"Token text: {[model.processor.tokenizer.decode([t]) for t in answer_token_ids]}")

# Generate with scoring
with torch.inference_mode():
    outputs = model.model.generate(
        **inputs,
        max_new_tokens=10,   # more tokens to see full output
        temperature=1.0,
        do_sample=False,
        output_scores=True,
        return_dict_in_generate=True,
    )

generated_ids  = outputs.sequences[0][inputs['input_ids'].shape[1]:]
generated_text = model.processor.tokenizer.decode(generated_ids, skip_special_tokens=False)

print(f"\nGenerated token IDs: {generated_ids.tolist()}")
print(f"Generated text (raw): {generated_text!r}")
print(f"Number of score steps: {len(outputs.scores)}")

# Show top tokens at each generated position
for pos, score in enumerate(outputs.scores):
    probs        = torch.nn.functional.softmax(score[0], dim=-1)
    top_probs, top_ids = torch.topk(probs, 5)
    chosen_id    = generated_ids[pos].item()
    yes_prob     = probs[answer_token_ids[0]].item() if answer_token_ids else 0.0

    print(f"\n  Step {pos} — chosen: {model.processor.tokenizer.decode([chosen_id])!r} (id={chosen_id})"
          f"  P('Yes')={yes_prob:.6f}")
    for p, tid in zip(top_probs, top_ids):
        print(f"    {model.processor.tokenizer.decode([tid.item()])!r:20s}  id={tid.item():8d}  P={p.item():.6f}")