"""
Diagnostic script for Qwen3.5 VQAScore.
Probes why scores are suspiciously low using forward_with_trace + debug mode.
Run: python probe_qwen35.py
"""
import torch
import t2v_metrics
from t2v_metrics.models.vqascore_models.qwen3vl_model import Qwen3VLModel

IMAGE = "images/0.png"
VIDEO = "videos/baby.mp4"
MODEL = "qwen3.5-9b"

print(f"Loading {MODEL}...")
scorer = t2v_metrics.VQAScore(model=MODEL)
model: Qwen3VLModel = scorer.model

print(f"\n{'='*60}")
print(f"Model name: {model.model_name}")
print(f"enable_thinking check: {'qwen3.5' in model.model_name}")
print(f"{'='*60}")

# -----------------------------------------------------------------------
# Test 1: Image with debug trace
# -----------------------------------------------------------------------
print("\n\n>>> TEST 1: IMAGE with forward_with_trace (debug=True)")
scores, traces = model.forward_with_trace(
    images=[IMAGE],
    texts=["a dog"],
    debug=True,
    score_position="end"
)
print(f"\nFinal score: {scores[0]:.6f}")

# -----------------------------------------------------------------------
# Test 2: Video with debug trace
# -----------------------------------------------------------------------
print("\n\n>>> TEST 2: VIDEO with forward_with_trace (debug=True)")
scores, traces = model.forward_with_trace(
    images=[VIDEO],
    texts=["a baby crying"],
    debug=True,
    score_position="end"
)
print(f"\nFinal score: {scores[0]:.6f}")

# -----------------------------------------------------------------------
# Test 3: Manually verify enable_thinking is being passed
# Check what the raw apply_chat_template produces
# -----------------------------------------------------------------------
print("\n\n>>> TEST 3: Verify enable_thinking=False is applied")
from PIL import Image as PILImage

image = PILImage.open(IMAGE).convert('RGB')
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": 'Does this figure show "a dog"? Please answer Yes or No.'}
        ]
    }
]

# With enable_thinking=False
text_no_thinking = model.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    **({'enable_thinking': False} if 'qwen3.5' in model.model_name else {})
)
print(f"\nWith enable_thinking=False:")
print(f"  Prompt (last 200 chars): ...{text_no_thinking[-200:]!r}")

# Without enable_thinking (to compare)
text_with_thinking = model.processor.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
print(f"\nWithout enable_thinking flag:")
print(f"  Prompt (last 200 chars): ...{text_with_thinking[-200:]!r}")

print(f"\nPrompts are identical: {text_no_thinking == text_with_thinking}")

# -----------------------------------------------------------------------
# Test 4: Try score_position="start" to see if the answer is at the
# beginning of generation instead of the end
# -----------------------------------------------------------------------
print("\n\n>>> TEST 4: score_position='start' vs 'end' comparison")
for position in ["start", "end"]:
    scores, traces = model.forward_with_trace(
        images=[VIDEO],
        texts=["a baby crying"],
        score_position=position,
        debug=False
    )
    t = traces[0]
    print(f"  position='{position}': score={scores[0]:.6f}  "
          f"scored_text={t['scored_tokens_text']!r}  "
          f"generated={t['generated_text'][:80]!r}")

# -----------------------------------------------------------------------
# Test 5: Direct generation to see what the model actually outputs
# -----------------------------------------------------------------------
print("\n\n>>> TEST 5: Raw generation output")
responses = model.generate(
    images=[IMAGE, VIDEO],
    texts=[
        'Does this figure show "a dog"? Please answer Yes or No.',
        'Does this figure show "a baby crying"? Please answer Yes or No.',
    ]
)
for media, response in zip([IMAGE, VIDEO], responses):
    print(f"  {media}: {response!r}")