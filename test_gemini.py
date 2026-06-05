"""
Diagnostic script for Gemini 2.5 Flash VQAScore on video.
Run: python probe_gemini.py
"""
import math
import t2v_metrics
from t2v_metrics.models.vqascore_models.gemini_model import (
    GeminiModel, SAFETY_SETTINGS, GenerateContentConfig,
    find_first_output_token_index, default_question_template
)

VIDEO  = "videos/baby.mp4"
TEXT   = "Does the video show 'a baby crying'? Answer 'Yes' or 'No'."
MODEL  = "gemini-2.5-flash"

# --- Load ---
print(f"Loading {MODEL}...")
scorer = t2v_metrics.VQAScore(model=MODEL)
model: GeminiModel = scorer.model

# --- Score ---
print(f"\nRunning VQAScore on: {VIDEO}")
score = scorer(images=[VIDEO], texts=[TEXT])
print(f"Score: {score}")

# --- Probe: call API directly with no error swallowing ---
question = default_question_template.format(TEXT)
answer   = 'Yes'
print(f"\nQuestion: {question!r}")

loaded = model.load_images([VIDEO])
data   = loaded[0]

config = GenerateContentConfig(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    response_logprobs=True,
    logprobs=20,
    max_output_tokens=65536,
    safety_settings=SAFETY_SETTINGS,
)

print("\nCalling Gemini API directly (no error catching)...")
try:
    response = model.client.models.generate_content(
        model=MODEL,
        contents=model._build_parts(data, question),
        config=config,
    )
    print(f"\nFull generated text: {response.text!r}")

    logprobs_result = response.candidates[0].logprobs_result
    if logprobs_result is None:
        print("\n[ERROR] logprobs_result is None")
    else:
        chosen    = logprobs_result.chosen_candidates
        top       = logprobs_result.top_candidates
        first_idx = find_first_output_token_index(chosen)

        print(f"Total tokens: {len(chosen)}")
        print(f"First real output token index: {first_idx}")

        if first_idx is None:
            print("[ERROR] No output token found — all tokens are thinking tokens")
        else:
            print(f"\nChosen token at [{first_idx}]: {chosen[first_idx].token!r}")
            print(f"\nTop {len(top[first_idx].candidates)} candidates:")
            for i, c in enumerate(top[first_idx].candidates):
                prob  = math.exp(c.log_probability)
                match = "✓" if answer.lower() in c.token.lower().strip() else " "
                print(f"  {i+1:2d}. {match} {c.token!r:20s}  prob={prob:.6f}")

except Exception as e:
    print(f"\n[ACTUAL ERROR — was being silently swallowed]: {e}")