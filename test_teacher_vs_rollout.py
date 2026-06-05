import t2v_metrics
import torch

print("="*80)
print("Testing VQAScore: Generation vs Teacher Forcing Comparison")
print("="*80)

# Initialize the model
qwen_score = t2v_metrics.VQAScore(model='qwen2.5-vl-7b')

def compare_methods(images, texts, answer_template, max_new_tokens, test_name):
    """Helper function to compare generation and teacher forcing methods"""
    print(f"\n{'='*80}")
    print(f"{test_name}")
    print(f"{'='*80}")
    print(f"Answer template: '{answer_template}'")
    print(f"Max new tokens: {max_new_tokens}")
    
    # Generation approach
    print("\n[Generation Approach]")
    gen_scores, gen_traces = qwen_score.model.forward_with_trace(
        images=images,
        texts=texts,
        answer_template=answer_template,
        max_new_tokens=max_new_tokens
    )
    
    # Teacher forcing approach
    print("\n[Teacher Forcing Approach]")
    tf_scores, tf_traces = qwen_score.model.forward_with_trace_teacher_forcing(
        images=images,
        texts=texts,
        answer_template=answer_template,
        max_new_tokens=max_new_tokens
    )
    
    # Compare results
    print(f"\n{'─'*80}")
    print("COMPARISON:")
    print(f"{'─'*80}")
    for i in range(len(images)):
        print(f"\nSample {i+1}: {texts[i]}")
        print(f"  Generated text: '{gen_traces[i]['generated_text']}'")
        print(f"  Scored tokens:  '{gen_traces[i]['scored_tokens_text']}'")
        print(f"  Generation probability:    {gen_scores[i].item():.8f}")
        print(f"  Teacher forcing probability: {tf_scores[i].item():.8f}")
        diff = abs(gen_scores[i].item() - tf_scores[i].item())
        print(f"  Absolute difference:        {diff:.2e}")
        print(f"  Match (< 1e-6):             {diff < 1e-6} {'✓' if diff < 1e-6 else '✗'}")
    
    # Overall match
    all_match = torch.allclose(gen_scores, tf_scores, atol=1e-6)
    print(f"\n{'─'*80}")
    print(f"ALL SAMPLES MATCH: {all_match} {'✓' if all_match else '✗'}")
    print(f"{'─'*80}")
    
    return gen_scores, tf_scores, gen_traces, tf_traces

# Test cases
video = "videos/baby.mp4"
text = "a baby crying"

# ============================================================================
# NON-COT CASES (Short generations)
# ============================================================================

print("\n" + "="*80)
print("NON-COT CASES")
print("="*80)

# Test 1: Single token answer
compare_methods(
    images=[video],
    texts=[text],
    answer_template="Yes",
    max_new_tokens=1,
    test_name="TEST 1: Single token answer - 'Yes'"
)

# Test 2: Single token answer - 'No'
compare_methods(
    images=[video],
    texts=[text],
    answer_template="No",
    max_new_tokens=1,
    test_name="TEST 2: Single token answer - 'No'"
)

# Test 3: Multi-token answer (2 tokens)
compare_methods(
    images=[video],
    texts=[text],
    answer_template="Yes, definitely",
    max_new_tokens=5,
    test_name="TEST 3: Multi-token answer (2 tokens)"
)

# Test 4: Multi-token answer (longer phrase)
compare_methods(
    images=[video],
    texts=[text],
    answer_template="Yes, it does",
    max_new_tokens=5,
    test_name="TEST 4: Multi-token answer (3 tokens)"
)

# Test 5: Batch of multiple videos
compare_methods(
    images=[video, video],
    texts=["a baby crying", "a dog barking"],
    answer_template="Yes",
    max_new_tokens=1,
    test_name="TEST 5: Batch processing (2 samples)"
)

# Test 6: Mismatched text (sanity check)
compare_methods(
    images=[video],
    texts=["a car driving"],
    answer_template="Yes",
    max_new_tokens=1,
    test_name="TEST 6: Mismatched content (should have low probability)"
)

# ============================================================================
# COT CASES (Longer generations)
# ============================================================================

print("\n" + "="*80)
print("COT CASES")
print("="*80)

# Test 7: CoT-style with longer generation
compare_methods(
    images=[video],
    texts=[text],
    answer_template="Yes",
    max_new_tokens=50,
    test_name="TEST 7: CoT-style generation (answer 'Yes' at end)"
)

# Test 8: CoT-style with 'No' answer
compare_methods(
    images=[video],
    texts=["a car driving"],
    answer_template="No",
    max_new_tokens=50,
    test_name="TEST 8: CoT-style generation (answer 'No' at end)"
)

# Test 9: CoT-style with multi-token answer at end
compare_methods(
    images=[video],
    texts=[text],
    answer_template="Yes, definitely",
    max_new_tokens=50,
    test_name="TEST 9: CoT-style with multi-token answer at end"
)

# ============================================================================
# CONSISTENCY TESTS
# ============================================================================

print("\n" + "="*80)
print("CONSISTENCY TESTS")
print("="*80)

# Test 10: Run same test twice with generation method
print("\nTEST 10: Generation method consistency")
print("─"*80)
score1, trace1 = qwen_score.model.forward_with_trace(
    images=[video], texts=[text], answer_template="Yes", max_new_tokens=1
)
score2, trace2 = qwen_score.model.forward_with_trace(
    images=[video], texts=[text], answer_template="Yes", max_new_tokens=1
)
print(f"Run 1: {score1.item():.8f}")
print(f"Run 2: {score2.item():.8f}")
print(f"Difference: {abs(score1.item() - score2.item()):.2e}")
print(f"Identical: {torch.allclose(score1, score2)} {'✓' if torch.allclose(score1, score2) else '✗'}")

# Test 11: Run same test twice with teacher forcing method
print("\nTEST 11: Teacher forcing method consistency")
print("─"*80)
score1, trace1 = qwen_score.model.forward_with_trace_teacher_forcing(
    images=[video], texts=[text], answer_template="Yes", max_new_tokens=1
)
score2, trace2 = qwen_score.model.forward_with_trace_teacher_forcing(
    images=[video], texts=[text], answer_template="Yes", max_new_tokens=1
)
print(f"Run 1: {score1.item():.8f}")
print(f"Run 2: {score2.item():.8f}")
print(f"Difference: {abs(score1.item() - score2.item()):.2e}")
print(f"Identical: {torch.allclose(score1, score2)} {'✓' if torch.allclose(score1, score2) else '✗'}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("SUMMARY OF TESTS")
print("="*80)
print("\nNON-COT CASES:")
print("  ✓ Test 1: Single token 'Yes'")
print("  ✓ Test 2: Single token 'No'")
print("  ✓ Test 3: Multi-token (2 tokens)")
print("  ✓ Test 4: Multi-token (3 tokens)")
print("  ✓ Test 5: Batch processing")
print("  ✓ Test 6: Mismatched content")
print("\nCOT CASES:")
print("  ✓ Test 7: CoT with 'Yes' at end")
print("  ✓ Test 8: CoT with 'No' at end")
print("  ✓ Test 9: CoT with multi-token answer")
print("\nCONSISTENCY:")
print("  ✓ Test 10: Generation method consistency")
print("  ✓ Test 11: Teacher forcing consistency")
print("\n" + "="*80)
print("EXPECTED RESULT: All generation vs teacher forcing comparisons")
print("should show < 1e-6 difference, confirming both methods are equivalent.")
print("="*80)