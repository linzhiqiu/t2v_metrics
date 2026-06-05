"""
Test Script: VQAScore and Generation Functionality
===================================================
Runs tests sequentially and stops immediately on the first failure,
printing the full traceback at the point of failure.

Model groups:
  Local models    : qwen2.5-vl-7b, qwen3-vl-8b, paligemma-3b-mix-224,
                    internvl2-8b, qwen3-omni-30b-a3b
  GPT             : gpt-4o              (requires OPENAI_API_KEY)
  Gemini Vertex   : gemini-2.5-pro, gemini-2.5-flash, gemini-3-flash-preview,
                    gemini-3-pro-preview, gemini-3.1-pro-preview
                    (requires GOOGLE_CLOUD_PROJECT + GOOGLE_CLOUD_LOCATION + ADC)
  Gemini API key  : same models         (requires GEMINI_API_KEY)

Usage:
    python test.py                          # run all groups
    python test.py --skip-local             # API models only
    python test.py --skip-api               # local models only
    python test.py --skip-gemini-vertex     # skip Vertex AI Gemini tests
    python test.py --skip-gemini-api        # skip Gemini API key tests
    python test.py --image images/0.png --video videos/baby.mp4
"""

import argparse
import contextlib
import os
import traceback
import sys
from typing import List

import torch
import t2v_metrics

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_IMAGE = "images/0.png"
DEFAULT_VIDEO = "videos/baby.mp4"

IMAGE_TEXT = "a dog"
VIDEO_TEXT = "Does the video show 'a baby crying'? Answer 'Yes' or 'No'."
GEN_PROMPT = "Briefly describe what you see in one sentence."

LOCAL_MODELS: List[str] = [
    # "qwen2.5-vl-7b",        # Qwen2.5-VL      (qwen2vl_model.py)
    # "qwen3-vl-8b",          # Qwen3-VL         (qwen3vl_model.py)
    # "qwen3.5-9b",           # Qwen3.5          (qwen3vl_model.py — tests enable_thinking path)
    # "paligemma-3b-mix-224", # PaliGemma        (paligemma_model.py — image-only)
    # "internvl3-8b",         # InternVL3        (internvl_model.py)
    # "internvl3.5-8b",       # InternVL3.5      (internvl_model.py)
    # "gemma-3-4b-it",        # Gemma 3          (gemma3_model.py)
    'gemma-4-12b-it',
    # "molmo2-4b",            # Molmo2           (molmo2_model.py)
    "qwen3-omni-30b-a3b",   # Qwen3-Omni       (qwen3omni_model.py)
]

GPT_MODELS: List[str] = [
    "gpt-4o",               # GPT-4o           (gpt4v_model.py)
    "gpt-4.1",              # GPT-4.1          (gpt4v_model.py)
]

GEMINI_MODELS_TO_TEST: List[str] = [
    "gemini-2.5-flash",     # Gemini 2.5 Flash (gemini_model.py)
    "gemini-2.5-pro",       # Gemini 2.5 Pro   (gemini_model.py)
]

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

PASS_TAG = "\033[92m PASS \033[0m"
FAIL_TAG = "\033[91m FAIL \033[0m"
SEP      = "-" * 60

# ---------------------------------------------------------------------------
# TestFailure exception — raised immediately on any failure
# ---------------------------------------------------------------------------

class TestFailure(Exception):
    """Raised when a test case fails. Carries the full context."""
    def __init__(self, model: str, test: str, message: str):
        self.model   = model
        self.test    = test
        self.message = message
        super().__init__(message)

# ---------------------------------------------------------------------------
# Context manager: temporarily suppress an env var
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def suppress_env_var(key: str):
    old = os.environ.pop(key, None)
    try:
        yield
    finally:
        if old is not None:
            os.environ[key] = old

# ---------------------------------------------------------------------------
# Individual test cases — raise TestFailure immediately on any issue
# ---------------------------------------------------------------------------

def test_single_image_score(scorer, model_name: str, image: str) -> None:
    test = "single_image_score"
    try:
        score = scorer(images=[image], texts=[IMAGE_TEXT])
        assert score.shape == (1, 1), f"Expected shape (1,1), got {score.shape}"
        val = score.item()
        assert 0.0 <= val <= 1.0, f"Score {val:.4f} outside [0, 1]"
        print(f"  [{PASS_TAG}] {test}: score={val:.4f}")
    except TestFailure:
        raise
    except Exception:
        raise TestFailure(model_name, test, traceback.format_exc())


def test_single_video_score(scorer, model_name: str, video: str) -> None:
    test = "single_video_score"
    try:
        score = scorer(images=[video], texts=[VIDEO_TEXT])
        assert score.shape == (1, 1), f"Expected shape (1,1), got {score.shape}"
        val = score.item()
        assert 0.0 <= val <= 1.0, f"Score {val:.4f} outside [0, 1]"
        print(f"  [{PASS_TAG}] {test}: score={val:.4f}")
    except TestFailure:
        raise
    except Exception:
        raise TestFailure(model_name, test, traceback.format_exc())


def test_multi_pair_score(scorer, model_name: str, image: str, video: str) -> None:
    test = "multi_pair_score"
    try:
        scores = scorer(images=[image, video], texts=[IMAGE_TEXT, VIDEO_TEXT])
        assert scores.shape == (2, 2), f"Expected shape (2,2), got {scores.shape}"
        assert torch.all((scores >= 0) & (scores <= 1)), "Some scores outside [0, 1]"
        print(f"  [{PASS_TAG}] {test}: shape={tuple(scores.shape)}")
    except TestFailure:
        raise
    except Exception:
        raise TestFailure(model_name, test, traceback.format_exc())


def test_generate_from_image(scorer, model_name: str, image: str) -> None:
    test = "generate_from_image"
    try:
        responses = scorer.model.generate(images=[image], texts=[GEN_PROMPT])
        assert isinstance(responses, list) and len(responses) == 1
        text = responses[0]
        assert isinstance(text, str) and len(text.strip()) > 0, \
            f"Empty or non-string response: {repr(text)}"
        print(f"  [{PASS_TAG}] {test}: \"{text[:100].strip()}\"")
    except TestFailure:
        raise
    except Exception:
        raise TestFailure(model_name, test, traceback.format_exc())


def test_generate_from_video(scorer, model_name: str, video: str) -> None:
    test = "generate_from_video"
    try:
        responses = scorer.model.generate(images=[video], texts=[GEN_PROMPT])
        assert isinstance(responses, list) and len(responses) == 1
        text = responses[0]
        assert isinstance(text, str) and len(text.strip()) > 0, \
            f"Empty or non-string response: {repr(text)}"
        print(f"  [{PASS_TAG}] {test}: \"{text[:100].strip()}\"")
    except TestFailure:
        raise
    except Exception:
        raise TestFailure(model_name, test, traceback.format_exc())

# ---------------------------------------------------------------------------
# Core runner — loads model, runs all 5 tests, cleans up
# ---------------------------------------------------------------------------

def run_all_tests(model_name: str, scorer, image: str, video: str) -> None:
    allows_video = getattr(scorer.model, 'allows_video', True)  # default True if not set

    print(f"\n  [1/5] single_image_score")
    test_single_image_score(scorer, model_name, image)

    if allows_video:
        print(f"\n  [2/5] single_video_score")
        test_single_video_score(scorer, model_name, video)

        print(f"\n  [3/5] multi_pair_score  (2 media x 2 texts)")
        test_multi_pair_score(scorer, model_name, image, video)
    else:
        print(f"\n  [2/5] single_video_score  [SKIPPED — image-only model]")
        print(f"\n  [3/5] multi_pair_score    [SKIPPED — image-only model]")

    print(f"\n  [4/5] generate_from_image")
    test_generate_from_image(scorer, model_name, image)

    if allows_video:
        print(f"\n  [5/5] generate_from_video")
        test_generate_from_video(scorer, model_name, video)
    else:
        print(f"\n  [5/5] generate_from_video  [SKIPPED — image-only model]")

        
def _load_and_run(model_name: str, image: str, video: str, **kwargs) -> None:
    """Load a model via VQAScore, run all tests, clean up GPU memory."""
    print(f"\n{SEP}")
    print(f"  Loading: {model_name}")
    print(SEP)

    try:
        scorer = t2v_metrics.VQAScore(model=model_name, **kwargs)
    except Exception:
        raise TestFailure(model_name, "model_load", traceback.format_exc())

    try:
        run_all_tests(model_name, scorer, image, video)
    finally:
        del scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
# Group runners
# ---------------------------------------------------------------------------

def run_local_models(image: str, video: str) -> None:
    print("\n\n>>> LOCAL MODELS")
    for model_name in LOCAL_MODELS:
        _load_and_run(model_name, image, video)


def run_gpt_models(image: str, video: str) -> None:
    print("\n\n>>> GPT MODELS")
    if not os.environ.get("OPENAI_API_KEY"):
        raise TestFailure("gpt", "env_check", "OPENAI_API_KEY is not set.")
    for model_name in GPT_MODELS:
        _load_and_run(model_name, image, video)


def run_gemini_vertex(image: str, video: str) -> None:
    print("\n\n>>> GEMINI MODELS — Vertex AI backend")
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project_id:
        raise TestFailure("gemini-vertex", "env_check", "GOOGLE_CLOUD_PROJECT is not set.")
    for model_name in GEMINI_MODELS_TO_TEST:
        _load_and_run(model_name, image, video, project_id=project_id)


def run_gemini_api_key(image: str, video: str) -> None:
    print("\n\n>>> GEMINI MODELS — API key backend")
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise TestFailure("gemini-api", "env_check", "GEMINI_API_KEY is not set.")
    with suppress_env_var("GOOGLE_CLOUD_PROJECT"):
        for model_name in GEMINI_MODELS_TO_TEST:
            _load_and_run(model_name, image, video, api_key=api_key)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test VQAScore and generation. Stops on first failure."
    )
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--video", default=DEFAULT_VIDEO)
    parser.add_argument("--skip-local",        action="store_true")
    parser.add_argument("--skip-api",           action="store_true")
    parser.add_argument("--skip-gemini-vertex", action="store_true")
    parser.add_argument("--skip-gemini-api",    action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "=" * 60)
    print("  t2v_metrics — VQAScore & Generation Test Suite")
    print("  Stops immediately on first failure.")
    print("=" * 60)
    print(f"  Image : {args.image}")
    print(f"  Video : {args.video}")

    try:
        if not args.skip_local:
            run_local_models(args.image, args.video)

        if not args.skip_api:
            run_gpt_models(args.image, args.video)

            if not args.skip_gemini_vertex:
                run_gemini_vertex(args.image, args.video)

            if not args.skip_gemini_api:
                run_gemini_api_key(args.image, args.video)

    except TestFailure as e:
        print(f"\n{'=' * 60}")
        print(f"  [{FAIL_TAG}] STOPPED at: {e.model}  →  {e.test}")
        print(f"{'=' * 60}")
        print(e.message)
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print(f"  [{PASS_TAG}] ALL TESTS PASSED")
    print(f"{'=' * 60}\n")
    sys.exit(0)


if __name__ == "__main__":
    main()