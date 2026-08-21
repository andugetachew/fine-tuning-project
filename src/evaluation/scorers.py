"""
Pluggable scoring interface for evaluation. Any scorer implements
score(prediction, gold) -> {"score": float, "details": dict}.
Swapping metrics is a config change (evaluation.scorer_type), never
a change to the training or evaluation pipeline.
"""
from abc import ABC, abstractmethod


class BaseScorer(ABC):
    name: str

    @abstractmethod
    def score(self, prediction: str, gold: str, prompt: list[dict] | None = None) -> dict:
        raise NotImplementedError


class ExactMatchScorer(BaseScorer):
    """Strict match. Fit for structured/factual outputs with one correct answer."""
    name = "exact_match"

    def score(self, prediction: str, gold: str, prompt=None) -> dict:
        match = prediction.strip() == gold.strip()
        return {"score": 1.0 if match else 0.0, "details": {"exact": match}}


class NormalizedMatchScorer(BaseScorer):
    """Case/punctuation/whitespace-insensitive match. Fit for short factual
    answers where phrasing varies but the substance shouldn't."""
    name = "normalized_match"

    def score(self, prediction: str, gold: str, prompt=None) -> dict:
        def norm(s: str) -> list[str]:
            return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).split()
        match = norm(prediction) == norm(gold)
        return {"score": 1.0 if match else 0.0, "details": {"exact": match}}


class SemanticSimilarityScorer(BaseScorer):
    """Cosine similarity between prediction and gold via sentence-transformers.
    Fit for open-ended technical explanations where exact wording varies
    but the substance should match. Score is 0.0-1.0."""
    name = "semantic_similarity"
    _model = None

    def _get_model(self):
        if SemanticSimilarityScorer._model is None:
            from sentence_transformers import SentenceTransformer
            SemanticSimilarityScorer._model = SentenceTransformer("all-MiniLM-L6-v2")
        return SemanticSimilarityScorer._model

    def score(self, prediction: str, gold: str, prompt=None) -> dict:
        from sentence_transformers import util
        model = self._get_model()
        emb = model.encode([prediction, gold], convert_to_tensor=True)
        similarity = float(util.cos_sim(emb[0], emb[1])[0][0])
        return {"score": similarity, "details": {"metric": "cosine_similarity"}}


class LLMJudgeScorer(BaseScorer):
    """Uses an LLM to rate whether a prediction matches the gold answer's
    substance, on a 0.0-1.0 scale. Requires ANTHROPIC_API_KEY in the
    environment."""
    name = "llm_judge"

    JUDGE_PROMPT = """You are grading a backend engineering assistant's answer against a reference answer.

Question context: {prompt}

Reference answer: {gold}

Candidate answer: {prediction}

Rate how well the candidate answer captures the same root cause and fix as the reference answer, on a scale of 0.0 to 1.0:
- 1.0 = correctly identifies the same root cause and fix, even if worded differently
- 0.5 = partially correct, misses key details or gives a plausible but incomplete diagnosis
- 0.0 = wrong root cause, wrong fix, or generic non-answer

Respond with ONLY a single number between 0.0 and 1.0, nothing else."""

    def score(self, prediction: str, gold: str, prompt=None) -> dict:
        import os

        try:
            import anthropic
        except ImportError:
            raise NotImplementedError(
                "llm_judge scoring requires the 'anthropic' package (pip install anthropic) "
                "and ANTHROPIC_API_KEY to be set."
            )

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise NotImplementedError(
                "llm_judge scoring requires ANTHROPIC_API_KEY to be set in the environment."
            )

        prompt_text = ""
        if prompt:
            user_turns = [m["content"] for m in prompt if m.get("role") == "user"]
            prompt_text = " ".join(user_turns)

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=10,
            messages=[{
                "role": "user",
                "content": self.JUDGE_PROMPT.format(prompt=prompt_text, gold=gold, prediction=prediction),
            }],
        )
        raw_score = response.content[0].text.strip()
        try:
            score = float(raw_score)
        except ValueError:
            score = 0.0
        score = max(0.0, min(1.0, score))
        return {"score": score, "details": {"metric": "llm_judge", "model": "claude-haiku-4-5"}}


class AIEvalFrameworkScorer(BaseScorer):
    """Delegates scoring to the separate ai-evaluation-framework project's
    deployed API, using its semantic_similarity scorer. Requires
    AI_EVAL_FRAMEWORK_URL (defaults to the deployed Render instance) and,
    if that service's API_KEY auth is enabled, AI_EVAL_FRAMEWORK_API_KEY."""
    name = "ai_eval_framework"

    DEFAULT_URL = "https://ai-eval-framework-35y8.onrender.com"

    def score(self, prediction: str, gold: str, prompt=None) -> dict:
        import os

        try:
            import requests
        except ImportError:
            raise NotImplementedError(
                "ai_eval_framework scoring requires the 'requests' package (pip install requests)."
            )

        base_url = os.environ.get("AI_EVAL_FRAMEWORK_URL", self.DEFAULT_URL)
        api_key = os.environ.get("AI_EVAL_FRAMEWORK_API_KEY")

        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key

        input_text = ""
        if prompt:
            user_turns = [m["content"] for m in prompt if m.get("role") == "user"]
            input_text = " ".join(user_turns)

        try:
            response = requests.post(
                f"{base_url}/eval-runs",
                json={
                    "name": "fine-tuning-project-eval",
                    "scorers": ["semantic_similarity"],
                    "items": [{
                        "id": "1",
                        "input": input_text,
                        "actual_output": prediction,
                        "expected_output": gold,
                    }],
                },
                headers=headers,
                timeout=60,  # free-tier Render can take ~50s to wake from cold start
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            raise NotImplementedError(
                f"ai_eval_framework request failed ({e}). "
                "Confirm the service is reachable and AI_EVAL_FRAMEWORK_URL/API_KEY are correct."
            )

        result = response.json()
        results_list = result.get("results", [])
        if not results_list:
            raise NotImplementedError("ai_eval_framework returned no results for the submitted item.")
        score = results_list[0].get("score", 0.0)
        return {"score": float(score), "details": {"metric": "ai_eval_framework", "backend": base_url}}


SCORER_REGISTRY: dict[str, type[BaseScorer]] = {
    "exact_match": ExactMatchScorer,
    "normalized_match": NormalizedMatchScorer,
    "semantic_similarity": SemanticSimilarityScorer,
    "llm_judge": LLMJudgeScorer,
    "ai_eval_framework": AIEvalFrameworkScorer,
}


def get_scorer(scorer_type: str) -> BaseScorer:
    if scorer_type not in SCORER_REGISTRY:
        raise ValueError(f"Unknown scorer_type '{scorer_type}'. Options: {list(SCORER_REGISTRY)}")
    return SCORER_REGISTRY[scorer_type]()