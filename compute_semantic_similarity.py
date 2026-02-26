"""
Semantic Similarity operator using sentence-transformers.

Unlike character- or word-level metrics (CER, WER, ANLS) which measure
surface-form overlap, semantic similarity measures whether two strings carry
the same *meaning* regardless of how that meaning is expressed. It does this
by embedding both strings into a shared vector space using a pre-trained
sentence encoder and computing the cosine similarity between the resulting
vectors.

Example: "The car is fast" vs "The automobile is quick" scores near 1.0
despite sharing no words, because a good sentence encoder places semantically
equivalent sentences close together in embedding space.

Scores are cosine similarities clamped to [0, 1]:
- 1.0: semantically identical
- 0.0: completely unrelated (or cosine similarity was negative)

The sentence encoder model is loaded once per operator execution and shared
across all samples, so the per-sample cost is just a forward pass — the
dominant cost is the initial model download/load.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|

"""

import fiftyone.operators as foo
from fiftyone.operators import types

from .base import BaseTextEvaluationOperator


# Curated list of sentence-transformers models ordered roughly by speed.
# All are hosted on HuggingFace and downloaded automatically on first use.
# - all-MiniLM-L6-v2:  fastest, 384-dim, excellent quality/speed trade-off
# - all-MiniLM-L12-v2: slightly better than L6 at modest speed cost
# - all-mpnet-base-v2: highest quality general-purpose English model, slower
# - paraphrase-multilingual-MiniLM-L12-v2: multilingual support (50+ languages)
_SUPPORTED_MODELS = [
    "all-MiniLM-L6-v2",
    "all-MiniLM-L12-v2",
    "all-mpnet-base-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]

_DEFAULT_MODEL = "all-MiniLM-L6-v2"


def _handle_calling(
    uri,
    sample_collection,
    pred_field,
    gt_field,
    output_field,
    model_name,
    delegate,
):
    """Bridge between the Python __call__ API and foo.execute_operator.

    All keyword arguments accepted by __call__ are forwarded here so that
    programmatic callers (e.g. notebooks, scripts) use the exact same
    execution path as the FiftyOne App UI.
    """
    ctx = dict(dataset=sample_collection)
    params = dict(
        pred_field=pred_field,
        gt_field=gt_field,
        output_field=output_field,
        model_name=model_name,
    )
    return foo.execute_operator(
        uri,
        ctx,
        params=params,
        delegate=delegate,
    )


def _compute_semantic_scores(gt_strings, pred_strings, model_name):
    """Compute per-sample cosine semantic similarity scores.

    Strings are encoded in two batch calls (one for gt, one for pred) rather
    than sample-by-sample. Sentence Transformers handles batching internally
    with optimised padding and GPU utilisation when a GPU is available, so
    this is significantly faster than calling encode() in a loop.

    Cosine similarity values are clamped to [0, 1]. Negative cosine values
    are theoretically possible but extremely rare for natural language and
    would be meaningless as an evaluation metric, so we treat them as 0.0.

    Args:
        gt_strings (list[str]): Ground truth strings.
        pred_strings (list[str]): Predicted strings.
        model_name (str): HuggingFace model identifier for SentenceTransformer.

    Returns:
        list[float]: Per-sample semantic similarity scores in [0, 1].
    """
    from sentence_transformers import SentenceTransformer, util

    model = SentenceTransformer(model_name)

    # Encode all ground truth and prediction strings in two batched calls.
    # show_progress_bar=False keeps the operator output clean; progress is
    # already visible at the FiftyOne operator level.
    gt_embeddings = model.encode(gt_strings, show_progress_bar=False)
    pred_embeddings = model.encode(pred_strings, show_progress_bar=False)

    scores = []
    for gt_emb, pred_emb in zip(gt_embeddings, pred_embeddings):
        # util.cos_sim returns a 1×1 tensor; .item() converts it to a Python float
        cosine = util.cos_sim(gt_emb, pred_emb).item()
        # Clamp to [0, 1] — negative cosine similarity is not a meaningful
        # evaluation signal for text and confuses downstream aggregation
        scores.append(max(0.0, cosine))

    return scores


class ComputeSemanticSimilarity(BaseTextEvaluationOperator):
    """Compute semantic similarity between predicted and ground truth text fields.

    Uses a pre-trained sentence encoder to embed both strings and measures
    cosine similarity in the resulting vector space. This captures meaning
    rather than surface form, making it complementary to metrics like ANLS
    (which measures character-level overlap) and exact match.

    Use this metric when:
    - Predictions may paraphrase the ground truth rather than copy it verbatim
    - Surface-form metrics systematically underestimate model quality
    - You want a human-interpretable "how similar in meaning" score

    Scores are stored as a FloatField on each sample in the range [0, 1].
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_semantic_similarity",
            label="Compute Semantic Similarity",
            description=(
                "Compute semantic similarity using sentence-transformers "
                "embeddings and cosine similarity"
            ),
            icon="/assets/spell-check-svgrepo-com.svg",
            # dynamic=True re-runs resolve_input on each param change,
            # enabling the output_field default to update as pred_field is selected.
            dynamic=True,
            allow_immediate_execution=True,
            allow_delegated_execution=True,
        )

    def __call__(
        self,
        sample_collection,
        pred_field,
        gt_field,
        output_field=None,
        model_name=_DEFAULT_MODEL,
        delegate=False,
    ):
        """
        Compute semantic similarity for text fields.

        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the StringField containing model predictions
            gt_field (str): Name of the StringField containing ground truth text
            output_field (str, optional): Name for the FloatField that will store
                per-sample semantic similarity scores. Defaults to
                ``"{pred_field}_semantic_similarity"``
            model_name (str, optional): Sentence-transformers model to use.
                Must be one of the supported models or any valid HuggingFace
                model identifier. Defaults to ``"all-MiniLM-L6-v2"``
            delegate (bool, optional): If True, execution is delegated to a
                background task (recommended for large datasets since model
                loading and inference can be slow). Defaults to False

        Returns:
            dict: ``{"output_field": str, "samples_evaluated": int,
                     "mean_semantic_similarity": float}``
        """
        if output_field is None:
            output_field = f"{pred_field}_semantic_similarity"

        return _handle_calling(
            self.uri,
            sample_collection,
            pred_field,
            gt_field,
            output_field,
            model_name,
            delegate,
        )

    def resolve_input(self, ctx):
        """Build the operator input form shown in the FiftyOne App.

        The form has two sections:
        1. Field selectors (pred, gt, output) — provided by the base class.
           The output field default updates dynamically as pred_field changes.
        2. Model selection — a dropdown of curated sentence-transformer models
           ordered by speed. Users can weigh quality against inference time.
        """
        inputs = types.Object()

        # Render pred/gt/output field selectors. Returns False (and adds a
        # warning) if the dataset has no StringFields to choose from.
        if not self._build_field_inputs(inputs, ctx):
            return types.Property(inputs)

        # Override the output_field added by _build_field_inputs with a
        # dynamically computed default that reflects the chosen pred_field.
        pred_field = ctx.params.get("pred_field", "")
        default_output = (
            f"{pred_field}_semantic_similarity" if pred_field
            else "semantic_similarity"
        )
        inputs.str(
            "output_field",
            default=default_output,
        )

        inputs.view(
            "header",
            types.Header(label="Semantic Similarity Parameters", divider=True)
        )

        inputs.enum(
            "model_name",
            values=_SUPPORTED_MODELS,
            label="Sentence Encoder Model",
            description=(
                "Sentence-transformers model used to embed text. "
                "all-MiniLM-L6-v2 is the fastest with excellent quality. "
                "all-mpnet-base-v2 gives the highest quality at the cost of "
                "slower inference. The multilingual model supports 50+ languages."
            ),
            required=True,
            default=_DEFAULT_MODEL,
            view=types.AutocompleteView(
                choices=[types.Choice(value=m, label=m) for m in _SUPPORTED_MODELS]
            ),
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Compute and store semantic similarity scores for all samples in the view.

        The sentence encoder model is loaded once at the start of execution
        and both field arrays are encoded in batched calls before per-sample
        scores are computed. This avoids the overhead of repeated model
        initialisation and maximises GPU utilisation when available.
        """
        gt_strings, pred_strings, target_view = self._get_text_pairs(ctx)

        model_name = ctx.params.get("model_name", _DEFAULT_MODEL)
        output_field = ctx.params.get("output_field")

        scores = _compute_semantic_scores(gt_strings, pred_strings, model_name)

        mean_score = self._save_scores(ctx, target_view, output_field, scores)

        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "mean_semantic_similarity": mean_score,
        }
