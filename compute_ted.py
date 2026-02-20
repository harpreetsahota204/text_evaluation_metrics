"""
Tree Edit Distance (TED) operator for structured text (JSON, HTML, Markdown).

TED measures structural similarity between two tree-structured documents by
computing the minimum number of node-level edit operations (insert, delete,
relabel) needed to transform one tree into the other. It is the natural
extension of Levenshtein distance to hierarchical data and is the standard
metric for evaluating structured OCR output such as JSON, HTML, and Markdown.

The raw TED value is unbounded (it grows with tree size), so we normalize it:

    TED_similarity = 1 - TED(T1, T2) / max(|T1|, |T2|)

This produces a score in [0, 1] where 1.0 is a perfect structural match and
0.0 means no shared structure at all.

When a string field cannot be parsed into a tree (e.g. the model produced
malformed JSON), the sample receives a score of 0.0. This is intentional:
failure to produce a valid structured output is itself a meaningful evaluation
signal and should not be silently skipped.

| Copyright 2017-2025, Voxel51, Inc.
| `voxel51.com <https://voxel51.com/>`_
|

"""

import json
import re

import fiftyone.operators as foo
from fiftyone.operators import types

from .base import BaseTextEvaluationOperator


def _handle_calling(
    uri,
    sample_collection,
    pred_field,
    gt_field,
    output_field,
    format,
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
        format=format,
    )
    return foo.execute_operator(
        uri,
        ctx,
        params=params,
        delegate=delegate,
    )


# ---------------------------------------------------------------------------
# APTED node and configuration
# ---------------------------------------------------------------------------

class _Node:
    """Lightweight tree node compatible with the apted library interface.

    The apted library only requires that nodes expose a ``name`` attribute
    (used as the node label for rename-cost computation) and a ``children``
    attribute (an ordered list of child nodes). No other interface is needed.
    """

    def __init__(self, name, children=None):
        self.name = name
        self.children = children or []


class _AptedConfig:
    """Cost model and tree-traversal config for the APTED algorithm.

    APTED's default Config class uses Python object identity for rename cost,
    which would always return 1 for our custom _Node objects even when their
    labels are equal. We override both methods explicitly:

    - ``rename``: 0 if node labels are identical strings, 1 otherwise.
      All three edit operations (insert, delete, relabel) therefore have
      uniform unit cost, which is the standard TED formulation.
    - ``children``: returns the ordered child list for tree traversal.
    """

    def rename(self, node1, node2):
        return 0 if node1.name == node2.name else 1

    def children(self, node):
        return node.children


def _count_nodes(node):
    """Return the total number of nodes in a tree rooted at ``node``.

    Used to compute the normalization denominator: max(|T1|, |T2|).
    """
    return 1 + sum(_count_nodes(c) for c in node.children)


# ---------------------------------------------------------------------------
# JSON tree builder
# ---------------------------------------------------------------------------

# Matches optional language tag after the opening fence, e.g. ```json ... ```
_FENCE_RE = re.compile(r"^```[\w]*\n?(.*?)\n?```$", re.DOTALL)


def _strip_fences(s):
    """Remove markdown code fences from a string if present.

    VLMs frequently wrap JSON output in ```json ... ``` blocks. Stripping
    these before calling json.loads avoids spurious parse failures on
    otherwise valid model output.
    """
    m = _FENCE_RE.match(s.strip())
    return m.group(1).strip() if m else s.strip()


def _json_to_node(obj):
    """Recursively convert a Python object (from json.loads) into a _Node tree.

    The mapping is:
    - dict  → internal node labeled "{}" whose children are key nodes.
              Keys are sorted alphabetically before building children so that
              {"a": 1, "b": 2} and {"b": 2, "a": 1} produce identical trees.
              JSON objects are semantically unordered; sorting normalizes away
              incidental key-order differences between prediction and ground truth.
    - list  → internal node labeled "[]" whose children are the element nodes
              in their original order (list order is semantically meaningful).
    - scalar → leaf node whose label is str(value). Type information is
              preserved implicitly: True → "True", 1 → "1", "1" → "1".
              This means a type mismatch (bool vs int) costs 0, which is an
              acceptable trade-off for OCR evaluation where the focus is
              structural fidelity rather than type-system correctness.
    """
    if isinstance(obj, dict):
        node = _Node("{}")
        for key in sorted(obj.keys()):
            # Each dict key becomes an intermediate node so that structural
            # differences in keys (e.g. "name" vs "Name") cost exactly 1
            # relabel operation rather than being invisible.
            key_node = _Node(str(key), children=[_json_to_node(obj[key])])
            node.children.append(key_node)
        return node
    if isinstance(obj, list):
        node = _Node("[]")
        node.children = [_json_to_node(item) for item in obj]
        return node
    # Scalar value — leaf node
    return _Node(str(obj))


def _parse_json(s):
    """Parse a JSON string into a _Node tree, returning None on failure.

    Strips markdown fences first so that models that wrap their output in
    ```json ... ``` blocks are handled gracefully. Any parse error (malformed
    JSON, empty string, non-JSON content) returns None, which the caller
    converts to a score of 0.0.
    """
    try:
        return _json_to_node(json.loads(_strip_fences(s)))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# HTML tree builder
# ---------------------------------------------------------------------------

def _bs4_to_node(element):
    """Recursively convert a BeautifulSoup element into a _Node tree.

    - HTML tags (Tag) become internal nodes labeled with the tag name (e.g.
      "div", "table", "td"). Attributes are ignored; only structure and text
      content are compared.
    - Text nodes (NavigableString) become leaf nodes labeled with their
      stripped text content. Empty or whitespace-only text nodes are skipped
      to avoid penalising insignificant whitespace differences between the
      prediction and ground truth.

    bs4 and NavigableString are imported lazily so that this module can be
    imported without beautifulsoup4 installed as long as the HTML format is
    not actually used.
    """
    from bs4 import NavigableString, Tag
    if isinstance(element, NavigableString):
        text = str(element).strip()
        # Skip whitespace-only text nodes (e.g. newlines between tags)
        return _Node(text) if text else None
    if isinstance(element, Tag):
        node = _Node(element.name)
        for child in element.children:
            child_node = _bs4_to_node(child)
            if child_node is not None:
                node.children.append(child_node)
        return node
    return None


def _parse_html(s):
    """Parse an HTML string into a _Node tree, returning None on failure.

    Uses BeautifulSoup with Python's built-in html.parser backend (no C
    dependencies required). A synthetic "document" root node is added so
    that fragments without a single root element (e.g. "<p>a</p><p>b</p>")
    still produce a well-formed tree.
    """
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(s.strip(), "html.parser")
        root = _Node("document")
        for child in soup.children:
            child_node = _bs4_to_node(child)
            if child_node is not None:
                root.children.append(child_node)
        return root
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Markdown tree builder
# ---------------------------------------------------------------------------

def _mistletoe_to_node(node):
    """Recursively convert a mistletoe AST node into a _Node tree.

    mistletoe represents Markdown as a typed AST. Each node type maps to a
    distinct Python class (e.g. Heading, Paragraph, List, RawText). We use
    the class name as the node label so that structural differences (e.g.
    a Heading where a Paragraph was expected) register as relabel costs.

    Leaf nodes such as RawText carry their string content in a ``content``
    attribute rather than a ``children`` list. For these we use the content
    string itself as the label so that text differences are captured directly.
    If content is empty, we fall back to the class name to ensure the node
    is still present in the tree.

    mistletoe is imported lazily so that this module can be imported without
    it installed as long as the Markdown format is not actually used.
    """
    label = type(node).__name__
    # Leaf node: has string content but no children
    if (
        hasattr(node, "content")
        and isinstance(node.content, str)
        and not getattr(node, "children", None)
    ):
        return _Node(node.content.strip() or label)

    apted_node = _Node(label)
    children_src = getattr(node, "children", None) or []
    for child in children_src:
        child_node = _mistletoe_to_node(child)
        if child_node is not None:
            apted_node.children.append(child_node)
    return apted_node


def _parse_markdown(s):
    """Parse a Markdown string into a _Node tree, returning None on failure.

    mistletoe.Document is the AST root; it contains block-level elements
    (headings, paragraphs, lists, code blocks, etc.) which in turn contain
    inline elements (emphasis, links, raw text, etc.).
    """
    try:
        import mistletoe
        doc = mistletoe.Document(s.strip())
        return _mistletoe_to_node(doc)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Unified TED computation
# ---------------------------------------------------------------------------

# Maps the user-facing format name to the corresponding parse function.
_PARSERS = {
    "json": _parse_json,
    "html": _parse_html,
    "markdown": _parse_markdown,
}


def _compute_ted_similarity(gt, pred, fmt):
    """Compute normalized TED similarity between two structured strings.

    Steps:
    1. Parse both strings into _Node trees using the format-appropriate parser.
    2. If either string fails to parse, return 0.0. A parse failure means the
       model did not produce a valid structured output, which is the worst
       possible outcome and should be scored accordingly.
    3. Run the APTED algorithm to compute the raw edit distance.
    4. Normalize by max(|T1|, |T2|) so the score is in [0, 1] regardless of
       document size.

    Args:
        gt (str): Ground truth structured string.
        pred (str): Predicted structured string.
        fmt (str): One of "json", "html", "markdown".

    Returns:
        float: TED similarity in [0.0, 1.0].
    """
    from apted import APTED

    parse = _PARSERS[fmt]
    gt_tree = parse(gt)
    pred_tree = parse(pred)

    # Either string could not be parsed — treat as maximally wrong
    if gt_tree is None or pred_tree is None:
        return 0.0

    distance = APTED(gt_tree, pred_tree, _AptedConfig()).compute_edit_distance()
    max_nodes = max(_count_nodes(gt_tree), _count_nodes(pred_tree))

    # Both trees are empty (e.g. two empty JSON objects "{}" reduce to a
    # single root node, so this branch is only hit for truly degenerate input)
    if max_nodes == 0:
        return 1.0

    return max(0.0, 1.0 - distance / max_nodes)


# ---------------------------------------------------------------------------
# FiftyOne Operator
# ---------------------------------------------------------------------------

class ComputeTED(BaseTextEvaluationOperator):
    """Compute Tree Edit Distance (TED) similarity for structured text fields.

    TED is the appropriate evaluation metric when OCR or VLM models are
    expected to output hierarchical data (JSON, HTML, Markdown) rather than
    plain text. Unlike character- or word-level metrics, TED captures
    structural correctness: a prediction that has the right words but the
    wrong nesting is penalised more than a prediction with a minor typo in
    an otherwise correct structure.

    Scores are stored as a FloatField on each sample in the range [0, 1]:
    - 1.0: the predicted structure is identical to the ground truth
    - 0.0: the structures share nothing, or the prediction could not be parsed
    """

    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_ted",
            label="Compute TED Similarity",
            description=(
                "Compute Tree Edit Distance similarity for structured text "
                "(JSON, HTML, or Markdown)"
            ),
            icon="/assets/spell-check-svgrepo-com.svg",
            # dynamic=True causes resolve_input to re-run whenever the user
            # changes any parameter, enabling the output_field default to
            # update live as pred_field is selected.
            dynamic=True,
        )

    def __call__(
        self,
        sample_collection,
        pred_field,
        gt_field,
        output_field=None,
        format="json",
        delegate=False,
    ):
        """
        Compute TED similarity for structured text fields.

        Args:
            sample_collection: A FiftyOne dataset or view
            pred_field (str): Name of the StringField containing model
                predictions (expected to hold structured text)
            gt_field (str): Name of the StringField containing ground truth
                structured text
            output_field (str, optional): Name for the FloatField that will
                store the per-sample TED similarity scores. Defaults to
                ``"{pred_field}_ted"``
            format (str, optional): The structured format of both fields.
                Must be one of ``"json"``, ``"html"``, or ``"markdown"``.
                Defaults to ``"json"``
            delegate (bool, optional): If True, execution is delegated to a
                background task (useful for large datasets). Defaults to False

        Returns:
            dict: ``{"output_field": str, "samples_evaluated": int,
                     "mean_ted_similarity": float}``
        """
        if output_field is None:
            output_field = f"{pred_field}_ted"

        return _handle_calling(
            self.uri,
            sample_collection,
            pred_field,
            gt_field,
            output_field,
            format,
            delegate,
        )

    def resolve_input(self, ctx):
        """Build the operator input form shown in the FiftyOne App.

        The form has two sections:
        1. Field selectors (pred, gt, output) — provided by the base class.
           The output field default is updated dynamically based on the
           selected pred_field.
        2. TED parameters — a radio group for the structured format. The
           choice here determines which parser is used at execution time.
        """
        inputs = types.Object()

        # Render pred/gt/output field selectors. Returns False (and adds a
        # warning view) if the dataset has no StringFields to choose from.
        if not self._build_field_inputs(inputs, ctx):
            return types.Property(inputs)

        # Override the output_field entry added by _build_field_inputs with
        # a dynamically computed default that reflects the chosen pred_field.
        pred_field = ctx.params.get("pred_field", "")
        default_output = f"{pred_field}_ted" if pred_field else "ted"
        inputs.str(
            "output_field",
            default=default_output,
        )

        inputs.view(
            "header",
            types.Header(label="TED Parameters", divider=True)
        )

        inputs.enum(
            "format",
            values=["json", "html", "markdown"],
            label="Structured Format",
            description=(
                "The structured format of the text fields. "
                "JSON fields may be wrapped in markdown code fences and will "
                "be stripped automatically."
            ),
            required=True,
            default="json",
            view=types.RadioGroupView(),
        )

        return types.Property(inputs)

    def execute(self, ctx):
        """Compute and store TED similarity scores for all samples in the view.

        Retrieves the (gt, pred) string pairs from the target view (which
        respects any active filters or sample selections in the App), computes
        a TED similarity score per sample, bulk-writes the scores back to the
        dataset, and returns aggregate statistics.
        """
        gt_strings, pred_strings, target_view = self._get_text_pairs(ctx)

        fmt = ctx.params.get("format", "json")
        output_field = ctx.params.get("output_field")

        scores = [
            _compute_ted_similarity(gt, pred, fmt)
            for gt, pred in zip(gt_strings, pred_strings)
        ]

        mean_score = self._save_scores(ctx, target_view, output_field, scores)

        return {
            "output_field": output_field,
            "samples_evaluated": len(target_view),
            "mean_ted_similarity": mean_score,
        }
