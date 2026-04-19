import json
import dspy

VALID_SUPPORT_LABELS = {
    "text_sufficient",
    "text_insufficient_possible_multimodal",
    "text_insufficient_missing_context",
    "text_insufficient_possible_transcription_error",
}

VALID_FLAGS = {
    "missing_context",
    "possible_bad_transcription",
    "possible_speaker_misattribution",
    "possible_truncation",
    "possible_alignment_error",
    "possible_multimodal_dependence",
}


def metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    # 1) Parse JSON safely
    try:
        obj = json.loads(pred.annotation_json)
    except Exception as e:
        result = {
            "score": 0.0,
            "feedback": f"Invalid JSON output: {e}. Return exactly one valid JSON object."
        }
        return dspy.Prediction(**result) if pred_name else 0.0

    score = 0.0
    feedback = []

    # 2) text_support_label (highest weight)
    pred_label = obj.get("text_support_label")
    gold_label = example.gold_support_label

    if pred_label not in VALID_SUPPORT_LABELS:
        feedback.append(
            f"text_support_label is invalid: {pred_label}. Use one of {sorted(VALID_SUPPORT_LABELS)}."
        )
    elif pred_label == gold_label:
        score += 0.55
    else:
        feedback.append(
            f"text_support_label mismatch: predicted {pred_label}, expected {gold_label}. "
            f"Gold note: {example.gold_note}"
        )

    # 3) data_quality_flags (partial credit)
    raw_pred_flags = obj.get("data_quality_flags", [])
    if not isinstance(raw_pred_flags, list):
        raw_pred_flags = []
        feedback.append("data_quality_flags must be a list.")

    pred_flags = {flag for flag in raw_pred_flags if flag in VALID_FLAGS}
    invalid_flags = [flag for flag in raw_pred_flags if flag not in VALID_FLAGS]
    gold_flags = set(example.gold_flags)

    if invalid_flags:
        feedback.append(
            f"Invalid data_quality_flags: {invalid_flags}. Use only allowed flags."
        )

    if len(gold_flags | pred_flags) == 0:
        flag_score = 1.0
    else:
        flag_score = len(gold_flags & pred_flags) / len(gold_flags | pred_flags)

    score += 0.20 * flag_score

    if pred_flags != gold_flags:
        feedback.append(
            f"data_quality_flags mismatch: predicted {sorted(pred_flags)}, expected {sorted(gold_flags)}."
        )

    # 4) hallucination_flag
    pred_hflag = obj.get("hallucination_flag")
    gold_hflag = example.gold_hallucination_flag

    if pred_hflag == gold_hflag:
        score += 0.25
    else:
        feedback.append(
            f"hallucination_flag mismatch: predicted {pred_hflag}, expected {gold_hflag}. "
            f"Gold note: {example.gold_note}"
        )

    # 5) Final packaging
    score = max(0.0, min(score, 1.0))

    result = {
        "score": score,
        "feedback": "Correct." if not feedback else " ".join(feedback)
    }

    return dspy.Prediction(**result) if pred_name else score