import json
import dspy

def load_examples(path):
    with open(path, "r") as f:
        raw = json.load(f)

    examples = []
    for row in raw:
        c_text = "\n".join(row["C_text"]) if isinstance(row["C_text"], list) else row["C_text"]

        ex = dspy.Example(
            id=row["id"],
            c_text=c_text,
            x_text=row["X_text"],
            x_speaker=row["X_speaker"],
            y=row["Y"],
            gold_support_label=row["gold_support_label"],
            gold_flags=row["gold_flags"],
            gold_hallucination_flag=row["gold_hallucination_flag"],
            gold_note=row["gold_note"],
        ).with_inputs("c_text", "x_text", "x_speaker", "y")

        examples.append(ex)

    return examples