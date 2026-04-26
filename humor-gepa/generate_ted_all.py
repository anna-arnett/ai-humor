import json
import os
from pathlib import Path

import dspy

IN_PATH = "ted_converted_all.json"
OUT_PATH = "ted_generated_all.json"
PROMPT_PATH = "prompt_v2_projectformat.txt"

lm = dspy.LM(
    "openai/gpt-4o-mini",
    api_key=os.environ["OPENAI_API_KEY"],
    temperature=0.2,
)
dspy.configure(lm=lm)

PROMPT_V2 = Path(PROMPT_PATH).read_text()

class HumorAnnotation(dspy.Signature):
    c_text: str = dspy.InputField(desc="preceding context or dialogue")
    x_text: str = dspy.InputField(desc="candidate humorous utterance")
    x_speaker: str = dspy.InputField(desc="speaker of X_text")
    y: int = dspy.InputField(desc="receiver reaction: 1=laughter, 0=no laughter")
    annotation_json: str = dspy.OutputField(desc="Return exactly one JSON object and nothing else.")

HumorAnnotation = HumorAnnotation.with_instructions(PROMPT_V2)
annotate = dspy.Predict(HumorAnnotation)

data = json.loads(Path(IN_PATH).read_text())

# Resume support
if Path(OUT_PATH).exists():
    results = json.loads(Path(OUT_PATH).read_text())
else:
    results = []

done_ids = {str(r["id"]) for r in results}

for i, row in enumerate(data, start=1):
    if str(row["id"]) in done_ids:
        continue

    c_text = "\n".join(row["C_text"]) if isinstance(row["C_text"], list) else str(row["C_text"])

    try:
        pred = annotate(
            c_text=c_text,
            x_text=row["X_text"],
            x_speaker=row["X_speaker"],
            y=row["Y"],
        )

        parsed = json.loads(pred.annotation_json)

        results.append({
            "id": row["id"],
            "source": row["source"],
            "C_text": row["C_text"],
            "X_text": row["X_text"],
            "X_speaker": row["X_speaker"],
            "Y": row["Y"],
            "model_output": parsed
        })

        print(f"[{len(results)}/{len(data)}] OK - id={row['id']}")

    except Exception as e:
        results.append({
            "id": row["id"],
            "source": row["source"],
            "C_text": row["C_text"],
            "X_text": row["X_text"],
            "X_speaker": row["X_speaker"],
            "Y": row["Y"],
            "error": str(e)
        })

        print(f"[{len(results)}/{len(data)}] ERROR - id={row['id']} - {e}")

    Path(OUT_PATH).write_text(json.dumps(results, indent=2))

print(f"Saved outputs to {OUT_PATH}")