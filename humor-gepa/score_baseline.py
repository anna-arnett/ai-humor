import json
import dspy
from load_data import load_examples
from metric import metric

# Configure the model DSPy should use
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load prompt
with open("prompt_v2.txt", "r") as f:
    PROMPT_V2 = f.read()

# Signature
class HumorAnnotation(dspy.Signature):
    c_text: str = dspy.InputField(desc="preceding context or dialogue")
    x_text: str = dspy.InputField(desc="candidate humorous utterance")
    x_speaker: str = dspy.InputField(desc="speaker of X_text")
    y: int = dspy.InputField(desc="receiver reaction: 1=laughter, 0=no laughter")
    annotation_json: str = dspy.OutputField(desc="Return exactly one JSON object and nothing else.")

HumorAnnotation = HumorAnnotation.with_instructions(PROMPT_V2)

annotate = dspy.Predict(HumorAnnotation)

examples = load_examples("humor_examples.json")

total_score = 0.0

for ex in examples:
    pred = annotate(
        c_text=ex.c_text,
        x_text=ex.x_text,
        x_speaker=ex.x_speaker,
        y=ex.y,
    )

    scored = metric(ex, pred, pred_name="annotation")

    print("=" * 80)
    print(f"ID: {ex.id}")
    print(f"SCORE: {scored.score:.2f}")
    print(f"FEEDBACK: {scored.feedback}")
    print("-" * 80)
    print(pred.annotation_json)
    print()

    total_score += scored.score

avg_score = total_score / len(examples)
print("=" * 80)
print(f"AVERAGE SCORE: {avg_score:.3f}")