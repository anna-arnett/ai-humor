import dspy
from load_data import load_examples

# Configure the model DSPy should use
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm)

# Load your Prompt V2 from file
with open("prompt_v2.txt", "r") as f:
    PROMPT_V2 = f.read()

# Define the DSPy signature
class HumorAnnotation(dspy.Signature):
    c_text: str = dspy.InputField(desc="preceding context or dialogue")
    x_text: str = dspy.InputField(desc="candidate humorous utterance")
    x_speaker: str = dspy.InputField(desc="speaker of X_text")
    y: int = dspy.InputField(desc="receiver reaction: 1=laughter, 0=no laughter")
    annotation_json: str = dspy.OutputField(desc="Return exactly one JSON object and nothing else.")

# Attach your prompt instructions
HumorAnnotation = HumorAnnotation.with_instructions(PROMPT_V2)

# Create the predictor
annotate = dspy.Predict(HumorAnnotation)

# Load your small labeled dataset
examples = load_examples("humor_examples_small.json")

print(f"Loaded {len(examples)} examples.\n")

# Run the baseline on every example
for ex in examples:
    result = annotate(
        c_text=ex.c_text,
        x_text=ex.x_text,
        x_speaker=ex.x_speaker,
        y=ex.y,
    )

    print("=" * 80)
    print(f"ID: {ex.id}")
    print(f"GOLD support label: {ex.gold_support_label}")
    print(f"GOLD flags: {ex.gold_flags}")
    print(f"GOLD hallucination flag: {ex.gold_hallucination_flag}")
    print("-" * 80)
    print(result.annotation_json)
    print()