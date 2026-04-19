import random
import dspy

from load_data import load_examples
from metric import metric

# ----------------------------
# 1) Configure models
# ----------------------------
# Student model: the model being optimized
student_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.2)

# Reflection model: GEPA uses this model to reflect on failures and propose better instructions
# For a first cheap smoke test, keep this the same as the student model.
# Later, you can swap this to a stronger model.
reflection_lm = dspy.LM("openai/gpt-4o-mini", temperature=1.0, max_tokens=16000)

dspy.configure(lm=student_lm)

# ----------------------------
# 2) Load prompt
# ----------------------------
with open("prompt_v2.txt", "r") as f:
    PROMPT_V2 = f.read()

# ----------------------------
# 3) Define the program
# ----------------------------
class HumorAnnotation(dspy.Signature):
    c_text: str = dspy.InputField(desc="preceding context or dialogue")
    x_text: str = dspy.InputField(desc="candidate humorous utterance")
    x_speaker: str = dspy.InputField(desc="speaker of X_text")
    y: int = dspy.InputField(desc="receiver reaction: 1=laughter, 0=no laughter")
    annotation_json: str = dspy.OutputField(desc="Return exactly one JSON object and nothing else.")

HumorAnnotation = HumorAnnotation.with_instructions(PROMPT_V2)

program = dspy.Predict(HumorAnnotation)

# ----------------------------
# 4) Load and split data
# ----------------------------
examples = load_examples("humor_examples_small.json")

# Deterministic shuffle so your run is reproducible
random.Random(0).shuffle(examples)

# For this 5-example debug run:
# 3 train, 2 val
trainset = examples[:3]
valset = examples[3:]

print(f"Loaded {len(examples)} examples.")
print(f"Train size: {len(trainset)}")
print(f"Val size: {len(valset)}")

# ----------------------------
# 5) Baseline evaluation
# ----------------------------
evaluate_all = dspy.Evaluate(
    devset=examples,
    metric=metric,
    num_threads=1,
    display_progress=True,
    display_table=False,
)

baseline_result = evaluate_all(program)
print("\n" + "=" * 80)
print(f"BASELINE SCORE ON ALL 5: {baseline_result.score:.3f}")

# ----------------------------
# 6) Run GEPA
# ----------------------------
optimizer = dspy.GEPA(
    metric=metric,
    auto="light",
    reflection_lm=reflection_lm,
    track_stats=True,
    reflection_minibatch_size=2,
)

optimized_program = optimizer.compile(
    program,
    trainset=trainset,
    valset=valset,
)

# ----------------------------
# 7) Evaluate optimized program
# ----------------------------
optimized_result = evaluate_all(optimized_program)
print("\n" + "=" * 80)
print(f"OPTIMIZED SCORE ON ALL 5: {optimized_result.score:.3f}")

# ----------------------------
# 8) Per-example comparison
# ----------------------------
print("\n" + "=" * 80)
print("PER-EXAMPLE COMPARISON")
print("=" * 80)

for ex in examples:
    base_pred = program(
        c_text=ex.c_text,
        x_text=ex.x_text,
        x_speaker=ex.x_speaker,
        y=ex.y,
    )
    opt_pred = optimized_program(
        c_text=ex.c_text,
        x_text=ex.x_text,
        x_speaker=ex.x_speaker,
        y=ex.y,
    )

    base_score = metric(ex, base_pred)
    opt_score = metric(ex, opt_pred)

    print("-" * 80)
    print(f"ID: {ex.id}")
    print(f"Gold label: {ex.gold_support_label}")
    print(f"Baseline score:  {base_score:.2f}")
    print(f"Optimized score: {opt_score:.2f}")
    print()

    print("BASELINE OUTPUT:")
    print(base_pred.annotation_json)
    print()
    print("OPTIMIZED OUTPUT:")
    print(opt_pred.annotation_json)
    print()

# ----------------------------
# 9) Optional save
# ----------------------------
optimized_program.save("optimized_small_gepa.json")
print("\nSaved optimized program to optimized_small_gepa.json")