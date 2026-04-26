import dspy
from load_data import load_examples
from metric import metric

student_lm = dspy.LM("openai/gpt-4o-mini", temperature=0.2)
dspy.configure(lm=student_lm)

with open("prompt_v1_weaker.txt", "r") as f:
    PROMPT_V1 = f.read()

with open("prompt_v2_projectformat.txt", "r") as f:
    PROMPT_V2 = f.read()

def build_program(prompt_text):
    class HumorAnnotation(dspy.Signature):
        c_text: str = dspy.InputField(desc="preceding context or dialogue")
        x_text: str = dspy.InputField(desc="candidate humorous utterance")
        x_speaker: str = dspy.InputField(desc="speaker of X_text")
        y: int = dspy.InputField(desc="receiver reaction: 1=laughter, 0=no laughter")
        annotation_json: str = dspy.OutputField(desc="Return exactly one JSON object and nothing else.")

    Sig = HumorAnnotation.with_instructions(prompt_text)
    return dspy.Predict(Sig)

testset = load_examples("humor_examples.json")

print(f"Test size: {len(testset)}")

program_v1 = build_program(PROMPT_V1)
program_v2 = build_program(PROMPT_V2)

evaluate_test = dspy.Evaluate(
    devset=testset,
    metric=metric,
    num_threads=1,
    display_progress=True,
    display_table=False,
)

v1_result = evaluate_test(program_v1)
print(f"\nPROMPT V1 TEST SCORE: {v1_result.score:.3f}")

v2_result = evaluate_test(program_v2)
print(f"PROMPT V2 TEST SCORE: {v2_result.score:.3f}")