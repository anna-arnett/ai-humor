import json
import random
import statistics
from collections import defaultdict

import dspy

from load_data import load_examples
from metric import metric


# ----------------------------
# 1) Config
# ----------------------------
SEEDS = [0, 1, 2, 3, 4]

STUDENT_MODEL = "openai/gpt-4o-mini"
REFLECTION_MODEL = "openai/gpt-4o-mini"

TRAIN_SIZE = 20
VAL_SIZE = 5
TEST_SIZE = 6

student_lm = dspy.LM(STUDENT_MODEL, temperature=0.2)
reflection_lm = dspy.LM(REFLECTION_MODEL, temperature=1.0, max_tokens=16000)

dspy.configure(lm=student_lm)

with open("prompt_v2.txt", "r") as f:
    PROMPT_V2 = f.read()


# ----------------------------
# 2) Build fresh program each run
# ----------------------------
def build_program():
    class HumorAnnotation(dspy.Signature):
        c_text: str = dspy.InputField(desc="preceding context or dialogue")
        x_text: str = dspy.InputField(desc="candidate humorous utterance")
        x_speaker: str = dspy.InputField(desc="speaker of X_text")
        y: int = dspy.InputField(desc="receiver reaction: 1=laughter, 0=no laughter")
        annotation_json: str = dspy.OutputField(desc="Return exactly one JSON object and nothing else.")

    AnnotSig = HumorAnnotation.with_instructions(PROMPT_V2)
    return dspy.Predict(AnnotSig)


# ----------------------------
# 3) Split helper
# ----------------------------
def split_examples(examples, seed):
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)

    trainset = shuffled[:TRAIN_SIZE]
    valset = shuffled[TRAIN_SIZE:TRAIN_SIZE + VAL_SIZE]
    testset = shuffled[TRAIN_SIZE + VAL_SIZE:TRAIN_SIZE + VAL_SIZE + TEST_SIZE]

    return trainset, valset, testset


# ----------------------------
# 4) Category scoring helper
# ----------------------------
def score_by_label(program, dataset):
    bucket_scores = defaultdict(list)

    for ex in dataset:
        pred = program(
            c_text=ex.c_text,
            x_text=ex.x_text,
            x_speaker=ex.x_speaker,
            y=ex.y,
        )
        s = metric(ex, pred)
        bucket_scores[ex.gold_support_label].append(s)

    return {
        label: sum(scores) / len(scores)
        for label, scores in bucket_scores.items()
    }


# ----------------------------
# 5) Run one seed
# ----------------------------
def run_one_seed(seed, all_examples):
    print("\n" + "=" * 80)
    print(f"RUNNING SEED {seed}")
    print("=" * 80)

    trainset, valset, testset = split_examples(all_examples, seed)

    print(f"Train size: {len(trainset)}")
    print(f"Val size:   {len(valset)}")
    print(f"Test size:  {len(testset)}")

    baseline_program = build_program()

    evaluate_test = dspy.Evaluate(
        devset=testset,
        metric=metric,
        num_threads=1,
        display_progress=True,
        display_table=False,
    )

    baseline_result = evaluate_test(baseline_program)
    print(f"\nBASELINE TEST SCORE (seed {seed}): {baseline_result.score:.3f}")

    optimizer = dspy.GEPA(
        metric=metric,
        auto="light",
        reflection_lm=reflection_lm,
        track_stats=True,
        reflection_minibatch_size=2,
    )

    # Important: compile from a fresh baseline program
    train_program = build_program()

    optimized_program = optimizer.compile(
        train_program,
        trainset=trainset,
        valset=valset,
    )

    optimized_result = evaluate_test(optimized_program)
    print(f"OPTIMIZED TEST SCORE (seed {seed}): {optimized_result.score:.3f}")

    baseline_by_label = score_by_label(baseline_program, testset)
    optimized_by_label = score_by_label(optimized_program, testset)

    per_example = []
    print("\nTEST SET COMPARISON")
    print("-" * 80)

    for ex in testset:
        base_pred = baseline_program(
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

        record = {
            "id": ex.id,
            "gold_label": ex.gold_support_label,
            "baseline_score": base_score,
            "optimized_score": opt_score,
        }
        per_example.append(record)

        print(
            f"ID: {ex.id} | "
            f"label={ex.gold_support_label} | "
            f"baseline={base_score:.2f} | "
            f"optimized={opt_score:.2f}"
        )

    optimized_program.save(f"optimized_full_gepa_seed_{seed}.json")

    return {
        "seed": seed,
        "baseline_test_score": baseline_result.score,
        "optimized_test_score": optimized_result.score,
        "delta": optimized_result.score - baseline_result.score,
        "baseline_by_label": baseline_by_label,
        "optimized_by_label": optimized_by_label,
        "per_example": per_example,
    }


# ----------------------------
# 6) Main
# ----------------------------
def main():
    all_examples = load_examples("humor_examples.json")
    print(f"Loaded {len(all_examples)} examples.")

    assert len(all_examples) >= TRAIN_SIZE + VAL_SIZE + TEST_SIZE, "Not enough examples."

    all_results = []

    for seed in SEEDS:
        result = run_one_seed(seed, all_examples)
        all_results.append(result)

    baseline_scores = [r["baseline_test_score"] for r in all_results]
    optimized_scores = [r["optimized_test_score"] for r in all_results]
    deltas = [r["delta"] for r in all_results]

    print("\n" + "=" * 80)
    print("SUMMARY ACROSS SEEDS")
    print("=" * 80)
    print(f"Seeds: {SEEDS}")
    print(f"Mean baseline test score:  {statistics.mean(baseline_scores):.3f}")
    print(f"Mean optimized test score: {statistics.mean(optimized_scores):.3f}")
    print(f"Mean delta:                {statistics.mean(deltas):.3f}")

    if len(SEEDS) > 1:
        print(f"Std baseline test score:   {statistics.pstdev(baseline_scores):.3f}")
        print(f"Std optimized test score:  {statistics.pstdev(optimized_scores):.3f}")
        print(f"Std delta:                 {statistics.pstdev(deltas):.3f}")

    improve = sum(1 for d in deltas if d > 0)
    same = sum(1 for d in deltas if d == 0)
    worse = sum(1 for d in deltas if d < 0)

    print(f"Improved seeds: {improve}")
    print(f"Unchanged seeds: {same}")
    print(f"Worse seeds: {worse}")

    with open("gepa_multi_seed_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved detailed results to gepa_multi_seed_results.json")


if __name__ == "__main__":
    main()