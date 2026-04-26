import ast
import json
from pathlib import Path

TED_PATH = "../data-generation/files/ted_reasoning_train.json"
EXCLUDE_PATH = "../humor_data_pilot.json"   # optional exclusion set
OUT_PATH = "ted_converted_all.json"


def extract_video_clip_dict(human_value: str):
    marker = "Given video clip:"
    idx = human_value.find(marker)
    if idx == -1:
        return None
    clip_str = human_value[idx + len(marker):].strip()
    try:
        return ast.literal_eval(clip_str)
    except Exception:
        return None


def convert_one(raw_item):
    conversations = raw_item.get("conversations", [])
    human_turn = None
    for turn in conversations:
        if turn.get("from") == "human":
            human_turn = turn.get("value", "")
            break
    if not human_turn:
        return None

    clip = extract_video_clip_dict(human_turn)
    if clip is None:
        return None

    ordered_keys = sorted(clip.keys(), key=lambda x: int(x))
    utterances = []
    laugh_index = None

    for i, k in enumerate(ordered_keys):
        turn = clip[k]
        utt = str(turn.get("Utterance", "")).strip()
        speaker = str(turn.get("Speaker", "")).strip()
        utterances.append({"speaker": speaker, "utterance": utt})
        if laugh_index is None and "(audience laughs)" in utt:
            laugh_index = i

    if laugh_index is None:
        return None

    x_turn = utterances[laugh_index]
    c_turns = utterances[:laugh_index]

    return {
        "id": str(raw_item.get("id")),
        "source": "TED",
        "C_text": [f"{t['speaker']}: {t['utterance']}" for t in c_turns],
        "X_text": x_turn["utterance"],
        "X_speaker": x_turn["speaker"],
        "Y": 1
    }


def main():
    ted_raw = json.loads(Path(TED_PATH).read_text())
    exclude_raw = json.loads(Path(EXCLUDE_PATH).read_text())

    exclude_ids = {str(row["id"]) for row in exclude_raw}

    converted = []
    seen = set()

    for row in ted_raw:
        ex = convert_one(row)
        if ex is None:
            continue
        if ex["id"] in exclude_ids:
            continue
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        converted.append(ex)

    Path(OUT_PATH).write_text(json.dumps(converted, indent=2))
    print(f"Saved {len(converted)} TED examples to {OUT_PATH}")


if __name__ == "__main__":
    main()