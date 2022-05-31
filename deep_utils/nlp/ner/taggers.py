from typing import List


def bio_tagger(tokens: List[str], label, others_label="o"):
    if label == others_label:
        return ["o"] * len(tokens)
    return [f"B-{label}"] + [f"I-{label}"] * (len(tokens) - 1)
