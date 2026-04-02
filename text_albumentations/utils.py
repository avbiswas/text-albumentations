from pydantic import BaseModel, Field

class AlpacaDataset(BaseModel):
    instruction: str
    input: str
    output: str


def count_words(value) -> int:
    if isinstance(value, str):
        return len(value.split())
    if isinstance(value, list):
        return sum(count_words(item) for item in value)
    if isinstance(value, tuple):
        return sum(count_words(item) for item in value)
    if isinstance(value, dict):
        return sum(count_words(item) for item in value.values())
    return 0


def estimate_max_length_from_words(
    value,
    multiplier: float,
    *,
    chars_per_word: int = 8,
    minimum: int = 0,
) -> int:
    word_count = count_words(value)
    estimated_length = int(word_count * multiplier * chars_per_word)
    return max(minimum, estimated_length)

def save_dataset(dataset: list[AlpacaDataset],
                 filename):
    if not filename.endswith(".jsonl"):
        filename = filename + ".jsonl"
    with open(filename, "a") as f:
        for d in dataset:
            f.write(d.model_dump_json()+"\n")
    
    print(f"Saved {len(dataset)} rows in {filename}")

if __name__ == "__main__":
    save_dataset(
        [
    AlpacaDataset(instruction="1", input="2", output="3"),
    AlpacaDataset(instruction="3", input="2", output="3"),
        ],
        "test.jsonl"
    )
