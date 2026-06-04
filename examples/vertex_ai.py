import asyncio
import text_albumentations as ta

MODEL_NAME = "gemini-2.5-flash"
PROJECT = "your-gcp-project"
LOCATION = "us-central1"
PASSAGE = (
    "Machine learning models can exhibit unexpected behavior when deployed "
    "in production environments due to data drift, where the statistical "
    "properties of real-world inputs diverge from training data distributions."
)

def main():
    model = ta.VertexAIModel(MODEL_NAME, project=PROJECT, location=LOCATION)

    print("model_type=VertexAIModel")
    print("mode=sync")

    rows = ta.augment(PASSAGE, tasks=["bullets", "qa_pairs"], model=model)
    for row in rows:
        print(row.model_dump_json())


async def amain():
    model = ta.VertexAIModel(
        MODEL_NAME,
        project=PROJECT,
        location=LOCATION,
        async_mode=True,
        total_concurrent_calls=10,
    )

    print("model_type=VertexAIModel")
    print("mode=async")

    rows = await ta.aaugment(PASSAGE, tasks=["bullets", "qa_pairs"], model=model)
    for row in rows:
        print(row.model_dump_json())


if __name__ == "__main__":
    main()
    asyncio.run(amain())
