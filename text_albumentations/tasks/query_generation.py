from pydantic import BaseModel, Field

from text_albumentations.base import BaseSingleChunkAugmentation
from text_albumentations.output_format_adapters import BaseAlpacaAdapter
from text_albumentations.runner import run_augmentation
from text_albumentations.runtime import get_default_outlines_runtime
from text_albumentations.utils import AlpacaDataset


class SearchQueries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6)


SYSTEM_PROMPT = """
Given this passage, generate realistic search queries a user might type to retrieve this passage or a document section like it. Queries should be short, specific, and grounded in the passage. Only generate answers.
    """

LIST_INSTRUCTION = "Generate search queries that could retrieve this passage."
PASSAGE_INSTRUCTION = "Given this search query, provide the passage it is meant to retrieve."


class QueryGenerationAdapter(BaseAlpacaAdapter[str, SearchQueries]):
    def convert(self, passages: str, output: SearchQueries) -> list[AlpacaDataset]:
        rows = [
            AlpacaDataset(
                instruction=LIST_INSTRUCTION,
                input=passages,
                output="\n".join(output.queries),
            )
        ]
        rows.extend(
            AlpacaDataset(
                instruction=PASSAGE_INSTRUCTION,
                input=query,
                output=passages,
            )
            for query in output.queries
        )
        return rows


class QueryGenerationAugmentation(BaseSingleChunkAugmentation[SearchQueries]):
    schema = SearchQueries
    system_prompt = SYSTEM_PROMPT
    selection_hint = "the passage contains specific information someone might search for."
    adapters = (QueryGenerationAdapter(),)
    temperature = 0.5
    instruction_templates = {
        LIST_INSTRUCTION: (
            LIST_INSTRUCTION,
            "Write search queries that could retrieve this passage.",
            "Generate passage-grounded search queries for this text.",
            "List realistic search queries someone might use to find this passage.",
            "Create specific search queries that match the passage content.",
        ),
        PASSAGE_INSTRUCTION: (
            PASSAGE_INSTRUCTION,
            "Return the passage that this search query is intended to retrieve.",
            "Provide the relevant passage for this search query.",
            "Given the query, output the passage it should retrieve.",
            "Use this search query to return the matching passage.",
        ),
    }


query_generation_augmentation = QueryGenerationAugmentation()


def main(passage: str) -> list[AlpacaDataset]:
    return run_augmentation(
        passage,
        query_generation_augmentation,
        get_default_outlines_runtime(),
    )


if __name__ == "__main__":
    dataset = main(
        """
The Transformer is based solely on attention mechanisms and is more parallelizable than recurrent models.
        """
    )
    print(len(dataset))
