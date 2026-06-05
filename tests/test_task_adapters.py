"""Offline tests for task schemas, adapters, and programmatic generators."""

from text_albumentations.tasks.backtranslation import (
    BacktranslatedInstruction,
    BacktranslationAdapter,
)
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.classification import ClassificationAdapter, PassageLabels
from text_albumentations.tasks.cloze import cloze_augmentation
from text_albumentations.tasks.counterfactual import Counterfactual, CounterfactualAdapter
from text_albumentations.tasks.extractive_qa import (
    ExtractiveQa,
    ExtractiveQaAdapter,
    ExtractiveQaItem,
    quote_in_passage,
)
from text_albumentations.tasks.qa_pairs import (
    JsonQaAdapter,
    MarkdownQaAdapter,
    QA,
    QAList,
    qa_pair_augmentation,
)
from text_albumentations.tasks.style_transfer import StyleTransferAugmentation
from text_albumentations.tasks.summarize import Summary, SummarizeAdapter, summarize_augmentation
from text_albumentations.tasks.title import TitleAdapter, TitleHeadline
from text_albumentations.tasks.triplets import triplet_augmentation


class TestCloze:
    def test_word_blanks_match_answers(self, passage):
        slices = cloze_augmentation.generate_one(passage, runtime=None)

        assert "[BLANK 1]" in slices.word_masked_passage
        for line in slices.word_answers.splitlines():
            blank, word = line.split(": ")
            assert blank.strip("[]").startswith("BLANK")
            assert word not in ("", None)
            # restoring every answer reproduces words absent from the masked text
            assert word in passage

    def test_sentence_mask_round_trips(self, passage):
        slices = cloze_augmentation.generate_one(passage, runtime=None)

        assert "[MISSING SENTENCE]" in slices.sentence_masked_passage
        assert slices.sentence_answer in passage
        assert slices.sentence_answer not in slices.sentence_masked_passage

    def test_short_passage_yields_no_rows(self):
        slices = cloze_augmentation.generate_one("Too short to mask.", runtime=None)
        rows = cloze_augmentation.adapters[0].convert("Too short to mask.", slices)
        assert rows == []


class TestExtractiveQa:
    def test_verified_quote_kept_fabricated_dropped(self, passage):
        output = ExtractiveQa(
            items=[
                ExtractiveQaItem(
                    question="What does the Transformer dispense with?",
                    supporting_quote="dispensing with recurrence and convolutions entirely",
                ),
                ExtractiveQaItem(
                    question="What dataset was used?",
                    supporting_quote="trained on the C4 dataset",  # not in passage
                ),
            ]
        )
        rows = ExtractiveQaAdapter().convert(passage, output)

        assert len(rows) == 1
        assert "dispensing with recurrence" in rows[0].output

    def test_quote_verification_normalizes_whitespace_and_case(self):
        assert quote_in_passage("Hello   WORLD", "hello world, again")
        assert not quote_in_passage("goodbye world", "hello world")


class TestQaPairs:
    def test_markdown_adapter_uses_first_pair_for_single_question(self, passage):
        output = QAList(
            qa_pairs=[
                QA(question="First question?", answer="First answer."),
                QA(question="Second question?", answer="Second answer."),
                QA(question="Third question?", answer="Third answer."),
            ]
        )

        rows = MarkdownQaAdapter().convert(passage, output)

        assert len(rows) == 12
        single_question_rows = [
            row for row in rows if row.instruction == "Generate a question from this passage"
        ]
        fact_rows = [
            row
            for row in rows
            if row.instruction
            == "Generate an important fact or piece of information from this passage"
        ]
        list_question_rows = [
            row
            for row in rows
            if row.instruction
            == "List the important questions answered by this passage using markdown."
        ]
        assert [row.output for row in single_question_rows] == ["First question?"]
        assert fact_rows == []
        assert list_question_rows == []

    def test_json_adapter_uses_first_pair_for_single_question(self, passage):
        output = QAList(
            qa_pairs=[
                QA(question="First question?", answer="First answer."),
                QA(question="Second question?", answer="Second answer."),
                QA(question="Third question?", answer="Third answer."),
            ]
        )

        rows = JsonQaAdapter().convert(passage, output)

        assert len(rows) == 12
        single_question_rows = [
            row for row in rows if row.instruction == "Generate a question from this passage"
        ]
        fact_rows = [
            row
            for row in rows
            if row.instruction
            == "Generate an important fact or piece of information from this passage"
        ]
        list_question_rows = [
            row
            for row in rows
            if row.instruction
            == "List the important questions answered by this passage. Return a JSON array of strings."
        ]
        assert [row.output for row in single_question_rows] == ["First question?"]
        assert fact_rows == []
        assert list_question_rows == []


class TestInstructionTemplates:
    def test_format_specific_instruction_templates_keep_format_words(self):
        for templates in (
            bullet_augmentation.instruction_templates,
            qa_pair_augmentation.instruction_templates,
            triplet_augmentation.instruction_templates,
        ):
            for default, variants in templates.items():
                normalized_default = default.lower()
                if "markdown" in normalized_default:
                    assert all("markdown" in variant.lower() for variant in variants)
                if "json" in normalized_default:
                    assert all("json" in variant.lower() for variant in variants)
                if "python list" in normalized_default:
                    assert all("python list" in variant.lower() for variant in variants)

    def test_default_instruction_is_first_variant(self):
        for templates in (
            bullet_augmentation.instruction_templates,
            qa_pair_augmentation.instruction_templates,
            triplet_augmentation.instruction_templates,
        ):
            for default, variants in templates.items():
                assert variants[0] == default


class TestCounterfactual:
    def test_instruction_is_self_contained(self, passage):
        output = Counterfactual(
            original_claim="The Transformer is more parallelizable.",
            altered_premise="Suppose the Transformer required recurrence.",
            question="What would follow for training time?",
            consequence="Training would no longer parallelize.",
        )
        rows = CounterfactualAdapter().convert(passage, output)

        assert len(rows) == 1
        assert "Suppose the Transformer required recurrence." in rows[0].instruction
        assert "What would follow for training time?" in rows[0].instruction


class TestSimpleAdapters:
    def test_summarize_emits_tldr_and_summary_rows(self, passage):
        rows = SummarizeAdapter().convert(passage, Summary(tldr="t", summary="s"))
        assert [row.output for row in rows] == ["t", "s"]
        assert all(row.input == passage for row in rows)

    def test_title_emits_title_and_headline_rows(self, passage):
        rows = TitleAdapter().convert(passage, TitleHeadline(title="T", headline="H"))
        assert [row.output for row in rows] == ["T", "H"]

    def test_classification_emits_three_label_rows(self, passage):
        labels = PassageLabels(topic="ML", tone="technical", audience="domain experts")
        rows = ClassificationAdapter().convert(passage, labels)
        assert [row.output for row in rows] == ["ML", "technical", "domain experts"]

    def test_backtranslation_passage_becomes_output(self, passage):
        rows = BacktranslationAdapter().convert(
            passage, BacktranslatedInstruction(instruction="Explain the Transformer.")
        )
        assert len(rows) == 1
        assert rows[0].instruction == "Explain the Transformer."
        assert rows[0].input == ""
        assert rows[0].output == passage


class TestStyleTransfer:
    def test_known_styles_build_prompts(self):
        aug = StyleTransferAugmentation(style="formal")
        assert "formal" in aug.system_prompt
        assert "formal" in aug.adapters[0].style_description

    def test_unknown_style_raises(self):
        try:
            StyleTransferAugmentation(style="piratese")
        except ValueError as error:
            assert "piratese" in str(error)
        else:
            raise AssertionError("expected ValueError")

    def test_custom_style_description(self):
        aug = StyleTransferAugmentation(style="pirate", style_description="pirate speak")
        assert "pirate speak" in aug.system_prompt


class TestDynamicSchemas:
    def test_summarize_schema_scales_with_passage(self, passage):
        schema = summarize_augmentation.get_schema(passage * 20)
        max_length = schema.model_fields["summary"].metadata[0].max_length
        assert max_length > 1000

    def test_summarize_schema_minimum_for_short_passages(self):
        schema = summarize_augmentation.get_schema("short passage")
        assert schema is summarize_augmentation.schema
