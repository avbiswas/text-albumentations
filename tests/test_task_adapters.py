"""Offline tests for task schemas, adapters, and programmatic generators."""

from text_albumentations.tasks.backtranslation import (
    BacktranslatedInstruction,
    BacktranslationAdapter,
)
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.claim_verification import (
    ClaimVerificationAdapter,
    claim_verification_augmentation,
    ClaimVerificationItem,
    ClaimVerifications,
    JSON_INSTRUCTION as CLAIM_JSON_INSTRUCTION,
)
from text_albumentations.tasks.classification import ClassificationAdapter, PassageLabels
from text_albumentations.tasks.cloze import cloze_augmentation
from text_albumentations.tasks.counterfactual import Counterfactual, CounterfactualAdapter
from text_albumentations.tasks.definition_extraction import (
    DefinitionExtractionAdapter,
    definition_extraction_augmentation,
    DefinitionItem,
    Definitions,
    QUOTE_INSTRUCTION,
)
from text_albumentations.tasks.distractor_qa import (
    DistractorQaAdapter,
    distractor_qa_augmentation,
    DISTRACTOR_INSTRUCTION,
    KeywordReplacement,
    MultipleChoiceQuestion,
    MultipleChoiceQuestions,
    VALIDITY_INSTRUCTION,
)
from text_albumentations.tasks.entity_extraction import (
    Entity,
    EntityExtractionAdapter,
    entity_extraction_augmentation,
    EntityList,
    EXTRACT_INSTRUCTION as ENTITY_EXTRACT_INSTRUCTION,
)
from text_albumentations.tasks.error_correction import (
    ErrorCorrection,
    ErrorCorrectionAdapter,
    error_correction_augmentation,
)
from text_albumentations.tasks.evidence_selection import (
    EvidenceSelectionAdapter,
    evidence_selection_augmentation,
    EvidenceSelectionItem,
    EvidenceSelections,
    QUOTE_SUPPORT_INSTRUCTION,
)
from text_albumentations.tasks.extractive_qa import (
    ExtractiveQa,
    ExtractiveQaAdapter,
    ExtractiveQaItem,
    quote_in_passage,
)
from text_albumentations.tasks.method_steps import (
    MethodStep,
    MethodSteps,
    MethodStepsAdapter,
    method_steps_augmentation,
    MISSING_STEP_INSTRUCTION,
)
from text_albumentations.tasks.qa_pairs import (
    JsonQaAdapter,
    MarkdownQaAdapter,
    QA,
    QAList,
    qa_pair_augmentation,
)
from text_albumentations.tasks.query_generation import (
    query_generation_augmentation,
    QueryGenerationAdapter,
    SearchQueries,
)
from text_albumentations.tasks.section_heading import (
    SectionHeading,
    SectionHeadingAdapter,
    section_heading_augmentation,
)
from text_albumentations.tasks.style_transfer import StyleTransferAugmentation
from text_albumentations.tasks.structured_records import (
    JSON_INSTRUCTION as RECORDS_JSON_INSTRUCTION,
    StructuredRecord,
    StructuredRecords,
    StructuredRecordsAdapter,
    structured_records_augmentation,
)
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

    def test_new_task_instruction_templates_default_first(self):
        for augmentation in (
            claim_verification_augmentation,
            definition_extraction_augmentation,
            distractor_qa_augmentation,
            entity_extraction_augmentation,
            error_correction_augmentation,
            evidence_selection_augmentation,
            method_steps_augmentation,
            query_generation_augmentation,
            section_heading_augmentation,
            structured_records_augmentation,
        ):
            assert augmentation.instruction_templates
            for default, variants in augmentation.instruction_templates.items():
                assert variants[0] == default

    def test_sensitive_new_task_templates_preserve_semantics(self):
        checks = [
            (claim_verification_augmentation, CLAIM_JSON_INSTRUCTION, ("json",)),
            (definition_extraction_augmentation, QUOTE_INSTRUCTION, ("quote", "evidence")),
            (
                distractor_qa_augmentation,
                DISTRACTOR_INSTRUCTION,
                ("incorrect", "wrong", "distractor"),
            ),
            (distractor_qa_augmentation, VALIDITY_INSTRUCTION, ("invalid",)),
            (entity_extraction_augmentation, ENTITY_EXTRACT_INSTRUCTION, ("json",)),
            (evidence_selection_augmentation, QUOTE_SUPPORT_INSTRUCTION, ("yes", "no")),
            (method_steps_augmentation, MISSING_STEP_INSTRUCTION, ("missing", "omitted")),
            (structured_records_augmentation, RECORDS_JSON_INSTRUCTION, ("json",)),
        ]
        for augmentation, instruction, required_words in checks:
            variants = augmentation.instruction_templates[instruction]
            assert all(
                any(required_word in variant.lower() for required_word in required_words)
                for variant in variants
            )


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


class TestNewTasks:
    def test_evidence_selection_keeps_verified_candidate(self, passage):
        output = EvidenceSelections(
            items=[
                EvidenceSelectionItem(
                    claim="The Transformer does not use recurrence.",
                    candidate_quotes=[
                        "dispensing with recurrence and convolutions entirely",
                        "The best performing models also connect the encoder and decoder",
                    ],
                    supporting_quote="dispensing with recurrence and convolutions entirely",
                    rationale="The quote says the model dispenses with recurrence.",
                ),
                EvidenceSelectionItem(
                    claim="The model was trained on C4.",
                    candidate_quotes=["trained on C4", "trained on Wikipedia"],
                    supporting_quote="trained on C4",
                    rationale="This quote is not in the passage.",
                ),
            ]
        )
        rows = EvidenceSelectionAdapter().convert(passage, output)
        assert len(rows) == 4
        assert rows[0].output == "dispensing with recurrence and convolutions entirely"
        assert rows[-2].output == "yes"
        assert rows[-1].output == "no"
        assert all("Write the claim supported" not in row.instruction for row in rows)

    def test_claim_verification_emits_label_and_evidence(self, passage):
        output = ClaimVerifications(
            items=[
                ClaimVerificationItem(
                    claim="The model uses attention.",
                    label="supported",
                    evidence="The passage says it is based on attention mechanisms.",
                )
            ]
        )
        rows = ClaimVerificationAdapter().convert(passage, output)
        assert len(rows) == 4
        assert rows[0].output.startswith("Label: supported")
        assert rows[1].output == "supported"
        assert '"label": "supported"' in rows[3].output

    def test_entity_extraction_emits_collection_and_type_rows(self, passage):
        output = EntityList(
            entities=[
                Entity(
                    name="Transformer",
                    type="method",
                    context="A network architecture.",
                )
            ]
        )
        rows = EntityExtractionAdapter().convert(passage, output)
        assert len(rows) == 2
        assert '"name": "Transformer"' in rows[0].output
        assert rows[1].output == "method"

    def test_definition_extraction_requires_verified_quote(self, passage):
        output = Definitions(
            definitions=[
                DefinitionItem(
                    term="Transformer",
                    definition="A network architecture based on attention.",
                    supporting_quote="the Transformer, based solely on attention mechanisms",
                ),
                DefinitionItem(
                    term="C4",
                    definition="A dataset.",
                    supporting_quote="trained on the C4 dataset",
                ),
            ]
        )
        rows = DefinitionExtractionAdapter().convert(passage, output)
        assert len(rows) == 2
        assert rows[0].output == "A network architecture based on attention."

    def test_method_steps_formats_numbered_steps(self, passage):
        output = MethodSteps(
            process_name="Sequence modeling",
            steps=[
                MethodStep(step="Encode the input sequence."),
                MethodStep(step="Decode the output sequence."),
            ]
        )
        rows = MethodStepsAdapter().convert(passage, output)
        assert len(rows) == 5
        assert rows[0].output == "1. Encode the input sequence.\n2. Decode the output sequence."
        assert rows[1].output == "Sequence modeling"
        assert rows[-1].output == "Decode the output sequence."
        assert all("How many ordered steps" not in row.instruction for row in rows)

    def test_structured_records_emits_json_and_text(self, passage):
        output = StructuredRecords(
            records=[
                StructuredRecord(
                    subject="Transformer",
                    attribute="basis",
                    value="attention mechanisms",
                )
            ]
        )
        rows = StructuredRecordsAdapter().convert(passage, output)
        assert len(rows) == 5
        assert '"attribute": "basis"' in rows[0].output
        assert "Transformer - basis: attention mechanisms" in rows[1].output
        assert rows[2].output == "attention mechanisms"
        assert rows[3].output == "basis"
        assert rows[4].output == "Transformer"
        assert all("Turn this structured record" not in row.instruction for row in rows)

    def test_section_heading_emits_heading_and_rationale(self, passage):
        rows = SectionHeadingAdapter().convert(
            passage,
            SectionHeading(
                heading="Attention-Based Sequence Models",
                rationale="The passage describes a model based on attention.",
            ),
        )
        assert [row.output for row in rows] == [
            "Attention-Based Sequence Models",
            "The passage describes a model based on attention.",
        ]

    def test_query_generation_emits_query_and_reverse_rows(self, passage):
        rows = QueryGenerationAdapter().convert(
            passage,
            SearchQueries(queries=["transformer attention sequence transduction"]),
        )
        assert len(rows) == 2
        assert rows[0].output == "transformer attention sequence transduction"
        assert rows[1].output == passage

    def test_distractor_qa_drops_duplicate_choices(self, passage):
        output = MultipleChoiceQuestions(
            questions=[
                MultipleChoiceQuestion(
                    question="What mechanism does the Transformer use?",
                    correct_answer="attention",
                    distractors=["recurrence", "convolutions"],
                    explanation="The passage says it is based solely on attention.",
                    keyword_replacements=[
                        KeywordReplacement(
                            keyword="attention",
                            replacement="recurrence",
                        )
                    ],
                ),
                MultipleChoiceQuestion(
                    question="Duplicate?",
                    correct_answer="attention",
                    distractors=["attention", "convolutions"],
                    explanation="Duplicate choices are malformed.",
                    keyword_replacements=[
                        KeywordReplacement(
                            keyword="attention",
                            replacement="recurrence",
                        )
                    ],
                ),
            ]
        )
        rows = DistractorQaAdapter().convert(passage, output)
        assert len(rows) == 7
        assert "Answer: attention" in rows[0].output
        assert rows[1].output == "attention"
        assert rows[-3].output == "invalid"
        assert rows[-2].output == passage
        assert rows[-1].output == "attention -> recurrence"

    def test_error_correction_outputs_original_passage(self, passage):
        rows = ErrorCorrectionAdapter().convert(
            passage,
            ErrorCorrection(
                corrupted_passage="The Transformer relies on recurrence.",
                correction_notes="Replace recurrence with attention mechanisms.",
            ),
        )
        assert rows[0].output == passage
        assert rows[1].output == "Replace recurrence with attention mechanisms."


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
