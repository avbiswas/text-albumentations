from text_albumentations.tasks.backtranslation import backtranslation_augmentation
from text_albumentations.tasks.backtranslation import main as generate_backtranslation
from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.bullets import main as generate_bullets
from text_albumentations.tasks.classification import classification_augmentation
from text_albumentations.tasks.classification import main as generate_classification
from text_albumentations.tasks.cloze import cloze_augmentation
from text_albumentations.tasks.cloze import main as generate_cloze
from text_albumentations.tasks.comparison import comparison_augmentation
from text_albumentations.tasks.comparison import main as generate_comparisons
from text_albumentations.tasks.continuation import continuation_augmentation
from text_albumentations.tasks.continuation import main as generate_continuation
from text_albumentations.tasks.counterfactual import counterfactual_augmentation
from text_albumentations.tasks.counterfactual import main as generate_counterfactual
from text_albumentations.tasks.extractive_qa import extractive_qa_augmentation
from text_albumentations.tasks.extractive_qa import main as generate_extractive_qa
from text_albumentations.tasks.qa_pairs import main as generate_qa_pairs
from text_albumentations.tasks.qa_pairs import qa_pair_augmentation
from text_albumentations.tasks.rephrase import main as generate_rephrase
from text_albumentations.tasks.rephrase import rephrase_augmentation
from text_albumentations.tasks.retrieval import main as generate_retrieval
from text_albumentations.tasks.retrieval import retrieval_augmentation
from text_albumentations.tasks.style_transfer import (
    casual_style_augmentation,
    eli5_style_augmentation,
    formal_style_augmentation,
    style_transfer_augmentation,
)
from text_albumentations.tasks.style_transfer import main as generate_style_transfer
from text_albumentations.tasks.summarize import main as generate_summaries
from text_albumentations.tasks.summarize import summarize_augmentation
from text_albumentations.tasks.title import main as generate_titles
from text_albumentations.tasks.title import title_augmentation
from text_albumentations.tasks.triplets import main as generate_triplets
from text_albumentations.tasks.triplets import triplet_augmentation

__all__ = [
    "backtranslation_augmentation",
    "bullet_augmentation",
    "casual_style_augmentation",
    "classification_augmentation",
    "cloze_augmentation",
    "comparison_augmentation",
    "continuation_augmentation",
    "counterfactual_augmentation",
    "eli5_style_augmentation",
    "extractive_qa_augmentation",
    "formal_style_augmentation",
    "generate_backtranslation",
    "generate_bullets",
    "generate_classification",
    "generate_cloze",
    "generate_comparisons",
    "generate_continuation",
    "generate_counterfactual",
    "generate_extractive_qa",
    "generate_qa_pairs",
    "generate_rephrase",
    "generate_retrieval",
    "generate_style_transfer",
    "generate_summaries",
    "generate_titles",
    "generate_triplets",
    "qa_pair_augmentation",
    "rephrase_augmentation",
    "retrieval_augmentation",
    "style_transfer_augmentation",
    "summarize_augmentation",
    "title_augmentation",
    "triplet_augmentation",
]
