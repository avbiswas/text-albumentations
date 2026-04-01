from text_albumentations.tasks.bullets import bullet_augmentation
from text_albumentations.tasks.bullets import main as generate_bullets
from text_albumentations.tasks.comparison import comparison_augmentation
from text_albumentations.tasks.comparison import main as generate_comparisons
from text_albumentations.tasks.continuation import continuation_augmentation
from text_albumentations.tasks.continuation import main as generate_continuation
from text_albumentations.tasks.qa_pairs import qa_pair_augmentation
from text_albumentations.tasks.qa_pairs import main as generate_qa_pairs
from text_albumentations.tasks.rephrase import rephrase_augmentation
from text_albumentations.tasks.rephrase import main as generate_rephrase
from text_albumentations.tasks.retrieval import retrieval_augmentation
from text_albumentations.tasks.retrieval import main as generate_retrieval
from text_albumentations.tasks.triplets import triplet_augmentation
from text_albumentations.tasks.triplets import main as generate_triplets

__all__ = [
    "bullet_augmentation",
    "comparison_augmentation",
    "continuation_augmentation",
    "generate_bullets",
    "generate_comparisons",
    "generate_continuation",
    "generate_qa_pairs",
    "generate_rephrase",
    "generate_retrieval",
    "generate_triplets",
    "qa_pair_augmentation",
    "rephrase_augmentation",
    "retrieval_augmentation",
    "triplet_augmentation",
]
