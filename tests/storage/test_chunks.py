# tests/storage/test_chunks.py
from storage.chunks import distance_to_similarity


def test_distance_to_similarity_identical_vectors():
    assert distance_to_similarity(0.0) == 1.0


def test_distance_to_similarity_orthogonal_vectors():
    assert distance_to_similarity(1.0) == 0.0


def test_distance_to_similarity_opposite_vectors():
    assert distance_to_similarity(2.0) == -1.0
