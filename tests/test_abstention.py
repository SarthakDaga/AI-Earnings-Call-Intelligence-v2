from src.qa.rag import is_abstention


def test_misspelled_abstention_is_detected():
    assert is_abstention("INSUFFICENT_EVIDENCE (no information in the passages)")
    assert is_abstention("insufficient evidence")
    assert is_abstention("INSUFFICIENT_EVIDENCE")
    assert is_abstention("")


def test_normal_answers_are_not_abstentions():
    assert not is_abstention("Revenue was $5 billion [c1].")
    assert not is_abstention("The evidence shows growth of 9x [c2].")