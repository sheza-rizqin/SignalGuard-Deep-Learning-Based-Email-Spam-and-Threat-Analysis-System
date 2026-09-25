from src.email_spam.model import build_model


def test_embedding_masks_padding_tokens():
    model = build_model(vocabulary_size=100, sequence_length=12)
    assert model.layers[0].mask_zero is True
