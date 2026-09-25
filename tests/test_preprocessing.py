from src.email_spam.preprocessing import normalize_email, normalize_label, parse_eml, parse_uploaded_file


def test_email_normalization_preserves_email_signals():
    email = normalize_email("Urgent!!!", "Visit https://example.com and email user@example.com")
    assert "URLTOKEN" in email.body
    assert "EMAILTOKEN" in email.body
    assert email.subject == "Urgent!!!"


def test_email_normalization_removes_invisible_formatting_controls():
    email = normalize_email("Meeting\u200b confirmed", "Please review\u2060 the attached agenda.")
    assert email.subject == "Meeting confirmed"
    assert email.body == "Please review the attached agenda."


def test_labels_are_case_insensitive():
    assert normalize_label("SPAM") == 1
    assert normalize_label("Ham") == 0


def test_eml_parser_ignores_attachment_content():
    content = b"Subject: Hello\nContent-Type: text/plain\n\nSafe body"
    assert parse_eml(content).subject == "Hello"
    assert "Safe body" in parse_eml(content).body


def test_invalid_upload_is_rejected():
    try:
        parse_uploaded_file("payload.html", b"<script>alert(1)</script>")
    except ValueError as error:
        assert "supported" in str(error)
    else:
        raise AssertionError("Unsupported file should be rejected")
