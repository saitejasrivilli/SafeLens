from safelens.data.multimodal.validation.text import check_text


def test_arabic_text_preserved_and_detected():
    result = check_text("لن تغرق سفينة انت قائدها")
    assert result.exists
    assert result.non_empty
    assert result.contains_arabic
    assert not result.has_replacement_char


def test_empty_text_flagged():
    result = check_text("   ")
    assert result.exists
    assert not result.non_empty


def test_missing_text_flagged():
    result = check_text(None)
    assert not result.exists
    assert result.error == "missing"


def test_non_string_text_flagged():
    result = check_text(12345)
    assert result.exists
    assert not result.non_empty
    assert "not a string" in (result.error or "")


def test_replacement_char_detected():
    result = check_text("some text with corruption � here")
    assert result.has_replacement_char


def test_non_arabic_text_not_flagged_as_arabic():
    result = check_text("just plain english text")
    assert not result.contains_arabic
    assert result.non_empty
