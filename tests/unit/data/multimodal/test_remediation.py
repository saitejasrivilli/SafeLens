from safelens.data.multimodal.preprocessing.remediation import remediate_train_leakage


def test_removes_train_example_matching_dev_caption(make_example_fn):
    train = [
        make_example_fn("t1", "leaked caption", "a.jpg", split="train"),
        make_example_fn("t2", "clean caption", "b.jpg", split="train"),
    ]
    dev = [make_example_fn("d1", "leaked caption", "c.jpg", split="dev")]
    test: list = []

    result = remediate_train_leakage(train, dev, test)

    assert result.original_count == 2
    assert result.removed_count == 1
    assert result.final_count == 1
    assert [ex.example_id for ex in result.clean_train] == ["t2"]
    assert [ex.example_id for ex in result.removed_examples] == ["t1"]


def test_removes_train_example_matching_test_caption(make_example_fn):
    train = [make_example_fn("t1", "leaked caption", "a.jpg", split="train")]
    dev: list = []
    test = [make_example_fn("s1", "leaked caption", "c.jpg", split="test")]

    result = remediate_train_leakage(train, dev, test)

    assert result.removed_count == 1
    assert result.clean_train == []


def test_dev_and_test_are_never_modified(make_example_fn):
    train = [make_example_fn("t1", "leaked caption", "a.jpg", split="train")]
    dev = [make_example_fn("d1", "leaked caption", "c.jpg", split="dev")]
    test = [make_example_fn("s1", "another caption", "d.jpg", split="test")]

    remediate_train_leakage(train, dev, test)

    # dev/test lists passed in are untouched (function does not mutate inputs)
    assert dev == [make_example_fn("d1", "leaked caption", "c.jpg", split="dev")]
    assert test == [make_example_fn("s1", "another caption", "d.jpg", split="test")]


def test_no_leakage_removes_nothing(make_example_fn):
    train = [make_example_fn("t1", "unique train caption", "a.jpg", split="train")]
    dev = [make_example_fn("d1", "unique dev caption", "b.jpg", split="dev")]
    test = [make_example_fn("s1", "unique test caption", "c.jpg", split="test")]

    result = remediate_train_leakage(train, dev, test)

    assert result.removed_count == 0
    assert result.final_count == 1


def test_remediation_is_deterministic_and_order_preserving(make_example_fn):
    train = [
        make_example_fn("t1", "clean one", "a.jpg", split="train"),
        make_example_fn("t2", "leaked", "b.jpg", split="train"),
        make_example_fn("t3", "clean two", "c.jpg", split="train"),
    ]
    dev = [make_example_fn("d1", "leaked", "z.jpg", split="dev")]
    test: list = []

    result_a = remediate_train_leakage(train, dev, test)
    result_b = remediate_train_leakage(train, dev, test)

    ids_a = [ex.example_id for ex in result_a.clean_train]
    ids_b = [ex.example_id for ex in result_b.clean_train]
    assert ids_a == ids_b == ["t1", "t3"]  # original relative order preserved
