from speech2summary import metrics


def test_wer_zero_and_nonzero():
    assert metrics.wer("ഇന്ന് ഞങ്ങൾ കോളേജിൽ", "ഇന്ന് ഞങ്ങൾ കോളേജിൽ") == 0
    assert metrics.wer("a b c d", "a b c x") == 0.25


def test_rouge_handles_indic():
    r = metrics.rouge("ഇന്ന് ഞങ്ങൾ കോളേജിൽ", "ഇന്ന് ഞങ്ങൾ കോളേജിൽ")
    assert r["rouge1"] == 1.0
