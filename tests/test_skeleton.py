from common.seeding import make_rng


def test_rng_is_reproducible():
    assert make_rng(3).random() == make_rng(3).random()
