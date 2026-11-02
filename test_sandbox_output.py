def test_large_output():
    print("A" * 60000)
    print("B" * 60000)
    print("C" * 60000)
    assert False, "Intentional failure for output limit testing"
