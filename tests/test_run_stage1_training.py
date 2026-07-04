def test_clear_cache_removes_dat(tmp_path):
    from scripts.run_stage1_training import _clear_cache

    npz = tmp_path / "stage1_binary.npz"
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir()
    dat = cache_dir / "stage1_binary_zscore_float16.dat"
    dat.write_text("dummy")
    _clear_cache(npz)
    assert not dat.exists()
