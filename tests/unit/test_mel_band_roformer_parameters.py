from audio_separator.separator.uvr_lib_v5.roformer.mel_band_roformer import MelBandRoformer


def test_mask_estimators_honor_mlp_expansion_factor():
    model = MelBandRoformer(
        dim=8,
        depth=1,
        num_bands=60,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        mlp_expansion_factor=1,
    )

    first_mask_mlp = model.mask_estimators[0].to_freqs[0][0]
    assert first_mask_mlp[0].out_features == 8
