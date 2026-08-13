import logging

from audio_separator.separator.common_separator import CommonSeparator
from audio_separator.separator.separator import Separator


def test_external_checkpoint_and_same_named_yaml_bypass_catalogue(tmp_path):
    checkpoint = tmp_path / "registry-model.ckpt"
    config = tmp_path / "registry-model.yaml"
    checkpoint.write_bytes(b"checkpoint")
    config.write_text("model: {}\n", encoding="utf-8")

    separator = Separator.__new__(Separator)
    separator.model_file_dir = str(tmp_path)
    separator.logger = logging.getLogger(__name__)
    separator.model_is_uvr_vip = False
    separator.model_friendly_name = None

    result = separator.download_model_files(checkpoint.name)

    assert result == (
        checkpoint.name,
        "MDXC",
        "registry-model",
        str(checkpoint),
        config.name,
    )


def test_roformer_is_detected_from_registry_model_config():
    separator = CommonSeparator.__new__(CommonSeparator)
    separator.model_data = {"model": {"num_bands": 60}}
    separator.model_path = "/models/becruily_deux.ckpt"
    separator.model_name = "Becruily Deux"

    assert separator._detect_roformer_model() is True
