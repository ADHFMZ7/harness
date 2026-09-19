import argparse

import pytest

from harness.cli import make_llm
from harness.config import Config, ConfigError, load_config, save_setting
from harness.llm import GroqLLM, OllamaLLM


def test_a_missing_file_means_no_settings(settings_home):
    assert load_config() == (Config(), [])


def test_settings_are_read(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text(
        'provider = "groq"\nstyle = "tide"\n\n[groq]\nmodel = "openai/gpt-oss-20b"\n'
    )

    config, problems = load_config()

    assert config == Config(provider="groq", style="tide",
                            providers={"groq": {"model": "openai/gpt-oss-20b"}})
    assert problems == []


def test_every_setting_is_read(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text(
        'max_iterations = 40\ndeny = ["*.pem", "secrets"]\n\n'
        '[ollama]\nmodel = "llama3.2"\nnum_ctx = 16384\nthink = false\n\n'
        '[groq]\nmax_retries = 6\n'
    )

    config, problems = load_config()

    assert problems == []
    assert config.max_iterations == 40
    assert config.deny == ["*.pem", "secrets"]
    assert config.providers == {
        "ollama": {"model": "llama3.2", "num_ctx": 16384, "think": False},
        "groq": {"max_retries": 6},
    }


def test_values_of_the_wrong_kind_are_skipped(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text(
        'max_iterations = 0\ndeny = ["keys/id_rsa"]\n\n'
        '[ollama]\nnum_ctx = true\nthink = "no"\n\n'
        '[groq]\nmax_retries = -1\n'
    )

    config, problems = load_config()

    assert config == Config(providers={"ollama": {}, "groq": {}})
    assert len(problems) == 5
    assert "num_ctx = True under [ollama] should be a whole number above 0" in problems


def test_bad_settings_are_reported_and_skipped(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text(
        'provider = "openai"\nstyle = "tide"\nstlye = "ember"\n\n[ollama]\nmodle = "x"\n'
    )

    config, problems = load_config()

    assert config == Config(style="tide", providers={"ollama": {}})
    assert problems == [
        "provider = 'openai' should be one of ollama, groq",
        "unknown setting 'stlye'",
        "unknown setting 'modle' under [ollama]",
    ]


def test_a_file_that_is_not_toml_is_reported(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text("style = \n")

    config, problems = load_config()

    assert config == Config()
    assert len(problems) == 1 and str(settings_home) in problems[0]


def test_saving_creates_a_commented_file(settings_home):
    save_setting("style", "ember")

    assert load_config() == (Config(style="ember"), [])
    text = settings_home.read_text()
    assert text.startswith("# harness settings")
    assert text.endswith('\nstyle = "ember"\n')


def test_saving_keeps_what_the_user_wrote(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text(
        '# mine\nprovider = "groq"  # fast\n\n[groq]\nmodel = "openai/gpt-oss-20b"\n'
    )

    save_setting("style", "tide")
    save_setting("style", "ember")

    text = settings_home.read_text()
    assert "# mine" in text and "# fast" in text
    assert load_config() == (
        Config(provider="groq", style="ember", providers={"groq": {"model": "openai/gpt-oss-20b"}}),
        [],
    )


def test_saving_never_overwrites_a_broken_file(settings_home):
    settings_home.parent.mkdir(parents=True)
    settings_home.write_text("style = \n")

    with pytest.raises(ConfigError):
        save_setting("style", "tide")

    assert settings_home.read_text() == "style = \n"


def args(provider=None, model=None):
    return argparse.Namespace(provider=provider, model=model)


def test_a_flag_beats_the_settings_which_beat_the_default():
    config = Config(providers={"ollama": {"model": "llama3.2"}})

    assert make_llm(args(), Config())[1] == "qwen3.5:9b"
    assert make_llm(args(), config)[1] == "llama3.2"
    assert make_llm(args(model="mistral"), config)[1] == "mistral"


def test_provider_settings_reach_the_llm(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "unused")
    config = Config(providers={
        "ollama": {"num_ctx": 16384, "think": False},
        "groq": {"max_retries": 6},
    })

    ollama, _ = make_llm(args(), config)
    groq, _ = make_llm(args(provider="groq"), config)

    assert isinstance(ollama, OllamaLLM) and isinstance(groq, GroqLLM)
    assert (ollama.options, ollama.think) == ({"num_ctx": 16384}, False)
    assert groq.client.max_retries == 6
