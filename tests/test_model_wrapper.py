"""Unit tests for ModelWrapper layer/default-hook behavior and activations."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import brain_surgery.model_wrapper as model_wrapper_module
from brain_surgery.model_wrapper import ModelWrapper, get_default_device
from conftest import FakeCausalLM, FakeTokenizer


def test_default_layer_uses_midpoint_for_24_layers(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify the default hook index is midpoint for a 24-layer model."""
    assert mock_model_wrapper_24.total_layers == 24
    assert mock_model_wrapper_24.layer_idx == 12


def test_activations_shape_and_cpu_default(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify captured activations are (batch, seq, 896) and on CPU by default."""
    _, activations = mock_model_wrapper_24.generate_with_activations(
        prompt="What is football?",
        max_tokens=4,
    )

    assert "layer" in activations
    tensor = activations["layer"]
    assert tensor.dim() == 3

    batch, seq_len, hidden_dim = tensor.shape
    assert batch == 1
    assert seq_len > 0
    assert hidden_dim == 896
    assert tensor.device.type == "cpu"


def test_is_loaded_true_after_init(mock_model_wrapper_24: ModelWrapper) -> None:
    """Verify model/tokenizer loaded state is reported correctly."""
    assert mock_model_wrapper_24.is_loaded() is True


def test_generate_empty_prompt_raises(mock_model_wrapper_24: ModelWrapper) -> None:
    """Verify prompt validation rejects empty input."""
    with pytest.raises(ValueError):
        mock_model_wrapper_24.generate_with_activations("   ")


def test_save_activations_roundtrip(
    mock_model_wrapper_24: ModelWrapper,
    tmp_path: Path,
) -> None:
    """Verify saved activation payload has tensor fields and expected metadata."""
    _text, _acts = mock_model_wrapper_24.generate_with_activations(
        prompt="hello world",
        max_tokens=3,
    )

    out_path = mock_model_wrapper_24.save_activations(
        save_dir=tmp_path,
        batch_idx=1,
        file_stem="sample",
    )

    payload = torch.load(out_path, map_location="cpu")
    assert isinstance(payload["token_ids"], torch.Tensor)
    assert isinstance(payload["activations"], torch.Tensor)
    assert payload["layer_idx"] == 12
    assert payload["activations"].shape[1] == 896


def test_save_without_generation_raises(mock_model_wrapper_24: ModelWrapper) -> None:
    """Verify save_activations fails before any generation run."""
    with pytest.raises(RuntimeError):
        mock_model_wrapper_24.save_activations()


def test_unregister_hooks_clears_state(mock_model_wrapper_24: ModelWrapper) -> None:
    """Verify unregister_hooks removes handles and activation buffers."""
    _text, _acts = mock_model_wrapper_24.generate_with_activations("test", max_tokens=2)
    mock_model_wrapper_24.unregister_hooks()

    assert mock_model_wrapper_24.hooks == []
    assert mock_model_wrapper_24.activations == {}


def test_repr_contains_key_fields(mock_model_wrapper_24: ModelWrapper) -> None:
    """Verify repr includes model name and configured layer index."""
    text = repr(mock_model_wrapper_24)
    assert "ModelWrapper" in text
    assert "layer_idx=12" in text


def test_get_default_device_returns_torch_device() -> None:
    """Verify default device helper always returns a torch.device instance."""
    device = get_default_device()
    assert isinstance(device, torch.device)


def test_generate_with_non_callable_generate_raises(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify runtime guard for models missing callable generate API."""

    class NoGenerateModel:
        def __init__(self) -> None:
            self.generate = None
            self._param = torch.nn.Parameter(torch.zeros(1))

        def parameters(self):
            yield self._param

    monkeypatch.setattr(mock_model_wrapper_24, "model", NoGenerateModel())
    monkeypatch.setattr(
        mock_model_wrapper_24,
        "_infer_model_input_device",
        lambda: torch.device("cpu"),
    )
    with pytest.raises(RuntimeError):
        mock_model_wrapper_24.generate_with_activations("test")


def test_generate_with_output_missing_sequences_raises(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fallback generate output validation when tensor sequences are absent."""

    class BadModel:
        device = torch.device("cpu")

        def parameters(self):
            yield torch.zeros(1)

        def generate(self, *args: object, **kwargs: object) -> object:
            return SimpleNamespace(sequences=[1, 2, 3])

    monkeypatch.setattr(mock_model_wrapper_24, "model", BadModel())
    with pytest.raises(RuntimeError):
        mock_model_wrapper_24.generate_with_activations("test")


def test_save_activations_invalid_format_raises(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify unsupported serialization format is rejected."""
    with pytest.raises(ValueError):
        mock_model_wrapper_24.save_activations(fmt="ptx")  # type: ignore[arg-type]


def test_save_activations_shape_guards(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify shape guards for invalid activation tensor ranks/batch sizes."""
    mock_model_wrapper_24.activations["layer"] = torch.randn(2, 3, 896)
    mock_model_wrapper_24._last_prompt = "x"  # noqa: SLF001
    mock_model_wrapper_24._last_generated_text = "y"  # noqa: SLF001
    mock_model_wrapper_24._last_output_ids = torch.tensor([[1, 2, 3]])  # noqa: SLF001
    mock_model_wrapper_24._last_token_texts = ["a", "b", "c"]  # noqa: SLF001
    with pytest.raises(ValueError):
        mock_model_wrapper_24.save_activations()

    mock_model_wrapper_24.activations["layer"] = torch.randn(896)
    with pytest.raises(ValueError):
        mock_model_wrapper_24.save_activations()


def test_save_activations_no_tokens_raises(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify alignment guard rejects empty token/activation intersections."""
    mock_model_wrapper_24.activations["layer"] = torch.empty(0, 896)
    mock_model_wrapper_24._last_prompt = "x"  # noqa: SLF001
    mock_model_wrapper_24._last_generated_text = "y"  # noqa: SLF001
    mock_model_wrapper_24._last_output_ids = torch.tensor([[]], dtype=torch.long)  # noqa: SLF001
    mock_model_wrapper_24._last_token_texts = []  # noqa: SLF001
    with pytest.raises(ValueError):
        mock_model_wrapper_24.save_activations()


def test_gitignore_large_artifact_appends_pattern(tmp_path: Path) -> None:
    """Verify large artifact helper appends ignore patterns for oversize files."""
    artifact = tmp_path / "big.pt"
    artifact.write_bytes(b"x" * 2048)
    gitignore = tmp_path / ".gitignore"

    ModelWrapper._gitignore_large_artifact(
        artifact_path=artifact,
        gitignore_path=gitignore,
        max_mb=0,
        mode="file",
    )

    content = gitignore.read_text(encoding="utf-8")
    assert "big.pt" in content


def test_model_wrapper_main_smoke_with_stub(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify module-level main executes with a lightweight stub wrapper."""

    class StubWrapper:
        def __init__(self, model_name: str, layer_idx: int) -> None:
            _ = model_name
            _ = layer_idx

        def generate_with_activations(
            self,
            prompt: str,
            max_tokens: int,
        ) -> tuple[str, dict[str, torch.Tensor]]:
            _ = prompt
            _ = max_tokens
            return "ok", {"layer": torch.randn(1, 2, 896)}

        def save_activations(self, batch_idx: int) -> Path:
            _ = batch_idx
            return Path("dummy.pt")

        def unregister_hooks(self) -> None:
            return None

    monkeypatch.setattr(model_wrapper_module, "ModelWrapper", StubWrapper)
    model_wrapper_module.main()


def test_init_missing_and_not_directory_errors(tmp_path: Path) -> None:
    """Verify constructor path validation for missing and non-directory model paths."""
    missing = tmp_path / "missing-dir"
    with pytest.raises(FileNotFoundError):
        ModelWrapper(model_name=str(missing), layer_idx=0)

    file_path = tmp_path / "model-file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        ModelWrapper(model_name=str(file_path), layer_idx=0)


def test_infer_model_input_device_fallbacks(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify input-device inference handles string/device/no-params branches."""

    class StrDeviceModel:
        device = "cpu"

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(mock_model_wrapper_24, "model", StrDeviceModel())
    assert mock_model_wrapper_24._infer_model_input_device().type == "cpu"

    class EmptyParamsModel:
        device = None

        def parameters(self):
            if False:
                yield torch.nn.Parameter(torch.zeros(1))
            return

    mock_model_wrapper_24.device = torch.device("cpu")
    monkeypatch.setattr(mock_model_wrapper_24, "model", EmptyParamsModel())
    assert mock_model_wrapper_24._infer_model_input_device().type == "cpu"


def test_save_activations_version_and_folder_gitignore(tmp_path: Path) -> None:
    """Verify versioned filename logic and folder-level gitignore pattern."""
    model = ModelWrapper.__new__(ModelWrapper)
    model.model_name = "m"
    model.layer_idx = 1
    model.device = torch.device("cpu")
    model.activation_device = torch.device("cpu")
    model.activations = {"layer": torch.randn(3, 896)}
    model._last_prompt = "p"
    model._last_generated_text = "g"
    model._last_output_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model._last_token_texts = ["a", "b", "c"]
    model._last_token_strs = ["a", "b", "c"]

    save_dir = tmp_path / "acts"
    save_dir.mkdir()
    first = save_dir / "sample.pt"
    first.write_bytes(b"already-exists")

    out = model.save_activations(
        save_dir=save_dir,
        file_stem="sample",
        gitignore_if_large=True,
        max_mb=0,
        gitignore_path=tmp_path / ".gitignore",
        gitignore_mode="folder",
    )
    assert out.name.startswith("sample_v")
    gitignore = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "data/activations/" in gitignore


def test_save_activations_keep_device_branch(tmp_path: Path) -> None:
    """Verify device='keep' branch keeps tensor serialization flow valid."""
    model = ModelWrapper.__new__(ModelWrapper)
    model.model_name = "m"
    model.layer_idx = 1
    model.device = torch.device("cpu")
    model.activation_device = torch.device("cpu")
    model.activations = {"layer": torch.randn(3, 896)}
    model._last_prompt = "p"
    model._last_generated_text = "g"
    model._last_output_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    model._last_token_texts = ["a", "b", "c"]
    model._last_token_strs = ["a", "b", "c"]

    out = model.save_activations(
        save_dir=tmp_path,
        file_stem="keep_case",
        device="keep",
        gitignore_if_large=False,
    )
    payload = torch.load(out)
    assert isinstance(payload["activations"], torch.Tensor)


def test_generate_non_positive_max_tokens_raises(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify generation validates positive max_tokens input."""
    with pytest.raises(ValueError):
        mock_model_wrapper_24.generate_with_activations("hi", max_tokens=0)


def test_constructor_layer_index_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify constructor rejects negative and out-of-range layer indices."""
    model_dir = tmp_path / "fake"
    model_dir.mkdir()
    monkeypatch.setattr(
        "brain_surgery.model_wrapper.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: FakeCausalLM(layers=24, hidden_dim=896),
    )
    monkeypatch.setattr(
        "brain_surgery.model_wrapper.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: FakeTokenizer(),
    )

    with pytest.raises(ValueError):
        ModelWrapper(model_name=str(model_dir), layer_idx=-1)
    with pytest.raises(ValueError):
        ModelWrapper(model_name=str(model_dir), layer_idx=99)


def test_resolve_transformer_layers_error_branch(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify unsupported model architecture fails in layer resolver."""
    monkeypatch.setattr(mock_model_wrapper_24, "model", object())
    with pytest.raises(RuntimeError):
        mock_model_wrapper_24._resolve_transformer_layers()


def test_generate_without_hooks_returns_empty_activation_dict(
    mock_model_wrapper_24: ModelWrapper,
) -> None:
    """Verify generation succeeds with empty activation capture when hooks removed."""
    mock_model_wrapper_24.unregister_hooks()
    _text, activations = mock_model_wrapper_24.generate_with_activations(
        "prompt", max_tokens=2
    )
    assert activations == {}


def test_generate_decode_list_and_token_str_single_string(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify defensive decode/list and convert_ids_to_tokens string branches."""

    class OddTokenizer(FakeTokenizer):
        def decode(self, token_ids: object, skip_special_tokens: bool = True):
            _ = token_ids
            _ = skip_special_tokens
            return ["decoded-list"]

        def convert_ids_to_tokens(self, token_ids: list[int]):
            _ = token_ids
            return "tok"

    monkeypatch.setattr(mock_model_wrapper_24, "tokenizer", OddTokenizer())
    text, _activations = mock_model_wrapper_24.generate_with_activations(
        "x", max_tokens=2
    )
    assert text == "decoded-list"
    assert mock_model_wrapper_24._last_token_strs == ["tok"]  # noqa: SLF001


def test_total_layers_error_paths(mock_model_wrapper_24: ModelWrapper) -> None:
    """Verify total_layers raises for unloaded model and missing config metadata."""
    wrapper = ModelWrapper.__new__(ModelWrapper)
    wrapper.model = None
    wrapper.tokenizer = None
    with pytest.raises(RuntimeError):
        _ = wrapper.total_layers

    wrapper2 = ModelWrapper.__new__(ModelWrapper)
    wrapper2.model = SimpleNamespace(config=SimpleNamespace())
    wrapper2.tokenizer = object()
    with pytest.raises(RuntimeError):
        _ = wrapper2.total_layers


def test_resolve_transformer_layers_transformer_h_branch(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify alternate transformer.h architecture branch is supported."""
    alt = SimpleNamespace(
        transformer=SimpleNamespace(h=torch.nn.ModuleList([torch.nn.Identity()]))
    )
    monkeypatch.setattr(mock_model_wrapper_24, "model", alt)
    layers = mock_model_wrapper_24._resolve_transformer_layers()
    assert len(layers) == 1


def test_register_hooks_layer_index_out_of_range_runtime() -> None:
    """Verify _register_hooks rejects layer indices outside discovered layers."""
    wrapper = ModelWrapper.__new__(ModelWrapper)
    wrapper.layer_idx = 5
    wrapper.hooks = []
    wrapper._activation_steps = []
    wrapper.activation_device = torch.device("cpu")
    wrapper._resolve_transformer_layers = lambda: torch.nn.ModuleList(
        [torch.nn.Identity()]
    )  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        wrapper._register_hooks()


def test_generate_sets_pad_token_when_missing(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify generation sets pad_token from eos_token when pad token is missing."""

    class PadlessTokenizer(FakeTokenizer):
        pad_token_id = None
        eos_token_id = 1
        pad_token = None
        eos_token = "<eos>"

    tok = PadlessTokenizer()
    monkeypatch.setattr(mock_model_wrapper_24, "tokenizer", tok)
    _text, _acts = mock_model_wrapper_24.generate_with_activations(
        "hello", max_tokens=2
    )
    assert tok.pad_token == tok.eos_token


def test_generate_sequences_object_success_branch(
    mock_model_wrapper_24: ModelWrapper,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify non-tensor generate output with tensor sequences is accepted."""

    class SeqModel:
        device = torch.device("cpu")

        def parameters(self):
            yield torch.nn.Parameter(torch.zeros(1))

        def generate(self, *args: object, **kwargs: object) -> object:
            _ = args
            _ = kwargs
            return SimpleNamespace(
                sequences=torch.tensor([[1, 2, 3]], dtype=torch.long)
            )

    monkeypatch.setattr(mock_model_wrapper_24, "model", SeqModel())
    monkeypatch.setattr(
        mock_model_wrapper_24,
        "_infer_model_input_device",
        lambda: torch.device("cpu"),
    )
    text, _acts = mock_model_wrapper_24.generate_with_activations("hello", max_tokens=2)
    assert isinstance(text, str)
    assert isinstance(mock_model_wrapper_24._last_output_ids, torch.Tensor)  # noqa: SLF001


def test_gitignore_large_artifact_oserror_and_existing_pattern(tmp_path: Path) -> None:
    """Verify gitignore helper safely handles stat errors and duplicate patterns."""
    missing = tmp_path / "missing.pt"
    gitignore = tmp_path / ".gitignore"
    ModelWrapper._gitignore_large_artifact(
        artifact_path=missing,
        gitignore_path=gitignore,
        max_mb=0,
        mode="file",
    )
    assert not gitignore.exists()

    artifact = tmp_path / "x.pt"
    artifact.write_bytes(b"x" * 2048)
    gitignore.write_text(str(artifact).replace("\\", "/"), encoding="utf-8")
    before = gitignore.read_text(encoding="utf-8")
    ModelWrapper._gitignore_large_artifact(
        artifact_path=artifact,
        gitignore_path=gitignore,
        max_mb=0,
        mode="file",
    )
    after = gitignore.read_text(encoding="utf-8")
    assert before == after


def test_model_wrapper_main_prints_missing_layer_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify module-level main reaches 'missing key layer' print branch."""

    class StubWrapperNoLayer:
        def __init__(self, model_name: str, layer_idx: int) -> None:
            _ = model_name
            _ = layer_idx

        def generate_with_activations(
            self,
            prompt: str,
            max_tokens: int,
        ) -> tuple[str, dict[str, torch.Tensor]]:
            _ = prompt
            _ = max_tokens
            return "ok", {}

        def save_activations(self, batch_idx: int) -> Path:
            _ = batch_idx
            return Path("dummy.pt")

        def unregister_hooks(self) -> None:
            return None

    monkeypatch.setattr(model_wrapper_module, "ModelWrapper", StubWrapperNoLayer)
    model_wrapper_module.main()
