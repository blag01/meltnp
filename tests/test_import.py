import subprocess
import sys
from pathlib import Path


def test_main_script_runs() -> None:
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "main.py"
    result = subprocess.run(
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "project=neural-processes-distribution-shift" in result.stdout
    assert "experiment=baseline" in result.stdout
    assert "mean_shape=(1, 2, 1)" in result.stdout


def test_attention_neural_process_shapes() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src_path = project_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    import torch

    from np_shift import AttentionNeuralProcess

    model = AttentionNeuralProcess()
    context_x = torch.tensor([[[-1.0], [0.0], [1.0]]], dtype=torch.float32)
    context_y = torch.tensor([[[-0.8], [0.1], [0.9]]], dtype=torch.float32)
    target_x = torch.tensor([[[-0.5], [0.5]]], dtype=torch.float32)

    output = model(
        context_x=context_x,
        context_y=context_y,
        target_x=target_x,
    )

    assert output.mean.shape == (1, 2, 1)
    assert output.variance.shape == (1, 2, 1)
    assert torch.all(output.variance > 0)
