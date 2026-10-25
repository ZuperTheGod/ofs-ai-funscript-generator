"""Can this PC's graphics card run the installed PyTorch? Used by the friend package's setup.

    python -m funscript_ai.gpu_check      exit 0 = works, 2 = card too old/new for this build, 1 = no NVIDIA card / CUDA

Checks that the build has code for the card's generation (e.g. sm_61 = GTX 10-series) AND runs the same kind of GPU
work the program does (an STFT and a small convolution).
"""
import sys


def main() -> int:
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"PyTorch not installed properly: {e}")
        return 1
    if not torch.cuda.is_available():
        print("No usable NVIDIA graphics card found (or its driver is too old - update the NVIDIA driver).")
        return 1
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    card = f"sm_{major}{minor}"
    built = torch.cuda.get_arch_list()
    print(f"GPU: {name} ({card}); PyTorch {torch.__version__} has code for {', '.join(built)}")
    import os
    if os.environ.get("JOI_TEST_REJECT") and os.environ["JOI_TEST_REJECT"] in torch.__version__:
        print("(test: pretending this build cannot run here)")
        return 2
    try:
        x = torch.randn(1, 44100, device="cuda")
        torch.stft(x, 1024, return_complex=True, window=torch.hann_window(1024, device="cuda"))
        torch.nn.functional.conv2d(torch.randn(1, 2, 32, 32, device="cuda"), torch.randn(4, 2, 3, 3, device="cuda"))
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"This PyTorch build cannot run on the {name}: {str(e).splitlines()[0]}")
        return 2
    print("GPU test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
