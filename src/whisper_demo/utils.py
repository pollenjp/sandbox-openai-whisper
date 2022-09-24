# Third Party Library
import torch


def get_available_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
