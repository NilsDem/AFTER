"""Load the frozen codec shared by offline preparation and online training."""
import gin
import torch


@gin.configurable
def load_codec(model_path=None, device="cpu", codec_type="torchscript"):
    if codec_type == "sage":
        from after.autoencoder.sage import SAGECodec
        model = SAGECodec(model_path)
    elif codec_type == "torchscript":
        if model_path is None:
            raise ValueError("TorchScript codec requires --emb_model_path")
        model = torch.jit.load(model_path, map_location="cpu")
    else:
        raise ValueError(f"Unknown codec: {codec_type}")
    return model.to(device).eval()
