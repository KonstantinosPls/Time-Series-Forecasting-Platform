import torch
torch.serialization.add_safe_globals([])
import torch.serialization
original_load = torch.load
torch.load = lambda *args, **kwargs: original_load(*args, **{**kwargs, "weights_only": False})

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from lag_llama.gluon.estimator import LagLlamaEstimator
from gluonts.dataset.pandas import PandasDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_PATH = None

def get_checkpoint():
    global CKPT_PATH
    if CKPT_PATH is None:
        CKPT_PATH = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt"
        )
    return CKPT_PATH

def load_model():
    ckpt_path = get_checkpoint()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    return ckpt["hyper_parameters"]["model_kwargs"]

def forecast(model_kwargs, series, timestamps, horizon, freq=None):
    ckpt_path = get_checkpoint()
    context_length = min(len(series), 512)

    # Build GluonTS dataset with uniform index
    df = pd.DataFrame({"target": series.values.astype("float32")}, index=pd.DatetimeIndex(timestamps))
    df = df[~df.index.duplicated(keep="first")]
    inferred_freq = freq or pd.infer_freq(df.index) or "D"
    df = df.asfreq(inferred_freq)
    df["target"] = df["target"].interpolate()
    dataset = PandasDataset(df, target="target", freq=inferred_freq)

    # Create estimator with checkpoint architecture
    estimator = LagLlamaEstimator(
        ckpt_path=ckpt_path,
        prediction_length=horizon,
        context_length=context_length,
        device=DEVICE,
        input_size=model_kwargs["input_size"],
        n_layer=model_kwargs["n_layer"],
        n_embd_per_head=model_kwargs["n_embd_per_head"],
        n_head=model_kwargs["n_head"],
        scaling=model_kwargs["scaling"],
        time_feat=model_kwargs["time_feat"],
        num_parallel_samples=100,
        rope_scaling={
            "type": "linear",
            "factor": max(1.0, (context_length + horizon) / model_kwargs["context_length"]),
        },
    )

    # Build predictor
    transformation = estimator.create_transformation()
    lightning_module = estimator.create_lightning_module()
    predictor = estimator.create_predictor(transformation, lightning_module)

    # Generate forecast
    forecast_it = predictor.predict(dataset)
    forecast_entry = next(iter(forecast_it))

    median = np.median(forecast_entry.samples, axis=0)
    lower = np.percentile(forecast_entry.samples, 10, axis=0)
    upper = np.percentile(forecast_entry.samples, 90, axis=0)

    return median, lower, upper
