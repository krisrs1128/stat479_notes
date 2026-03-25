"""
helpers to run emotion probes on transformer hidden states.

1. load_envent()         – load & label the enVent dataset
2. extract_hidden_states() – collect layer outputs via forward hooks
3. train_probes()        – fit logistic-regression probes per (layer, token)
"""
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

EMOTIONS = [
    'anger', 'boredom', 'disgust', 'fear', 'guilt', 'joy',
    'neutral', 'pride', 'relief', 'sadness', 'shame', 'surprise', 'trust',
]
EMOTION_TO_ID = {e: i for i, e in enumerate(EMOTIONS)}


def load_envent(csv_path: str) -> pd.DataFrame:
    """Load the enVent dataset and add prompt / label columns.

    Parameters
    ----------
    csv_path : path to enVent_gen_Data.csv

    Returns
    -------
    DataFrame with columns: emotion, emotion_id, prompt
    """
    df = pd.read_csv(csv_path, encoding='ISO-8859-1')
    df['emotion'] = df['emotion'].replace('no-emotion', 'neutral')
    df = df[df['emotion'].isin(EMOTIONS)].reset_index(drop=True)
    df['emotion_id'] = df['emotion'].map(EMOTION_TO_ID)
    df['prompt'] = df['hidden_emo_text'].apply(
        lambda t: (
            "What are the inferred emotions in the following contexts? "
            f"Context: {t} Answer:"
        )
    )
    return df


def extract_hidden_states(
    texts: list[str],
    model,
    tokenizer,
    device: str,
    token_positions: list[int] = [-1, -2, -3, -4, -5],
    batch_size: int = 4,
    max_length: int = 256,
) -> dict[int, torch.Tensor]:
    """Collect the residual-stream hidden state at the output of each layer.

    Uses a forward hook on every decoder block so no custom model wrappers are
    needed — standard ``AutoModelForCausalLM`` works fine.

    Parameters
    ----------
    texts : list of prompt strings
    model : HuggingFace causal LM
    tokenizer : matching tokenizer, will be set to left-pad
    device : 'cpu', 'cuda', or 'mps'
    token_positions : token indices to collect (negative = from end)
    batch_size : prompts per forward pass
    max_length : truncation length

    Returns
    -------
    hidden : dict  layer_idx -> Tensor[n_samples, n_positions, hidden_dim]

    Notes
    -----
    This assumes the model exposes ``model.model.layers`` (Llama, Mistral,
    Gemma, …).  For GPT-2 the attribute is ``model.transformer.h``.
    """
    # Left-pad so that position -1 is always the last *real* token.
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_layers = model.config.num_hidden_layers
    buffers: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    _cache: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def _hook(module, inp, output):
            # For decoder layers, output[0] is [batch, seq_len, hidden_dim].
            hs = output[0] if isinstance(output, tuple) else output
            _cache[idx] = hs.detach().cpu()
        return _hook

    hooks = [
        layer.register_forward_hook(make_hook(i))
        for i, layer in enumerate(model.model.layers)
    ]

    model.eval()
    for start in tqdm(range(0, len(texts), batch_size), desc="Extracting hidden states"):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch, padding='longest', truncation=True,
            max_length=max_length, return_tensors='pt',
        ).to(device)

        with torch.no_grad():
            model(**enc)

        for i in range(n_layers):
            hs = _cache[i]                                    # [B, T, D]
            pos_hs = torch.stack([hs[:, p, :] for p in token_positions], dim=1)
            buffers[i].append(pos_hs)                         # [B, P, D]

    for h in hooks:
        h.remove()

    return {i: torch.cat(buffers[i], dim=0) for i in range(n_layers)}


def train_probes(
    hidden: dict[int, torch.Tensor],
    labels: np.ndarray,
    token_positions: list[int],
) -> dict[int, dict[int, float]]:
    """Fit one logistic-regression probe per (layer, token_position) pair.

    Parameters
    ----------
    hidden : output of extract_hidden_states()
    labels : integer class labels, shape [n_samples]
    token_positions : must match those used in extract_hidden_states()

    Returns
    -------
    accuracy : dict  layer_idx -> {token_position -> held-out accuracy}
    """
    n_layers = len(hidden)
    accuracy: dict[int, dict[int, float]] = {}

    for layer in tqdm(range(n_layers), desc="Training probes"):
        accuracy[layer] = {}
        for t_idx, pos in enumerate(token_positions):
            X = hidden[layer][:, t_idx, :].numpy()
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, labels, test_size=0.2, random_state=42
            )
            clf = LogisticRegression(C=1.0, max_iter=1000)
            clf.fit(X_tr, y_tr)
            accuracy[layer][pos] = float((clf.predict(X_te) == y_te).mean())

    return accuracy