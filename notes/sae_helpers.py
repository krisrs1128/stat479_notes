import numpy as np
import altair as alt
import pandas as pd


def generate_v_true():
    """Return the 2D dictionary atoms used in the SAE toy example."""
    return np.array([[1.0, 0.3], [0.2, 1.0]]).T


def generate_z_true(K_true=2, N=500, rng=None):
    """Generate sparse latent coefficients for two asymmetric activation wings."""
    if rng is None:
        rng = np.random

    z_true = np.zeros((K_true, N))
    for i in range(N):
        if rng.rand() < 0.5:
            z_true[0, i] = rng.exponential(1.0)
            z_true[1, i] = rng.exponential(0.1)
        else:
            z_true[0, i] = rng.exponential(0.1)
            z_true[1, i] = rng.exponential(1.0)

    return z_true


def _arrow_df(V, labels, kind):
    rows = []
    for i, lab in enumerate(labels):
        rows.append(
            {
                "x1": 0,
                "x2": 0,
                "x1_end": V[0, i],
                "x2_end": V[1, i],
                "label": lab,
                "type": kind,
            }
        )
    return pd.DataFrame(rows)


def plot_dictionary_recovery(X, v_true, V_learned, K_true=None, width=350, height=350):
    """Plot synthetic data with true and learned dictionary vectors."""
    if K_true is None:
        K_true = v_true.shape[1]

    df_data = pd.DataFrame({"x1": X[0], "x2": X[1], "type": "data"})
    df_true = _arrow_df(v_true, [f"true v{k+1}" for k in range(K_true)], "true")
    df_learned = _arrow_df(
        V_learned,
        [f"learned v{k+1}" for k in range(V_learned.shape[1])],
        "learned",
    )
    df_arrows = pd.concat([df_true, df_learned], ignore_index=True)

    points = alt.Chart(df_data).mark_circle(size=20, opacity=0.4, color="gray").encode(
        x=alt.X("x1:Q", title="x1"),
        y=alt.Y("x2:Q", title="x2"),
    )

    arrows = alt.Chart(df_arrows).mark_line(strokeWidth=2.5).encode(
        x="x1:Q",
        y="x2:Q",
        x2="x1_end:Q",
        y2="x2_end:Q",
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["true", "learned"], range=["#e45756", "#4c78a8"]),
        ),
        strokeDash=alt.StrokeDash(
            "type:N",
            scale=alt.Scale(domain=["true", "learned"], range=[[1, 0], [5, 3]]),
        ),
    )

    arrow_points = alt.Chart(df_arrows).mark_point(size=60, filled=True).encode(
        x="x1_end:Q",
        y="x2_end:Q",
        color=alt.Color(
            "type:N",
            scale=alt.Scale(domain=["true", "learned"], range=["#e45756", "#4c78a8"]),
        ),
    )

    return (points + arrows + arrow_points).properties(width=width, height=height)
