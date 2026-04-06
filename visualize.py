"""
B-Spline Neural Network — Inference Visualizer
================================================
Usage:
    python visualize.py [--ckpt checkpoints/best.pt] [--port 8050]

Then open  http://localhost:8050

Navigation hierarchy
--------------------
Architecture view
  └── click a Block  ──► Block view
                            ├── click Attention ──► Attention heatmap
                            └── click FFN       ──► B-Spline bubble chart
                                                      └── click bubble ──► B-Spline detail
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import dash
from dash import Dash, dcc, html, Input, Output, State, ctx, no_update, ALL
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="B-Spline Inference Visualizer")
    p.add_argument("--ckpt", default="", help="Path to .pt checkpoint")
    p.add_argument("--port", type=int, default=8050)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

_BG      = "#0E0E1A"
_SURFACE = "#16162A"
_BORDER  = "#2A2A45"
_TEXT    = "#E2E2F0"
_SUBTEXT = "#8888B0"
_ACCENT  = "#7B61FF"
_POS     = "#5EE8A0"
_NEG     = "#FF6B6B"
_MUTED   = "#3A3A5C"
_GRID    = "#1E1E35"


def _lerp_color(t: float) -> str:
    """Muted → accent based on normalised [0, 1] activation level."""
    r0, g0, b0 = 0x3A, 0x3A, 0x5C
    r1, g1, b1 = 0x7B, 0x61, 0xFF
    t = float(np.clip(t, 0, 1))
    return (f"rgb({int(r0+t*(r1-r0))},"
            f"{int(g0+t*(g1-g0))},"
            f"{int(b0+t*(b1-b0))})")


def _empty_fig(msg: str = "No data") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False, font=dict(color=_SUBTEXT, size=14),
    )
    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=20, b=20), height=340,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Architecture view
# ---------------------------------------------------------------------------

def _fig_arch(store: dict, threshold: float) -> go.Figure:
    n_layers  = store["n_layers"]
    residuals = store.get("block_residuals", {})

    vals     = [residuals.get(str(i), 0.0) for i in range(n_layers)]
    max_val  = max(vals) if vals else 1.0
    nv       = [v / (max_val + 1e-9) for v in vals]

    labels   = ["Embed"] + [f"Block {i}" for i in range(n_layers)] + ["LM Head"]
    node_ids = ["embed"] + [str(i) for i in range(n_layers)] + ["lmhead"]
    all_nv   = [0.0] + nv + [0.0]
    total    = len(labels)
    xs       = list(range(total))

    colors  = [_lerp_color(v) for v in all_nv]
    borders = [
        _ACCENT if (node_ids[i] not in ("embed", "lmhead") and all_nv[i] >= threshold)
        else _BORDER
        for i in range(total)
    ]
    widths  = [3 if b == _ACCENT else 1 for b in borders]

    # Hover text with activation values
    hover = [
        f"<b>{labels[i]}</b><br>Mean |act|: {vals[i-1]:.4f}<br><i>Click to inspect</i>"
        if node_ids[i] not in ("embed", "lmhead")
        else f"<b>{labels[i]}</b>"
        for i in range(total)
    ]

    fig = go.Figure()

    # Connector lines
    for i in range(total - 1):
        fig.add_annotation(
            x=xs[i + 1] - 0.38, y=0,
            ax=xs[i] + 0.38, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor=_BORDER, arrowwidth=2,
        )

    fig.add_trace(go.Scatter(
        x=xs, y=[0] * total,
        mode="markers+text",
        marker=dict(
            size=56,
            color=colors,
            line=dict(color=borders, width=widths),
            symbol="square",
        ),
        text=labels,
        textposition="middle center",
        textfont=dict(color=_TEXT, size=10, family="monospace"),
        customdata=node_ids,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover,
    ))

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        xaxis=dict(visible=False, range=[-0.7, total - 0.3]),
        yaxis=dict(visible=False, range=[-1.2, 1.2]),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(
            text="Architecture — click a Block to drill in",
            font=dict(color=_TEXT, size=14),
        ),
        showlegend=False,
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Block view
# ---------------------------------------------------------------------------

def _fig_block(store: dict, block_idx: int, threshold: float) -> go.Figure:
    bidx      = str(block_idx)
    residuals = store.get("block_residuals", {})
    max_val   = max(residuals.values()) if residuals else 1.0
    bval      = residuals.get(bidx, 0.0)
    norm      = bval / (max_val + 1e-9)

    labels   = ["Input", "Attention", "FFN", "Output"]
    node_ids = ["input", "attn", "ffn", "output"]
    nv       = [0.0, norm, norm, 0.0]
    xs       = [0, 1, 2, 3]

    colors  = [_lerp_color(v) for v in nv]
    borders = [
        _ACCENT if (node_ids[i] in ("attn", "ffn") and nv[i] >= threshold)
        else _BORDER
        for i in range(4)
    ]
    widths  = [3 if b == _ACCENT else 1 for b in borders]
    hover   = [
        f"<b>{lbl}</b><br><i>Click to inspect</i>"
        if node_ids[i] in ("attn", "ffn") else f"<b>{lbl}</b>"
        for i, lbl in enumerate(labels)
    ]

    fig = go.Figure()
    for i in range(3):
        fig.add_annotation(
            x=xs[i + 1] - 0.38, y=0,
            ax=xs[i] + 0.38, ay=0,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True, arrowhead=2, arrowsize=1.2,
            arrowcolor=_BORDER, arrowwidth=2,
        )

    fig.add_trace(go.Scatter(
        x=xs, y=[0] * 4,
        mode="markers+text",
        marker=dict(size=64, color=colors,
                    line=dict(color=borders, width=widths), symbol="square"),
        text=labels,
        textposition="middle center",
        textfont=dict(color=_TEXT, size=12, family="monospace"),
        customdata=node_ids,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover,
    ))

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        xaxis=dict(visible=False, range=[-0.8, 3.8]),
        yaxis=dict(visible=False, range=[-1.2, 1.2]),
        margin=dict(l=10, r=10, t=40, b=10),
        title=dict(text=f"Block {block_idx} — click Attention or FFN",
                   font=dict(color=_TEXT, size=14)),
        showlegend=False,
        height=260,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: Attention heatmap
# ---------------------------------------------------------------------------

def _fig_attn(store: dict, block_idx: int,
               head_idx: int, threshold: float) -> go.Figure:
    raw = store.get("attn_weights", {}).get(str(block_idx))
    if raw is None:
        return _empty_fig("No attention data captured.")

    attn = np.array(raw, dtype=np.float32)  # (H, T, T)
    H, T, _ = attn.shape
    tok = store.get("token_labels", [str(i) for i in range(T)])

    if head_idx < 0:
        mat   = attn.mean(axis=0)
        title = f"Block {block_idx} · Attention (avg {H} heads)"
    else:
        mat   = attn[min(head_idx, H - 1)]
        title = f"Block {block_idx} · Attention head {head_idx}"

    fig = px.imshow(
        mat,
        color_continuous_scale="Viridis",
        zmin=0.0, zmax=float(mat.max()) or 1.0,
        labels=dict(x="Key token", y="Query token", color="Weight"),
        x=tok[:T], y=tok[:T],
        title=title,
    )

    # Highlight cells above threshold
    for r, c in np.argwhere(mat >= threshold):
        fig.add_shape(
            type="rect",
            x0=c - 0.5, x1=c + 0.5, y0=r - 0.5, y1=r + 0.5,
            line=dict(color=_ACCENT, width=1.5),
            fillcolor="rgba(0,0,0,0)",
        )

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        margin=dict(l=60, r=20, t=50, b=60),
        coloraxis_colorbar=dict(tickfont=dict(color=_TEXT)),
        title=dict(font=dict(color=_TEXT, size=13)),
        xaxis=dict(tickfont=dict(color=_SUBTEXT), side="bottom"),
        yaxis=dict(tickfont=dict(color=_SUBTEXT)),
        height=min(80 * T + 120, 700),
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: FFN / B-Spline bubble chart
# ---------------------------------------------------------------------------

def _fig_ffn(store: dict, block_idx: int,
              layer_name: str, threshold: float) -> go.Figure:
    kan_layer = store.get("kan", {}).get(str(block_idx), {}).get(layer_name)
    if kan_layer is None:
        return _empty_fig("No KAN data. Is ffn_type='kan' in the config?")

    splines = kan_layer["splines"]  # list of dicts, sorted by mean_abs desc

    ix, jx, sz, col, cdat, htxt = [], [], [], [], [], []
    max_abs = max((s["mean_abs"] for s in splines), default=1.0)

    for s in splines:
        v = s["mean_abs"]
        if v < threshold * max_abs:
            continue
        ix.append(s["i"])
        jx.append(s["j"])
        sz.append(v)
        col.append(s["mean_val"])
        cdat.append(f"{s['i']},{s['j']}")
        htxt.append(
            f"Input dim: {s['i']}<br>"
            f"Output dim: {s['j']}<br>"
            f"Mean |contrib|: {v:.5f}<br>"
            f"<i>Click to inspect B-Spline</i>"
        )

    if not ix:
        return _empty_fig(
            f"No splines above threshold ({threshold:.2f}). "
            "Try lowering the threshold."
        )

    max_sz = max(sz)
    marker_sizes = [max(5, 36 * v / (max_sz + 1e-9)) for v in sz]
    abs_col = [abs(c) for c in col]

    fig = go.Figure(go.Scatter(
        x=ix, y=jx,
        mode="markers",
        marker=dict(
            size=marker_sizes,
            color=col,
            colorscale=[[0, _NEG], [0.5, _MUTED], [1, _POS]],
            cmin=-max_abs, cmax=max_abs,
            colorbar=dict(
                title="Mean contrib",
                tickfont=dict(color=_TEXT),
                title_font=dict(color=_SUBTEXT),
            ),
            line=dict(color=_BORDER, width=0.4),
            opacity=0.82,
        ),
        customdata=cdat,
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=htxt,
    ))

    in_f  = kan_layer["in_features"]
    out_f = kan_layer["out_features"]
    shown = len(ix)

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        xaxis=dict(
            title=f"Input dimension (0 – {in_f-1})",
            gridcolor=_GRID, tickfont=dict(color=_SUBTEXT),
        ),
        yaxis=dict(
            title=f"Output dimension (0 – {out_f-1})",
            gridcolor=_GRID, tickfont=dict(color=_SUBTEXT),
        ),
        margin=dict(l=70, r=20, t=60, b=60),
        title=dict(
            text=(
                f"Block {block_idx} · {layer_name}  "
                f"({shown} splines shown) — click a bubble"
            ),
            font=dict(color=_TEXT, size=13),
        ),
        height=560,
    )
    return fig


# ---------------------------------------------------------------------------
# Figure: B-Spline detail
# ---------------------------------------------------------------------------

def _fig_bspline(store: dict, block_idx: int,
                  layer_name: str, si: int, sj: int) -> go.Figure:
    from classes.inference_hooks import _compute_basis_np

    kan_layer = store.get("kan", {}).get(str(block_idx), {}).get(layer_name)
    if kan_layer is None:
        return _empty_fig("No KAN data available.")

    # Find the stored spline entry for (si, sj)
    entry = next(
        (s for s in kan_layer["splines"] if s["i"] == si and s["j"] == sj),
        None,
    )
    if entry is None:
        return _empty_fig(f"Spline ({si}, {sj}) not in store (below top-{500} threshold).")

    cp_ij   = np.array(entry["control_points"], dtype=np.float32)  # (num_basis,)
    x_val   = float(entry["x_scaled_mean"])
    kv      = np.array(kan_layer["knot_vector"], dtype=np.float32)
    degree  = int(kan_layer["degree"])
    num_ctrl = int(kan_layer["num_ctrl"])
    num_basis = int(kan_layer["num_basis"])

    # Dense evaluation grid
    t_grid = np.linspace(0, 1, 400, dtype=np.float32)[:, np.newaxis]  # (400, 1)
    basis_grid = _compute_basis_np(t_grid, kv, num_ctrl, degree)[:, 0, :]  # (400, nb)
    spline_curve = basis_grid @ cp_ij  # (400,)

    # Basis at the actual x_val
    x_q      = np.array([[x_val]], dtype=np.float32)
    basis_at = _compute_basis_np(x_q, kv, num_ctrl, degree)[0, 0, :]  # (nb,)
    active   = basis_at > 1e-6
    spline_at_x = float(basis_at @ cp_ij)

    fig = go.Figure()
    t_1d = t_grid[:, 0].tolist()

    # Individual basis functions
    for k in range(num_basis):
        is_act = bool(active[k])
        fig.add_trace(go.Scatter(
            x=t_1d,
            y=basis_grid[:, k].tolist(),
            mode="lines",
            line=dict(
                color=_ACCENT if is_act else _MUTED,
                width=1.6 if is_act else 0.7,
                dash="dot",
            ),
            opacity=0.75 if is_act else 0.2,
            name=f"N\u2080{k}(t)" + (" \u2713" if is_act else ""),
        ))

    # Full spline curve (on top)
    fig.add_trace(go.Scatter(
        x=t_1d,
        y=spline_curve.tolist(),
        mode="lines",
        line=dict(color=_TEXT, width=2.8),
        name="\u03c6(t)  [spline]",
    ))

    # Active knot-span shading
    active_idx = np.where(active)[0]
    if len(active_idx):
        lo_knot = max(0, active_idx[0])
        hi_knot = min(len(kv) - 1, active_idx[-1] + degree + 1)
        span_lo = float(kv[lo_knot])
        span_hi = float(kv[hi_knot])
        fig.add_vrect(
            x0=span_lo, x1=span_hi,
            fillcolor=_ACCENT, opacity=0.09,
            layer="below", line_width=0,
            annotation_text="active knot span",
            annotation_font=dict(color=_SUBTEXT, size=10),
        )

    # Vertical marker at actual input position
    fig.add_vline(
        x=x_val,
        line=dict(color=_NEG, width=2.0, dash="dash"),
        annotation=dict(
            text=f"x\u209c={x_val:.3f}",
            font=dict(color=_NEG, size=11),
            yanchor="top",
        ),
    )

    # Output dot
    fig.add_trace(go.Scatter(
        x=[x_val], y=[spline_at_x],
        mode="markers",
        marker=dict(color=_NEG, size=11, symbol="circle",
                    line=dict(color=_TEXT, width=1)),
        name=f"\u03c6({x_val:.3f}) = {spline_at_x:.4f}",
    ))

    range_min = float(kan_layer["range_min"])
    range_max = float(kan_layer["range_max"])
    x_actual  = x_val * (range_max - range_min) + range_min

    fig.update_layout(
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(color=_TEXT),
        xaxis=dict(
            title="Scaled input  t \u2208 [0, 1]",
            gridcolor=_GRID, tickfont=dict(color=_SUBTEXT),
        ),
        yaxis=dict(
            title="Spline output value",
            gridcolor=_GRID, tickfont=dict(color=_SUBTEXT),
        ),
        legend=dict(
            bgcolor=_SURFACE, bordercolor=_BORDER,
            font=dict(color=_TEXT, size=10),
        ),
        margin=dict(l=65, r=20, t=65, b=65),
        title=dict(
            text=(
                f"Block {block_idx} \u00b7 {layer_name} \u00b7 "
                f"Spline ({si} \u2192 {sj})\u2002\u2002"
                f"x={x_actual:.3f}  \u03c6={spline_at_x:.4f}"
            ),
            font=dict(color=_TEXT, size=13),
        ),
        height=560,
    )
    return fig


# ---------------------------------------------------------------------------
# Breadcrumb builder
# ---------------------------------------------------------------------------

def _breadcrumbs(nav: dict) -> list:
    view  = nav.get("view", "arch")
    bidx  = nav.get("block_idx")
    lname = nav.get("layer_name")
    si    = nav.get("spline_i")
    sj    = nav.get("spline_j")

    crumbs: list[tuple[str, dict]] = [("Architecture", {"view": "arch"})]

    if view in ("block", "attn", "ffn", "bspline") and bidx is not None:
        crumbs.append((f"Block {bidx}", {"view": "block", "block_idx": bidx}))

    if view == "attn":
        crumbs.append(("Attention", {"view": "attn", "block_idx": bidx}))

    if view in ("ffn", "bspline") and lname:
        crumbs.append(("FFN", {"view": "ffn", "block_idx": bidx, "layer_name": lname}))

    if view == "bspline" and si is not None:
        crumbs.append((
            f"Spline ({si}\u2192{sj})",
            {"view": "bspline", "block_idx": bidx,
             "layer_name": lname, "spline_i": si, "spline_j": sj},
        ))

    items = []
    for idx, (label, nav_payload) in enumerate(crumbs):
        is_last = (idx == len(crumbs) - 1)
        if not is_last:
            items.append(html.Span(
                label,
                id={"type": "crumb", "index": idx},
                n_clicks=0,
                **{"data-nav": json.dumps(nav_payload)},
                style={"color": _ACCENT, "cursor": "pointer",
                       "textDecoration": "underline"},
            ))
            items.append(html.Span(
                " › ", style={"color": _SUBTEXT, "margin": "0 5px"}
            ))
        else:
            items.append(html.Span(
                label, style={"color": _TEXT, "fontWeight": "600"}
            ))

    return items


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

_LABEL = {"color": _SUBTEXT, "fontSize": "11px",
           "marginBottom": "4px", "display": "block"}

_INPUT = {
    "width": "100%", "backgroundColor": _BG, "color": _TEXT,
    "border": f"1px solid {_BORDER}", "borderRadius": "4px",
    "padding": "6px 8px", "fontSize": "12px", "fontFamily": "monospace",
    "boxSizing": "border-box",
}

_CARD = {
    "backgroundColor": _SURFACE,
    "border": f"1px solid {_BORDER}",
    "borderRadius": "8px",
    "padding": "16px",
}

_BTN = {
    "width": "100%", "padding": "10px", "marginTop": "12px",
    "backgroundColor": _ACCENT, "color": "#fff", "border": "none",
    "borderRadius": "6px", "cursor": "pointer",
    "fontWeight": "600", "fontSize": "13px",
}

app = Dash(__name__, title="B-Spline Visualizer",
           suppress_callback_exceptions=True)

app.layout = html.Div(
    style={"backgroundColor": _BG, "minHeight": "100vh",
           "fontFamily": "'Inter','Segoe UI',sans-serif",
           "color": _TEXT, "padding": "20px"},
    children=[
        html.H2("B-Spline Neural Network — Inference Visualizer",
                style={"color": _TEXT, "marginBottom": "20px",
                       "fontWeight": "700"}),

        html.Div(
            style={"display": "flex", "gap": "20px", "alignItems": "flex-start"},
            children=[
                # ── Controls panel ─────────────────────────────────────────
                html.Div(
                    style={**_CARD, "width": "270px", "flexShrink": "0"},
                    children=[
                        html.H4("Controls", style={"color": _TEXT,
                                                    "marginTop": 0,
                                                    "marginBottom": "16px"}),

                        html.Label("Checkpoint path", style=_LABEL),
                        dcc.Input(id="ckpt-input", type="text",
                                  placeholder="checkpoints/best.pt",
                                  debounce=False, style=_INPUT),

                        html.Label("Prompt", style={**_LABEL, "marginTop": "12px"}),
                        dcc.Textarea(
                            id="prompt-input",
                            placeholder="Once upon a time…",
                            style={**_INPUT, "height": "72px", "resize": "vertical"},
                        ),

                        html.Label("Activation threshold",
                                   style={**_LABEL, "marginTop": "14px"}),
                        dcc.Slider(
                            id="threshold-slider",
                            min=0.0, max=1.0, step=0.01, value=0.1,
                            marks={
                                0:   {"label": "0",   "style": {"color": _SUBTEXT}},
                                0.5: {"label": "0.5", "style": {"color": _SUBTEXT}},
                                1:   {"label": "1",   "style": {"color": _SUBTEXT}},
                            },
                            tooltip={"always_visible": True, "placement": "bottom",
                                     "style": {"color": _TEXT, "fontSize": "11px"}},
                        ),

                        html.Button("Run Inference", id="run-btn",
                                    n_clicks=0, style=_BTN),

                        html.Div(id="status-msg",
                                 style={"color": _SUBTEXT, "fontSize": "11px",
                                        "marginTop": "10px", "minHeight": "20px"}),

                        # Head selector (shown only in attention view)
                        html.Div(
                            id="head-container",
                            style={"display": "none", "marginTop": "14px"},
                            children=[
                                html.Label("Attention head", style=_LABEL),
                                dcc.Dropdown(
                                    id="head-dropdown",
                                    value=-1,
                                    clearable=False,
                                    style={
                                        "backgroundColor": _BG,
                                        "color": "#000",
                                        "border": f"1px solid {_BORDER}",
                                        "fontSize": "12px",
                                    },
                                ),
                            ],
                        ),

                        # KAN layer radio (shown in FFN view)
                        html.Div(
                            id="kan-container",
                            style={"display": "none", "marginTop": "14px"},
                            children=[
                                html.Label("KAN layer", style=_LABEL),
                                dcc.RadioItems(
                                    id="kan-radio",
                                    options=[
                                        {"label": " kan1 (in → hidden)", "value": "kan1"},
                                        {"label": " kan2 (hidden → out)", "value": "kan2"},
                                    ],
                                    value="kan1",
                                    style={"color": _TEXT, "fontSize": "12px"},
                                    labelStyle={"display": "block", "marginBottom": "4px"},
                                ),
                            ],
                        ),

                        # Token chips
                        html.Div(id="token-display",
                                 style={"marginTop": "18px",
                                        "fontSize": "10px",
                                        "color": _SUBTEXT,
                                        "lineHeight": "2.0"}),
                    ],
                ),

                # ── Visualisation panel ────────────────────────────────────
                html.Div(
                    style={"flex": "1", "minWidth": "0"},
                    children=[
                        html.Div(
                            id="breadcrumb-bar",
                            style={
                                "fontSize": "13px", "marginBottom": "12px",
                                "padding": "8px 14px",
                                "backgroundColor": _SURFACE,
                                "border": f"1px solid {_BORDER}",
                                "borderRadius": "6px",
                            },
                        ),
                        dcc.Graph(
                            id="main-graph",
                            config={"displayModeBar": False},
                            figure=_empty_fig(
                                "Load a checkpoint, enter a prompt, "
                                "and click  Run Inference."
                            ),
                        ),
                    ],
                ),
            ],
        ),

        # ── Hidden state ────────────────────────────────────────────────────
        dcc.Store(id="act-store",  storage_type="memory", data={}),
        dcc.Store(id="nav-store",  storage_type="memory", data={"view": "arch"}),
        # Relay for breadcrumb clicks (span → store)
        html.Div(id="crumb-relay", style={"display": "none"}),
    ],
)

# Global CSS
app.index_string = """<!DOCTYPE html>
<html>
  <head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
      * { box-sizing: border-box; }
      body { margin: 0; background-color: #0E0E1A; }
      ::-webkit-scrollbar { width: 6px; }
      ::-webkit-scrollbar-track  { background: #0E0E1A; }
      ::-webkit-scrollbar-thumb  { background: #2A2A45; border-radius: 3px; }
      .rc-slider-track  { background-color: #7B61FF !important; }
      .rc-slider-handle { border-color: #7B61FF !important;
                          background-color: #7B61FF !important; }
      .rc-slider-rail   { background-color: #2A2A45 !important; }
    </style>
  </head>
  <body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
  </body>
</html>"""


# ---------------------------------------------------------------------------
# Callback: Run Inference
# ---------------------------------------------------------------------------

@app.callback(
    Output("act-store",    "data"),
    Output("status-msg",   "children"),
    Output("token-display","children"),
    Input("run-btn", "n_clicks"),
    State("ckpt-input",   "value"),
    State("prompt-input", "value"),
    prevent_initial_call=True,
)
def cb_run_inference(_, ckpt_path, prompt):
    if not ckpt_path:
        return no_update, "Enter a checkpoint path.", ""
    if not prompt:
        return no_update, "Enter a prompt.", ""

    ckpt_path = ckpt_path.strip()
    if not Path(ckpt_path).exists():
        return no_update, f"\u26a0 Not found: {ckpt_path}", ""

    try:
        import torch
        from classes.inference_hooks import load_model_from_checkpoint, HookedModel

        device = (
            "cuda" if torch.cuda.is_available() else
            "mps"  if torch.backends.mps.is_available() else
            "cpu"
        )
        model, tokenizer, _ = load_model_from_checkpoint(ckpt_path, device=device)
        store = HookedModel(model, tokenizer).run(prompt)

        toks = store.get("token_labels", [])
        chips = [
            html.Span(
                t,
                style={
                    "display": "inline-block",
                    "backgroundColor": _SURFACE,
                    "border": f"1px solid {_BORDER}",
                    "borderRadius": "3px",
                    "padding": "1px 5px",
                    "margin": "2px",
                    "fontSize": "10px",
                    "fontFamily": "monospace",
                },
            )
            for t in toks
        ]
        status = (
            f"\u2713 {len(toks)} tokens \u00b7 "
            f"{store['n_layers']} blocks \u00b7 "
            f"{store['ffn_type'].upper()} FFN \u00b7 {device}"
        )
        return store, status, chips

    except Exception as exc:
        return no_update, f"\u26a0 {exc}", ""


# ---------------------------------------------------------------------------
# Callback: breadcrumb span clicks → relay
# ---------------------------------------------------------------------------

@app.callback(
    Output("crumb-relay", "children"),
    Input({"type": "crumb", "index": ALL}, "n_clicks"),
    State({"type": "crumb", "index": ALL}, "data-nav"),
    prevent_initial_call=True,
)
def cb_breadcrumb(n_list, nav_list):
    if not any(n_list):
        return no_update
    triggered = ctx.triggered_id
    if triggered is None:
        return no_update
    idx = triggered["index"]
    return nav_list[idx] if nav_list and idx < len(nav_list) else no_update


# ---------------------------------------------------------------------------
# Callback: graph clicks + crumb relay → update nav-store
# ---------------------------------------------------------------------------

@app.callback(
    Output("nav-store", "data"),
    Input("main-graph",   "clickData"),
    Input("crumb-relay",  "children"),
    State("nav-store",    "data"),
    State("act-store",    "data"),
    State("kan-radio",    "value"),
    prevent_initial_call=True,
)
def cb_update_nav(click_data, crumb_nav, nav, store, kan_layer):
    triggered = ctx.triggered_id

    if triggered == "crumb-relay" and crumb_nav:
        try:
            return json.loads(crumb_nav)
        except Exception:
            return nav

    if not click_data or not store:
        return nav

    custom = click_data["points"][0].get("customdata")
    if custom is None:
        return nav

    view = nav.get("view", "arch")

    if view == "arch":
        s = str(custom)
        if s.lstrip("-").isdigit():
            return {"view": "block", "block_idx": int(s)}

    elif view == "block":
        bidx = nav.get("block_idx", 0)
        if custom == "attn":
            return {"view": "attn", "block_idx": bidx}
        if custom == "ffn":
            return {"view": "ffn", "block_idx": bidx,
                    "layer_name": kan_layer or "kan1"}

    elif view == "ffn":
        bidx   = nav.get("block_idx", 0)
        lname  = nav.get("layer_name", "kan1")
        if isinstance(custom, str) and "," in custom:
            si, sj = custom.split(",")
            return {
                "view": "bspline",
                "block_idx": bidx,
                "layer_name": lname,
                "spline_i": int(si),
                "spline_j": int(sj),
            }

    return nav


# ---------------------------------------------------------------------------
# Callback: render view
# ---------------------------------------------------------------------------

@app.callback(
    Output("main-graph",     "figure"),
    Output("breadcrumb-bar", "children"),
    Output("head-container", "style"),
    Output("head-dropdown",  "options"),
    Output("head-dropdown",  "value"),
    Output("kan-container",  "style"),
    Input("nav-store",         "data"),
    Input("threshold-slider",  "value"),
    Input("head-dropdown",     "value"),
    Input("kan-radio",         "value"),
    State("act-store",         "data"),
    prevent_initial_call=True,
)
def cb_render(nav, threshold, head_idx, kan_layer, store):
    view   = nav.get("view", "arch")
    bidx   = nav.get("block_idx", 0)
    lname  = nav.get("layer_name", "kan1")
    si     = nav.get("spline_i", 0)
    sj     = nav.get("spline_j", 0)

    head_style = {"display": "none",  "marginTop": "14px"}
    kan_style  = {"display": "none",  "marginTop": "14px"}
    head_opts  = [{"label": "Average", "value": -1}]
    head_val   = head_idx if head_idx is not None else -1

    crumbs = _breadcrumbs(nav)

    if not store:
        return (
            _empty_fig("Run inference first."),
            crumbs, head_style, head_opts, head_val, kan_style,
        )

    if view == "arch":
        fig = _fig_arch(store, threshold)

    elif view == "block":
        fig = _fig_block(store, bidx, threshold)

    elif view == "attn":
        raw = store.get("attn_weights", {}).get(str(bidx))
        if raw:
            H = len(raw)
            head_opts = [{"label": "Average", "value": -1}] + [
                {"label": f"Head {h}", "value": h} for h in range(H)
            ]
        head_style = {"display": "block", "marginTop": "14px"}
        fig = _fig_attn(store, bidx, head_val, threshold)

    elif view == "ffn":
        kan_style = {"display": "block", "marginTop": "14px"}
        # Sync layer_name from radio when user changes it
        lname = kan_layer if kan_layer else lname
        fig = _fig_ffn(store, bidx, lname, threshold)

    elif view == "bspline":
        fig = _fig_bspline(store, bidx, lname, si, sj)

    else:
        fig = _empty_fig(f"Unknown view: {view}")

    return fig, crumbs, head_style, head_opts, head_val, kan_style


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = _parse_args()
    app.run(debug=True, port=args.port, host="127.0.0.1")
