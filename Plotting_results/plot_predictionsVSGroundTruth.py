"""
plot_graph_temp.py

    from plot_graph_temp import plot_graph
    plot_graph(graph, result)
    plt.show()

الفروق عن النسخة القديمة:
  * نطاق الألوان بيتحسب من القيم الصالحة بس -- الأصفار والـ NaN
    مبيدخلوش، فمدى 395-410 K بيبان بتدرج حقيقي بدل لون واحد.
  * الطبقات بتتفصل في أعمدة معنونة، والـ edges اللي بين الطبقات
    بتترسم خفيفة عشان ما تعملش شبكة عنكبوت.
  * النودز اللي خطأها كبير بتاخد حلقة حمرا.
  * بيحذّرك لو فيه أصفار -- دي مشكلة داتا مش رسم.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def _arr(t):
    if t is None:
        return None
    if hasattr(t, "detach"):
        t = t.detach().cpu().numpy()
    return np.asarray(t)


def plot_graph(
    graph,
    result,
    cmap="turbo",
    fontsize=7.5,
    node_size=2200,
    figsize=(16, 7),
    error_ring_K=10,       # حلقة حمرا فوق الخطأ ده
    color_by="truth",       # "truth" | "prediction" | "error"
    show_cross_layer=True,
    save=None,
):
    pred = _arr(result).reshape(-1).astype(float)
    N = pred.size

    # ---------- ground truth ----------
    true = np.full(N, np.nan)
    gt = _arr(getattr(graph, "y", None))
    if gt is None:
        gt = _arr(getattr(graph, "real_y", None))
    if gt is not None:
        gt = gt.reshape(-1)
        if gt.size == N:
            true[:] = gt
        else:                                   # جراف قديم: real_y أقصر
            m = _arr(getattr(graph, "target_mask", None))
            if m is not None and m.astype(bool).sum() == gt.size:
                true[m.astype(bool)] = gt
            else:
                true[:gt.size] = gt

    # ---------- تحذير الأصفار ----------
    n_zero = int((pred == 0).sum() + np.nansum(true == 0))
    if n_zero:
        print(f"[plot_graph] تحذير: {n_zero} قيمة تساوي صفر بالظبط. "
              f"دي على الأغلب مشكلة داتا مش رسم -- اتشالت من مدى الألوان.")

    # ---------- الإحداثيات ----------
    pos = _arr(getattr(graph, "pos", None))
    if pos is None:
        th = np.linspace(0, 2 * np.pi, N, endpoint=False)
        x, y, layer = np.cos(th), np.sin(th), np.zeros(N, int)
    else:
        x, y = pos[:N, 0].astype(float), pos[:N, 1].astype(float)
        z = pos[:N, 2] if pos.shape[1] > 2 else np.zeros(N)
        layer = np.searchsorted(np.unique(np.round(z, 6)), np.round(z, 6))

    layers = np.unique(layer)
    if layers.size > 1:
        gap = np.ptp(x) * 1.30 + 1e-9
        x = x + layer * gap

    # ---------- الألوان ----------
    err = pred - true
    if color_by == "prediction":
        val, cm, clabel = pred, cmap, "Predicted temperature (K)"
    elif color_by == "error":
        val, cm, clabel = err, "coolwarm", "prediction - truth (K)"
    else:
        val = np.where(np.isnan(true), pred, true)
        cm, clabel = cmap, "Temperature (K)"

    # المفتاح: النطاق من القيم الصالحة فقط
    good = np.isfinite(val) & (val != 0)
    if color_by == "error":
        v = np.nanmax(np.abs(err)) if np.isfinite(err).any() else 1.0
        vmin, vmax = -v, v
    elif good.any():
        vmin, vmax = np.percentile(val[good], [1, 99])
        if vmax - vmin < 1e-6:
            vmin, vmax = val[good].min() - 1, val[good].max() + 1
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=figsize)

    # ---------- الـ edges ----------
    ei = _arr(getattr(graph, "edge_index", None))
    if ei is not None:
        seen = set()
        for s, d in zip(ei[0], ei[1]):
            if s >= N or d >= N:
                continue
            k = (min(s, d), max(s, d))
            if k in seen:
                continue
            seen.add(k)
            same = layer[s] == layer[d]
            if not same and not show_cross_layer:
                continue
            ax.plot([x[s], x[d]], [y[s], y[d]],
                    color="0.55" if same else "#8fb8d8",
                    lw=1.1 if same else 0.5,
                    alpha=0.75 if same else 0.28,
                    zorder=1 if same else 0)

    # ---------- النودز ----------
    plotted = np.where(np.isfinite(val), val, vmin)
    sc = ax.scatter(x, y, c=plotted, s=node_size, cmap=cm,
                    vmin=vmin, vmax=vmax,
                    edgecolors="black", linewidths=1.0, zorder=3)

    # حلقة حمرا للنودز الوحشة
    bad = np.isfinite(err) & (np.abs(err) > error_ring_K)
    if bad.any():
        ax.scatter(x[bad], y[bad], s=node_size * 1.30, facecolors="none",
                   edgecolors="#e53e3e", linewidths=2.2, zorder=2)

    cb = plt.colorbar(sc, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label(clabel, fontsize=10)

    # ---------- الأرقام جوه النود ----------
    norm = plt.Normalize(vmin, vmax)
    rgba = plt.get_cmap(cm)(norm(plotted))
    lum = 0.299 * rgba[:, 0] + 0.587 * rgba[:, 1] + 0.114 * rgba[:, 2]

    for i in range(N):
        if np.isfinite(true[i]):
            txt = f"{true[i]:.1f}\n{pred[i]:.1f}"
        else:
            txt = f"{pred[i]:.1f}"
        ax.annotate(txt, (x[i], y[i]), ha="center", va="center",
                    fontsize=fontsize, family="monospace", weight="bold",
                    color="white" if lum[i] < 0.55 else "black", zorder=5)

    # ---------- عناوين الطبقات ----------
    if layers.size > 1:
        for lz in layers:
            k = layer == lz
            ax.text(x[k].mean(), y.max() + 0.42 * max(np.ptp(y), 1.0),
                    f"layer {lz}", ha="center", fontsize=11,
                    fontweight="bold", color="#2b6cb0",
                    bbox=dict(boxstyle="round,pad=0.35", fc="#ebf8ff",
                              ec="#3182ce"))

    # ---------- العنوان ----------
    ok = np.isfinite(err)
    if ok.any():
        e = err[ok]
        sub = (f"RMSE {np.sqrt((e ** 2).mean()):.2f} K     "
               f"MAE {np.abs(e).mean():.2f} K     "
               f"max {np.abs(e).max():.2f} K     "
               f"bias {e.mean():+.2f} K     "
               f"{N} nodes")
    else:
        sub = f"{N} nodes, no ground truth"

    ax.set_title("Ground truth (top)  /  Prediction (bottom)\n" + sub,
                 fontsize=13, fontweight="bold")

    handles = [Line2D([], [], color="0.55", lw=1.4, label="lateral edge")]
    if show_cross_layer:
        handles.append(Line2D([], [], color="#8fb8d8", lw=1.4,
                              label="vertical edge"))
    if bad.any():
        handles.append(Line2D([], [], marker="o", ls="", mfc="none",
                              mec="#e53e3e", mew=2, ms=11,
                              label=f"|error| > {error_ring_K:g} K"))
    ax.legend(handles=handles, fontsize=9, loc="lower right", framealpha=0.9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.margins(0.10)
    ax.axis("off")
    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
    return fig, ax