"""
============================================================
PER-GRAPH BIAS DIAGNOSTIC
============================================================
الهدف: نعرف هل الـ bias الكبير اللي شايفينه في بعض الجرافات
(زي صورة 4 و5) مرتبط بـ:
  - قيمة معينة في الـ global_features (ambient temp, total power...)
    طالعة بره الـ distribution اللي اتدرب عليها الموديل (extrapolation)
  - أو بعدد النودز في الجراف (الجرافات الصغيرة ممثلة أقل في التدريب)
  - أو مفيش علاقة واضحة، يبقى المشكلة مكان تاني (architecture/loss)

الكود ده بيحسب لكل جراف في test set:
  1. bias = mean(prediction - target)  بالـ Kelvin
  2. mean_abs_error للجراف
  3. كل قيمة global feature (denormalized لو أمكن)
  4. عدد النودز

وبعدين بيحسب correlation بين الـ bias وكل عمود، ويرتبهم تنازليًا،
وبيطبع أعلى/أقل N جراف من حيث الـ bias مع قيمهم عشان تقارن يدويًا.

استخدمه كده (بعد ما يخلص train_model وعندك model مدرب / أو بعد
تحميل checkpoint):

    from graph_bias_diagnostic import run_per_graph_bias_diagnostic
    df = run_per_graph_bias_diagnostic(model, loaders["test"], device,
                                        target_temp_mean, target_temp_scale,
                                        std_ambient_temp_K)
============================================================
"""

import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def collect_per_graph_records(model, loader, device, target_temp_mean, target_temp_scale):
    """
    بيرجع list of dicts، كل dict بيمثل جراف واحد من الـ test set،
    فيه: bias, mae, rmse, num_nodes, وكل global feature عمود لوحده
    (باسم global_feat_0, global_feat_1, ...  -- لو عندك أسماء حقيقية
    للـ global features استبدل الأسماء دي في GLOBAL_FEATURE_NAMES تحت).
    """
    model.eval()
    records = []

    for batch in loader:
        batch = batch.to(device)

        prediction_normalized = model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.global_features[batch.batch],
        )

        # denormalize لـ Kelvin (نفس log-space transform اللي في الكود الأصلي)
        log_pred = prediction_normalized * target_temp_scale + target_temp_mean
        prediction_kelvin = torch.exp(log_pred)

        target_kelvin = batch.y.reshape(-1)
        node_graph_id = batch.batch

        num_graphs = batch.num_graphs
        global_features = batch.global_features  # shape: [num_graphs, global_feat_dim]

        for g in range(num_graphs):
            mask = node_graph_id == g
            pred_g = prediction_kelvin[mask]
            target_g = target_kelvin[mask]

            error = (pred_g - target_g)
            bias = error.mean().item()
            mae = error.abs().mean().item()
            rmse = torch.sqrt((error ** 2).mean()).item()
            num_nodes = int(mask.sum().item())

            record = {
                "num_nodes": num_nodes,
                "bias_K": bias,
                "mae_K": mae,
                "rmse_K": rmse,
                "mean_target_K": target_g.mean().item(),
                "min_target_K": target_g.min().item(),
                "max_target_K": target_g.max().item(),
            }

            g_feats = global_features[g].detach().cpu().numpy()
            for i, val in enumerate(g_feats):
                record[f"global_feat_{i}"] = float(val)

            records.append(record)

    return records


def add_denormalized_ambient(df, ambient_col_index, std_ambient_temp_K):
    """
    لو عارف إن الـ global feature index رقم `ambient_col_index` هو
    الـ ambient temperature (زي ما ظاهر في الكود الأصلي:
    T_amb_normalized = batch.global_features[:, 0])، نرجعه لـ Kelvin
    الحقيقية عشان يبقى قابل للقراءة والمقارنة مباشرة مع الـ training range.
    """
    col = f"global_feat_{ambient_col_index}"
    if col not in df.columns:
        return df

    scale = float(np.asarray(std_ambient_temp_K.scale_).reshape(-1)[0])
    mean = float(np.asarray(std_ambient_temp_K.mean_).reshape(-1)[0])

    df["ambient_temp_K"] = df[col] * scale + mean
    return df


def run_per_graph_bias_diagnostic(
    model, test_loader, device, target_temp_mean, target_temp_scale,
    std_ambient_temp_K=None, ambient_col_index=0, top_n=10,
):
    records = collect_per_graph_records(
        model, test_loader, device, target_temp_mean, target_temp_scale
    )
    df = pd.DataFrame(records)

    if std_ambient_temp_K is not None:
        df = add_denormalized_ambient(df, ambient_col_index, std_ambient_temp_K)

    print("\n" + "=" * 70)
    print(f"PER-GRAPH DIAGNOSTIC -- {len(df)} test graphs")
    print("=" * 70)

    # -------------------------------------------------------
    # 1) Correlation بين bias وكل عمود رقمي (يشمل num_nodes وكل
    #    global feature). ده بيوريك مباشرة أي حاجة مرتبطة بالانحياز.
    # -------------------------------------------------------
    numeric_cols = [c for c in df.columns if c not in ("bias_K", "mae_K", "rmse_K")]
    correlations = {}
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            continue  # عمود ثابت، معندوش تباين يتحسب منه correlation
        corr = df[col].corr(df["bias_K"])
        correlations[col] = corr

    corr_series = pd.Series(correlations).sort_values(key=lambda s: s.abs(), ascending=False)

    print("\nCorrelation with per-graph BIAS (sorted by |correlation|):")
    print("-" * 70)
    for col, corr in corr_series.items():
        flag = "  <-- شك قوي" if abs(corr) > 0.4 else ""
        print(f"  {col:25s}: {corr:+.3f}{flag}")

    # -------------------------------------------------------
    # 2) نفس الفكرة لكن مع |bias| (عشان نلقط علاقات غير خطية بسيطة
    #    زي: كل ما num_nodes يقل، |bias| يزيد -- حتى لو اتجاه bias نفسه متغير)
    # -------------------------------------------------------
    df["abs_bias_K"] = df["bias_K"].abs()
    correlations_abs = {}
    for col in numeric_cols:
        if df[col].nunique() <= 1:
            continue
        corr = df[col].corr(df["abs_bias_K"])
        correlations_abs[col] = corr

    corr_abs_series = pd.Series(correlations_abs).sort_values(key=lambda s: s.abs(), ascending=False)

    print("\nCorrelation with |BIAS| (sorted by |correlation|):")
    print("-" * 70)
    for col, corr in corr_abs_series.items():
        flag = "  <-- شك قوي" if abs(corr) > 0.4 else ""
        print(f"  {col:25s}: {corr:+.3f}{flag}")

    # -------------------------------------------------------
    # 3) أعلى وأقل N جراف من حيث bias -- قارن قيمهم يدويًا هنا
    # -------------------------------------------------------
    print(f"\nTop {top_n} graphs by BIAS (most positive, i.e. over-predicting):")
    print("-" * 70)
    print(df.sort_values("bias_K", ascending=False).head(top_n).to_string(index=False))

    print(f"\nTop {top_n} graphs by |BIAS| (worst overall):")
    print("-" * 70)
    print(df.sort_values("abs_bias_K", ascending=False).head(top_n).to_string(index=False))

    # -------------------------------------------------------
    # 4) هل الجرافات الصغيرة (few nodes) عندها bias أكبر بشكل ممنهج؟
    #    نقسم على median عدد النودز ونقارن متوسط |bias| في كل نص.
    # -------------------------------------------------------
    median_nodes = df["num_nodes"].median()
    small_graphs = df[df["num_nodes"] <= median_nodes]
    large_graphs = df[df["num_nodes"] > median_nodes]

    print("\nSmall vs Large graphs (split at median num_nodes = "
          f"{median_nodes:.0f}):")
    print("-" * 70)
    print(f"  Small graphs (n={len(small_graphs)}): "
          f"mean |bias| = {small_graphs['abs_bias_K'].mean():.3f} K, "
          f"mean RMSE = {small_graphs['rmse_K'].mean():.3f} K")
    print(f"  Large graphs (n={len(large_graphs)}): "
          f"mean |bias| = {large_graphs['abs_bias_K'].mean():.3f} K, "
          f"mean RMSE = {large_graphs['rmse_K'].mean():.3f} K")

    if std_ambient_temp_K is not None and "ambient_temp_K" in df.columns:
        train_like_range = (
            df["ambient_temp_K"].quantile(0.05),
            df["ambient_temp_K"].quantile(0.95),
        )
        print(f"\nAmbient temp range in this test set (5th-95th pct): "
              f"{train_like_range[0]:.2f} - {train_like_range[1]:.2f} K")
        print("(قارن الرينج ده يدويًا مع رينج الـ ambient في التريننج ست بتاعك -- "
              "لو مش متطابقين، ده extrapolation فعلاً)")

    return df


if __name__ == "__main__":
    print(
        "هذا الملف Module فقط، استدعِ run_per_graph_bias_diagnostic() "
        "من كود التدريب/التقييم بتاعك بعد تحميل الموديل والـ loaders."
    )