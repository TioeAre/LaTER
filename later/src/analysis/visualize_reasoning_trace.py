import csv
import json
import os
import sys

import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from loguru import logger


def find_transition_idx(mask_list):
    for i in range(1, len(mask_list)):
        if mask_list[i - 1] == 0 and mask_list[i] == 1:
            return i
    return -1


def truncate_trace(h_list, mask_list, explicit_steps):
    trans_idx = find_transition_idx(mask_list)
    if explicit_steps == -1:
        return h_list, mask_list, trans_idx

    if trans_idx != -1:
        # Keep all latent steps and the first n explicit steps after transition.
        truncate_idx = min(len(h_list), trans_idx + explicit_steps)
        return h_list[:truncate_idx], mask_list[:truncate_idx], trans_idx

    if 0 not in mask_list:
        truncate_idx = min(len(h_list), explicit_steps)
        return h_list[:truncate_idx], mask_list[:truncate_idx], trans_idx

    return h_list, mask_list, trans_idx


def build_pca_records(sample_idx, source_file, h_pca, mask_list, projection_dim, explicit_steps, trans_idx, explained_variance):
    points = []
    for step, mask in enumerate(mask_list):
        point = {
            "step": step,
            "token_type": "latent" if mask == 0 else "explicit",
            "type_mask": int(mask),
            "is_start": step == 0,
            "is_end": step == len(mask_list) - 1,
            "is_transition": trans_idx != -1 and step == trans_idx,
        }
        for dim in range(projection_dim):
            point[f"pc{dim + 1}"] = float(h_pca[step, dim])
        points.append(point)

    explained_variance_ratio = [float(v) for v in explained_variance]
    return {
        "sample_idx": int(sample_idx),
        "source_file": source_file,
        "projection_dim": int(projection_dim),
        "explicit_steps": int(explicit_steps),
        "transition_idx": int(trans_idx),
        "explained_variance_ratio": explained_variance_ratio,
        "explained_variance_percent": [float(v * 100.0) for v in explained_variance_ratio],
        "points": points,
    }


def save_pca_records(records, save_dir, sample_idx, projection_dim):
    json_path = os.path.join(save_dir, f"sample_{sample_idx}_trajectory_{projection_dim}d.json")
    csv_path = os.path.join(save_dir, f"sample_{sample_idx}_trajectory_{projection_dim}d.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "sample_idx",
        "source_file",
        "projection_dim",
        "step",
        "token_type",
        "type_mask",
        "is_start",
        "is_end",
        "is_transition",
    ] + [f"pc{i + 1}" for i in range(projection_dim)]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for point in records["points"]:
            writer.writerow(
                {
                    "sample_idx": records["sample_idx"],
                    "source_file": records["source_file"],
                    "projection_dim": records["projection_dim"],
                    **point,
                }
            )

    logger.debug(f"Saved PCA trajectory data to: {json_path} and {csv_path}")


def plot_pca_trajectory_3d(pt_file_path: str, save_dir: str = "./pca_outputs", explicit_steps=100):
    logger.debug(f"Loading data from: {pt_file_path}")
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"Cannot find file {pt_file_path}")

    data = torch.load(pt_file_path, map_location="cpu")

    hidden_states_batch = data.get("hidden_states", [])
    type_masks_batch = data.get("type_masks", [])

    os.makedirs(save_dir, exist_ok=True)
    batch_size = len(hidden_states_batch)

    batch_name = os.path.basename(pt_file_path).replace(".pt", "")
    batch_idx = int(batch_name.split("_")[1])

    for b in range(batch_size):
        sample_idx = batch_idx + b
        h_list = hidden_states_batch[b]
        mask_list = type_masks_batch[b]

        if not h_list or len(h_list) < 2:
            logger.debug(f"Sample {sample_idx} has insufficient hidden states, skipping...")
            continue

        h_list, mask_list, trans_idx = truncate_trace(h_list, mask_list, explicit_steps)

        # 1. 数据预处理
        try:
            h_tensor = torch.stack([h.squeeze().float() for h in h_list])  # [num_steps, hidden_dim]
            h_array = h_tensor.numpy()
        except Exception as e:
            logger.debug(f"Error stacking hidden states for sample {sample_idx}: {e}")
            continue

        # 2. PCA 降维 (3维)
        logger.debug(f"Sample {sample_idx}: Performing 3D PCA on shape {h_array.shape}...")
        pca = PCA(n_components=3)
        h_pca = pca.fit_transform(h_array)  # [num_steps, 3]

        explained_variance = pca.explained_variance_ratio_ * 100
        records = build_pca_records(
            sample_idx=sample_idx,
            source_file=pt_file_path,
            h_pca=h_pca,
            mask_list=mask_list,
            projection_dim=3,
            explicit_steps=explicit_steps,
            trans_idx=trans_idx,
            explained_variance=pca.explained_variance_ratio_,
        )
        save_pca_records(records, save_dir, sample_idx, projection_dim=3)

        # 3. 准备 3D 绘图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

        latent_idx = [i for i, m in enumerate(mask_list) if m == 0]
        explicit_idx = [i for i, m in enumerate(mask_list) if m == 1]

        # ================== 修改点 2：绘制带颜色的连续曲线 ==================
        # 为了让线条不断开，Latent 的线条要把第一个 Explicit 点包含进去连起来
        if latent_idx and explicit_idx:
            latent_line_idx = latent_idx + [explicit_idx[0]]
        else:
            latent_line_idx = latent_idx

        if latent_line_idx:
            points = h_pca[latent_line_idx]
            n = len(points)
            cmap = plt.get_cmap("Blues")
            for i in range(n - 1):
                ax.plot(
                    points[i : i + 2, 0],
                    points[i : i + 2, 1],
                    points[i : i + 2, 2],
                    color=cmap(0.3 + 0.7 * i / max(1, n - 1)),
                    alpha=0.8,
                    linestyle="-",
                    linewidth=2.5,
                    label="Latent Trajectory" if i == 0 else "",
                    zorder=1,
                )

        if explicit_idx:
            points = h_pca[explicit_idx]
            n = len(points)
            cmap = plt.get_cmap("Oranges")
            for i in range(n - 1):
                ax.plot(
                    points[i : i + 2, 0],
                    points[i : i + 2, 1],
                    points[i : i + 2, 2],
                    color=cmap(0.3 + 0.7 * i / max(1, n - 1)),
                    alpha=0.8,
                    linestyle="-",
                    linewidth=2.5,
                    label="Explicit Trajectory" if i == 0 else "",
                    zorder=1,
                )
        # ===================================================================

        # ================== 修改点 3：按步长降采样绘制散点 ==================
        # 动态计算采样率，保证每段轨迹大概画 8-12 个点
        sample_rate_latent = max(1, len(latent_idx) // 10) if latent_idx else 1
        sample_rate_explicit = max(1, len(explicit_idx) // 10) if explicit_idx else 1

        sampled_latent = latent_idx[::sample_rate_latent] if latent_idx else []
        sampled_explicit = explicit_idx[::sample_rate_explicit] if explicit_idx else []

        if sampled_latent:
            cmap = plt.get_cmap("Blues")
            colors = [
                cmap(0.3 + 0.7 * i / max(1, len(latent_idx) - 1)) for i in range(0, len(latent_idx), sample_rate_latent)
            ]
            ax.scatter(
                h_pca[sampled_latent, 0],
                h_pca[sampled_latent, 1],
                h_pca[sampled_latent, 2],
                c=colors,
                s=40,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )

        if sampled_explicit:
            cmap = plt.get_cmap("Oranges")
            colors = [
                cmap(0.3 + 0.7 * i / max(1, len(explicit_idx) - 1))
                for i in range(0, len(explicit_idx), sample_rate_explicit)
            ]
            ax.scatter(
                h_pca[sampled_explicit, 0],
                h_pca[sampled_explicit, 1],
                h_pca[sampled_explicit, 2],
                c=colors,
                s=40,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )
        # ===================================================================

        # 5. 标记关键节点 (起点、终点、切换点)
        ax.scatter(
            h_pca[0, 0], h_pca[0, 1], h_pca[0, 2], c="green", marker="*", s=400, label="Start (Prompt)", zorder=4
        )

        if explicit_steps != -1 and (trans_idx != -1 or (0 not in mask_list)):
            end_label = f"End (Truncated {explicit_steps} steps)"
        else:
            end_label = "End (EOS)"
        ax.scatter(h_pca[-1, 0], h_pca[-1, 1], h_pca[-1, 2], c="red", marker="X", s=300, label=end_label, zorder=4)

        if trans_idx != -1 and trans_idx < len(h_pca):
            ax.scatter(
                h_pca[trans_idx, 0],
                h_pca[trans_idx, 1],
                h_pca[trans_idx, 2],
                c="purple",
                marker="D",
                s=200,
                label="Transition (Latent -> Explicit)",
                zorder=5,
            )

        # 6. 图表美化与视角调整
        ax.set_title(f"3D LLM Hidden State Trajectory (Sample {sample_idx})\n", fontsize=18, fontweight="bold")
        ax.set_xlabel(f"PC 1 ({explained_variance[0]:.1f}%)", fontsize=12, labelpad=10)
        ax.set_ylabel(f"PC 2 ({explained_variance[1]:.1f}%)", fontsize=12, labelpad=10)
        ax.set_zlabel(f"PC 3 ({explained_variance[2]:.1f}%)", fontsize=12, labelpad=10)

        # 设置一个较好的初始观察视角 (仰角 20 度，方位角 45 度)
        ax.view_init(elev=20, azim=45)

        # 将图例放在最佳位置，防止遮挡 3D 图形
        ax.legend(loc="upper left", fontsize=11, bbox_to_anchor=(1.05, 1))

        # 提取文件名信息并保存
        save_path = os.path.join(save_dir, f"sample_{sample_idx}_trajectory_3d.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.debug(f"Saved 3D PCA trajectory plot to: {save_path}")


def plot_pca_trajectory_2d(pt_file_path: str, save_dir: str = "./pca_outputs", explicit_steps=100):
    logger.debug(f"Loading data from: {pt_file_path}")
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"Cannot find file {pt_file_path}")

    data = torch.load(pt_file_path, map_location="cpu")

    hidden_states_batch = data.get("hidden_states", [])
    type_masks_batch = data.get("type_masks", [])

    os.makedirs(save_dir, exist_ok=True)
    batch_size = len(hidden_states_batch)

    batch_name = os.path.basename(pt_file_path).replace(".pt", "")
    batch_idx = int(batch_name.split("_")[1])

    for b in range(batch_size):
        sample_idx = batch_idx + b
        h_list = hidden_states_batch[b]
        mask_list = type_masks_batch[b]

        if not h_list or len(h_list) < 2:
            logger.debug(f"Sample {sample_idx} has insufficient hidden states, skipping...")
            continue

        h_list, mask_list, trans_idx = truncate_trace(h_list, mask_list, explicit_steps)

        # 1. 数据预处理
        try:
            h_tensor = torch.stack([h.squeeze().float() for h in h_list])  # [num_steps, hidden_dim]
            h_array = h_tensor.numpy()
        except Exception as e:
            logger.debug(f"Error stacking hidden states for sample {sample_idx}: {e}")
            continue

        # 2. PCA 降维 (2维)
        logger.debug(f"Sample {sample_idx}: Performing 2D PCA on shape {h_array.shape}...")
        pca = PCA(n_components=2)
        h_pca = pca.fit_transform(h_array)  # [num_steps, 2]

        explained_variance = pca.explained_variance_ratio_ * 100
        records = build_pca_records(
            sample_idx=sample_idx,
            source_file=pt_file_path,
            h_pca=h_pca,
            mask_list=mask_list,
            projection_dim=2,
            explicit_steps=explicit_steps,
            trans_idx=trans_idx,
            explained_variance=pca.explained_variance_ratio_,
        )
        save_pca_records(records, save_dir, sample_idx, projection_dim=2)

        # 3. 准备 2D 绘图
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111)

        latent_idx = [i for i, m in enumerate(mask_list) if m == 0]
        explicit_idx = [i for i, m in enumerate(mask_list) if m == 1]

        # ================== 绘制带颜色的连续曲线 ==================
        # 为了让线条不断开，Latent 的线条要把第一个 Explicit 点包含进去连起来
        if latent_idx and explicit_idx:
            latent_line_idx = latent_idx + [explicit_idx[0]]
        else:
            latent_line_idx = latent_idx

        if latent_line_idx:
            points = h_pca[latent_line_idx]
            n = len(points)
            cmap = plt.get_cmap("Blues")
            for i in range(n - 1):
                ax.plot(
                    points[i : i + 2, 0],
                    points[i : i + 2, 1],
                    color=cmap(0.3 + 0.7 * i / max(1, n - 1)),
                    alpha=0.8,
                    linestyle="-",
                    linewidth=2.5,
                    label="Latent Trajectory" if i == 0 else "",
                    zorder=1,
                )

        if explicit_idx:
            points = h_pca[explicit_idx]
            n = len(points)
            cmap = plt.get_cmap("Oranges")
            for i in range(n - 1):
                ax.plot(
                    points[i : i + 2, 0],
                    points[i : i + 2, 1],
                    color=cmap(0.3 + 0.7 * i / max(1, n - 1)),
                    alpha=0.8,
                    linestyle="-",
                    linewidth=2.5,
                    label="Explicit Trajectory" if i == 0 else "",
                    zorder=1,
                )
        # ===================================================================

        # ================== 按步长降采样绘制散点 ==================
        sample_rate_latent = max(1, len(latent_idx) // 10) if latent_idx else 1
        sample_rate_explicit = max(1, len(explicit_idx) // 10) if explicit_idx else 1

        sampled_latent = latent_idx[::sample_rate_latent] if latent_idx else []
        sampled_explicit = explicit_idx[::sample_rate_explicit] if explicit_idx else []

        if sampled_latent:
            cmap = plt.get_cmap("Blues")
            colors = [
                cmap(0.3 + 0.7 * i / max(1, len(latent_idx) - 1)) for i in range(0, len(latent_idx), sample_rate_latent)
            ]
            ax.scatter(
                h_pca[sampled_latent, 0],
                h_pca[sampled_latent, 1],
                c=colors,
                s=40,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )

        if sampled_explicit:
            cmap = plt.get_cmap("Oranges")
            colors = [
                cmap(0.3 + 0.7 * i / max(1, len(explicit_idx) - 1))
                for i in range(0, len(explicit_idx), sample_rate_explicit)
            ]
            ax.scatter(
                h_pca[sampled_explicit, 0],
                h_pca[sampled_explicit, 1],
                c=colors,
                s=40,
                edgecolors="white",
                linewidth=0.5,
                zorder=2,
            )
        # ===================================================================

        # 5. 标记关键节点 (起点、终点、切换点)
        ax.scatter(h_pca[0, 0], h_pca[0, 1], c="green", marker="*", s=400, label="Start (Prompt)", zorder=4)

        if explicit_steps != -1 and (trans_idx != -1 or (0 not in mask_list)):
            end_label = f"End (Truncated {explicit_steps} steps)"
        else:
            end_label = "End (EOS)"
        ax.scatter(h_pca[-1, 0], h_pca[-1, 1], c="red", marker="X", s=300, label=end_label, zorder=4)

        if trans_idx != -1 and trans_idx < len(h_pca):
            ax.scatter(
                h_pca[trans_idx, 0],
                h_pca[trans_idx, 1],
                c="purple",
                marker="D",
                s=200,
                label="Transition (Latent -> Explicit)",
                zorder=5,
            )

        # 6. 图表美化
        ax.set_title(f"2D LLM Hidden State Trajectory (Sample {sample_idx})\n", fontsize=18, fontweight="bold")
        ax.set_xlabel(f"PC 1 ({explained_variance[0]:.1f}%)", fontsize=12, labelpad=10)
        ax.set_ylabel(f"PC 2 ({explained_variance[1]:.1f}%)", fontsize=12, labelpad=10)
        ax.legend(loc="upper left", fontsize=11, bbox_to_anchor=(1.05, 1))
        ax.grid(True, linestyle="--", alpha=0.6)

        # 提取文件名信息并保存
        save_path = os.path.join(save_dir, f"sample_{sample_idx}_trajectory_2d.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.debug(f"Saved 2D PCA trajectory plot to: {save_path}")


def plot_pca_trajectory_6d(pt_file_path: str, save_dir: str = "./pca_outputs", explicit_steps=100):
    logger.debug(f"Loading data from: {pt_file_path}")
    if not os.path.exists(pt_file_path):
        raise FileNotFoundError(f"Cannot find file {pt_file_path}")

    data = torch.load(pt_file_path, map_location="cpu")

    hidden_states_batch = data.get("hidden_states", [])
    type_masks_batch = data.get("type_masks", [])

    os.makedirs(save_dir, exist_ok=True)
    batch_size = len(hidden_states_batch)

    batch_name = os.path.basename(pt_file_path).replace(".pt", "")
    batch_idx = int(batch_name.split("_")[1])

    for b in range(batch_size):
        sample_idx = batch_idx + b
        h_list = hidden_states_batch[b]
        mask_list = type_masks_batch[b]

        if not h_list or len(h_list) < 6:
            logger.debug(f"Sample {sample_idx} has insufficient hidden states, skipping...")
            continue

        h_list, mask_list, trans_idx = truncate_trace(h_list, mask_list, explicit_steps)

        # 1. 数据预处理
        try:
            h_tensor = torch.stack([h.squeeze().float() for h in h_list])  # [num_steps, hidden_dim]
            h_array = h_tensor.numpy()
        except Exception as e:
            logger.debug(f"Error stacking hidden states for sample {sample_idx}: {e}")
            continue

        if h_array.shape[0] < 6:
            logger.debug(f"Sample {sample_idx} has insufficient hidden states after truncation for 6D PCA.")
            continue

        # 2. PCA 降维 (6维)
        logger.debug(f"Sample {sample_idx}: Performing 6D PCA on shape {h_array.shape}...")
        pca = PCA(n_components=6)
        h_pca = pca.fit_transform(h_array)  # [num_steps, 6]

        explained_variance = pca.explained_variance_ratio_ * 100
        records = build_pca_records(
            sample_idx=sample_idx,
            source_file=pt_file_path,
            h_pca=h_pca,
            mask_list=mask_list,
            projection_dim=6,
            explicit_steps=explicit_steps,
            trans_idx=trans_idx,
            explained_variance=pca.explained_variance_ratio_,
        )
        save_pca_records(records, save_dir, sample_idx, projection_dim=6)

        # 3. 准备 6D 绘图 (3个子图)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

        latent_idx = [i for i, m in enumerate(mask_list) if m == 0]
        explicit_idx = [i for i, m in enumerate(mask_list) if m == 1]

        # ================== 绘制带颜色的连续曲线 ==================
        if latent_idx and explicit_idx:
            latent_line_idx = latent_idx + [explicit_idx[0]]
        else:
            latent_line_idx = latent_idx

        # ================== 按步长降采样绘制散点 ==================
        sample_rate_latent = max(1, len(latent_idx) // 10) if latent_idx else 1
        sample_rate_explicit = max(1, len(explicit_idx) // 10) if explicit_idx else 1

        sampled_latent = latent_idx[::sample_rate_latent] if latent_idx else []
        sampled_explicit = explicit_idx[::sample_rate_explicit] if explicit_idx else []

        pairs = [(0, 1), (2, 3), (4, 5)]

        for i, (d1, d2) in enumerate(pairs):
            ax = axes[i]

            # 绘制 Latent 曲线
            if latent_line_idx:
                points = h_pca[latent_line_idx]
                n = len(points)
                cmap = plt.get_cmap("Blues")
                for k in range(n - 1):
                    ax.plot(
                        points[k : k + 2, d1],
                        points[k : k + 2, d2],
                        color=cmap(0.3 + 0.7 * k / max(1, n - 1)),
                        alpha=0.8,
                        linestyle="-",
                        linewidth=2.5,
                        label="Latent Trajectory" if k == 0 and i == 0 else "",
                        zorder=1,
                    )

            # 绘制 Explicit 曲线
            if explicit_idx:
                points = h_pca[explicit_idx]
                n = len(points)
                cmap = plt.get_cmap("Oranges")
                for k in range(n - 1):
                    ax.plot(
                        points[k : k + 2, d1],
                        points[k : k + 2, d2],
                        color=cmap(0.3 + 0.7 * k / max(1, n - 1)),
                        alpha=0.8,
                        linestyle="-",
                        linewidth=2.5,
                        label="Explicit Trajectory" if k == 0 and i == 0 else "",
                        zorder=1,
                    )

            # 绘制散点
            if sampled_latent:
                cmap = plt.get_cmap("Blues")
                colors = [
                    cmap(0.3 + 0.7 * k / max(1, len(latent_idx) - 1))
                    for k in range(0, len(latent_idx), sample_rate_latent)
                ]
                ax.scatter(
                    h_pca[sampled_latent, d1],
                    h_pca[sampled_latent, d2],
                    c=colors,
                    s=40,
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=2,
                )

            if sampled_explicit:
                cmap = plt.get_cmap("Oranges")
                colors = [
                    cmap(0.3 + 0.7 * k / max(1, len(explicit_idx) - 1))
                    for k in range(0, len(explicit_idx), sample_rate_explicit)
                ]
                ax.scatter(
                    h_pca[sampled_explicit, d1],
                    h_pca[sampled_explicit, d2],
                    c=colors,
                    s=40,
                    edgecolors="white",
                    linewidth=0.5,
                    zorder=2,
                )

            # 5. 标记关键节点
            ax.scatter(
                h_pca[0, d1],
                h_pca[0, d2],
                c="green",
                marker="*",
                s=400,
                label="Start (Prompt)" if i == 0 else "",
                zorder=4,
            )

            if explicit_steps != -1 and (trans_idx != -1 or (0 not in mask_list)):
                end_label = f"End (Truncated {explicit_steps} steps)"
            else:
                end_label = "End (EOS)"
            ax.scatter(
                h_pca[-1, d1], h_pca[-1, d2], c="red", marker="X", s=300, label=end_label if i == 0 else "", zorder=4
            )

            if trans_idx != -1 and trans_idx < len(h_pca):
                ax.scatter(
                    h_pca[trans_idx, d1],
                    h_pca[trans_idx, d2],
                    c="purple",
                    marker="D",
                    s=200,
                    label="Transition (Latent -> Explicit)" if i == 0 else "",
                    zorder=5,
                )

            # 6. 图表美化
            ax.set_xlabel(f"PC {d1 + 1} ({explained_variance[d1]:.1f}%)", fontsize=12)
            ax.set_ylabel(f"PC {d2 + 1} ({explained_variance[d2]:.1f}%)", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.6)
            if i == 0:
                ax.legend(loc="upper left", fontsize=10)

        fig.suptitle(f"6D LLM Hidden State Trajectory (Sample {sample_idx})", fontsize=20, fontweight="bold")

        save_path = os.path.join(save_dir, f"sample_{sample_idx}_trajectory_6d.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.debug(f"Saved 6D PCA trajectory plot to: {save_path}")


def plot_dir(task, method, time_stamp, data_dir, explicit_steps=100, plot_nums=-1):
    save_dir_3d = f"reasoning_trace/3d/{task}/{method}/{time_stamp}"
    save_dir_2d = f"reasoning_trace/2d/{task}/{method}/{time_stamp}"
    save_dir_6d = f"reasoning_trace/6d/{task}/{method}/{time_stamp}"
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        return

    try:
        file_list = sorted((f for f in os.listdir(data_dir) if f.endswith(".pt")), key=lambda f: int(f.split("_")[1]))
    except (ValueError, IndexError):
        logger.warning("Could not sort files numerically, falling back to alphabetical sort.")
        file_list = sorted(f for f in os.listdir(data_dir) if f.endswith(".pt"))

    if plot_nums != -1:
        file_list = file_list[:plot_nums]

    for file_name in file_list:
        target_file = os.path.join(data_dir, file_name)
        logger.info(f"Processing file: {target_file}")
        plot_pca_trajectory_3d(target_file, save_dir=save_dir_3d, explicit_steps=explicit_steps)
        plot_pca_trajectory_2d(target_file, save_dir=save_dir_2d, explicit_steps=explicit_steps)
        plot_pca_trajectory_6d(target_file, save_dir=save_dir_6d, explicit_steps=explicit_steps)
