import cv2, os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import seaborn as sns
import torch
from typing import List, Dict
from loguru import logger


class EntropyVisualizer:
    @staticmethod
    def get_color(entropy, min_ent, max_ent):
        """Map entropy values to colors (BGR). Use blue->red gradient here"""

        if max_ent == min_ent:
            norm_entropy = 0.0
        else:
            norm_entropy = (entropy - min_ent) / (max_ent - min_ent)

        r = int(255 * norm_entropy)
        g = int(100 * (1 - abs(norm_entropy - 0.5) * 2))
        b = int(255 * (1 - norm_entropy))
        return (b, g, r)

    @classmethod
    def draw_text_with_entropy(cls, tokens, entropies, save_path, solution="None", font_path="arial.ttf"):

        min_ent = min(entropies) if entropies else 0
        max_ent = max(entropies) if entropies else 1

        font_size = 24
        line_height = 40
        margin = 50
        img_width = 1000

        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()

        lines = []
        current_line = []
        curr_x = margin
        for token, ent in zip(tokens, entropies):
            t_width = font.getlength(token)
            if curr_x + t_width > img_width - margin:
                lines.append(current_line)
                current_line = []
                curr_x = margin
            current_line.append((token, ent, curr_x))
            curr_x += t_width
        lines.append(current_line)

        img_height = len(lines) * line_height + margin * 2 + 100
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            y = margin + i * line_height
            for token, ent, x in line:
                color = cls.get_color(ent, min_ent, max_ent)
                draw.text((x, y), token, font=font, fill=(color[2], color[1], color[0]))
        cb_x, cb_y, cb_w, cb_h = margin, img_height - 60, 300, 20
        draw.text(
            (cb_x, cb_y - 25),
            f"Entropy: {min_ent:.2f} (Blue) to {max_ent:.2f} (Red), solution: {solution}",
            font=font,
            fill=(0, 0, 0),
        )
        for i in range(cb_w):
            current_val = min_ent + (i / cb_w) * (max_ent - min_ent)
            c = cls.get_color(current_val, min_ent, max_ent)
            draw.line([(cb_x + i, cb_y), (cb_x + i, cb_y + cb_h)], fill=(c[2], c[1], c[0]))

        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cv2_img)

    @classmethod
    def draw_text_with_entropy_from_ids(
        cls, tokenizer, token_ids, entropies, save_path, solution="None", font_path="arial.ttf"
    ):
        """
        输入为 token_ids (List[int] 或 Tensor) 和对应的 entropies。
        需要传入 tokenizer 进行解码。
        """
        # 1. 预处理：确保输入转为标准 Python List
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if hasattr(entropies, "tolist"):
            entropies = entropies.tolist()

        # 2. 核心转换：将 ID 解码为 Token 字符串
        # 注意：逐个 decode 是为了保持 token 和 entropy 列表的一一对应和对齐
        # replace('\n', ' ') 是为了防止换行符破坏我们的自定义排版逻辑
        tokens = [tokenizer.decode([tid], skip_special_tokens=False).replace("\n", " ") for tid in token_ids]

        # --- 以下为原绘图逻辑 ---

        min_ent = min(entropies) if entropies else 0
        max_ent = max(entropies) if entropies else 1

        font_size = 24
        line_height = 40
        margin = 50
        img_width = 1000

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            # 如果找不到 arial.ttf，回退到默认字体，但默认字体可能不支持 getlength
            # 为了保险，尝试加载一个系统默认的等宽字体或由 PIL 自动处理
            font = ImageFont.load_default()

        lines = []
        current_line = []
        curr_x = margin

        for token, ent in zip(tokens, entropies):
            # 获取当前 token 渲染宽度
            try:
                t_width = font.getlength(token)
            except AttributeError:
                # 兼容旧版 Pillow
                t_width = font.getsize(token)[0]  # type: ignore

            # 自动换行判断
            if curr_x + t_width > img_width - margin:
                lines.append(current_line)
                current_line = []
                curr_x = margin

            current_line.append((token, ent, curr_x))
            curr_x += t_width

        # 添加最后一行
        if current_line:
            lines.append(current_line)

        # 计算画布总高度
        img_height = len(lines) * line_height + margin * 2 + 100
        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        # 绘制每一行
        for i, line in enumerate(lines):
            y = margin + i * line_height
            for token, ent, x in line:
                # 假设 get_color 已经在类中定义
                color = cls.get_color(ent, min_ent, max_ent)
                # PIL 使用 RGB, cv2 使用 BGR，这里保持与原逻辑一致 (fill用RGB)
                draw.text((x, y), token, font=font, fill=(color[2], color[1], color[0]))

        # 绘制底部的 Color Bar 和说明文字
        cb_x, cb_y, cb_w, cb_h = margin, img_height - 60, 300, 20
        draw.text(
            (cb_x, cb_y - 25),
            f"Entropy: {min_ent:.2f} (Blue) to {max_ent:.2f} (Red), solution: {solution}",
            font=font,
            fill=(0, 0, 0),
        )

        # 绘制渐变条
        for i in range(cb_w):
            current_val = min_ent + (i / cb_w) * (max_ent - min_ent)
            c = cls.get_color(current_val, min_ent, max_ent)
            draw.line([(cb_x + i, cb_y), (cb_x + i, cb_y + cb_h)], fill=(c[2], c[1], c[0]))

        # 转换并保存
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cv2_img)

    @classmethod
    def get_color_gradient(cls, value, min_v, max_v, start_color, end_color):
        if max_v == min_v:
            ratio = 0.5
        else:
            ratio = (value - min_v) / (max_v - min_v)

        # 限制 ratio 在 0-1 之间
        ratio = max(0.0, min(1.0, ratio))

        r = int(start_color[0] + ratio * (end_color[0] - start_color[0]))
        g = int(start_color[1] + ratio * (end_color[1] - start_color[1]))
        b = int(start_color[2] + ratio * (end_color[2] - start_color[2]))
        return (r, g, b)

    @classmethod
    def draw_text_with_entropy_from_ids_with_latent(
        cls,
        tokenizer,
        token_ids,
        entropies,
        save_path,
        token_types=None,
        solution="None",
        font_path="arial.ttf",
    ):
        # 1. 数据预处理
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if hasattr(entropies, "tolist"):
            entropies = entropies.tolist()
        if token_types is not None and hasattr(token_types, "tolist"):
            token_types = token_types.tolist()

        min_len = len(token_ids)
        if len(entropies) < min_len:
            min_len = len(entropies)
        if token_types is not None and len(token_types) < min_len:
            min_len = len(token_types)

        token_ids = token_ids[:min_len]
        entropies = entropies[:min_len]
        if token_types is not None:
            token_types = token_types[:min_len]

        # 2. 解码 Token
        raw_tokens = [tokenizer.decode([tid], skip_special_tokens=False) for tid in token_ids]
        # 统一处理换行符，避免排版混乱，但 Type 1 的空格保留
        tokens = [t.replace("\n", " ").replace("\r", "") for t in raw_tokens]

        # 3. 分别计算 entropy 范围
        min_ent_global = min(entropies) if entropies else 0
        max_ent_global = max(entropies) if entropies else 1

        min_ent_0, max_ent_0 = 0, 1
        min_ent_1, max_ent_1 = 0, 1

        if token_types is not None:
            ents_0 = [e for t, e in zip(token_types, entropies) if t == 0]
            if ents_0:
                min_ent_0, max_ent_0 = min(ents_0), max(ents_0)

            ents_1 = [e for t, e in zip(token_types, entropies) if t != 0]
            if ents_1:
                min_ent_1, max_ent_1 = min(ents_1), max(ents_1)

        # 4. 绘图配置
        font_size = 24
        line_height = 40
        margin = 50
        img_width = 1200

        try:
            font = ImageFont.truetype(font_path, font_size)
        except Exception:
            font = ImageFont.load_default()

        # [修改点] 定义颜色范围：熵小(浅) -> 熵大(深)

        # 全局:
        COLOR_GLOBAL_LOW = (0, 0, 255)  # rgb(0, 0, 255)
        COLOR_GLOBAL_HIGH = (255, 0, 0)  # rgb(255, 0, 0)

        # Type 0 (Latent):
        COLOR_TYPE0_LOW = (0, 255, 255)  # rgb(0, 255, 255)
        COLOR_TYPE0_HIGH = (0, 100, 0)  # rgb(0, 100, 0)

        # Type 1 (Explicit):
        COLOR_TYPE1_LOW = (255, 200, 200)  # rgb(255, 200, 200)
        COLOR_TYPE1_HIGH = (139, 0, 0)  # rgb(139, 0, 0)

        lines = []
        current_line = []
        curr_x = margin

        special_symbol = "֋"

        # 5. 排版计算 (Layout)
        for i, (token, ent) in enumerate(zip(tokens, entropies)):
            t_type = token_types[i] if token_types is not None else -1

            # 判断是否为空内容
            is_empty_content = (not token) or (token.strip() == "")

            # 确定绘制的文本
            draw_token = token

            # [修改逻辑]
            # Type 0: 如果是空内容，替换为特殊符号 ֋
            if t_type == 0 and is_empty_content:
                draw_token = special_symbol

            # Type 1: 保持原样 (即使是空格也原样绘制，PIL会留空)

            # 计算宽度
            try:
                t_width = font.getlength(draw_token)
            except AttributeError:
                t_width = font.getsize(draw_token)[0]  # type: ignore

            # 换行判断
            if curr_x + t_width > img_width - margin:
                lines.append(current_line)
                current_line = []
                curr_x = margin

            current_line.append(
                {
                    "token": draw_token,
                    "ent": ent,
                    "type": t_type,
                    "x": curr_x,
                }
            )
            curr_x += t_width

        if current_line:
            lines.append(current_line)

        # 6. 绘制画布
        legend_height = 100 if token_types is None else 140
        img_height = len(lines) * line_height + margin * 2 + legend_height

        img = Image.new("RGB", (img_width, img_height), (255, 255, 255))  # 背景纯白
        draw = ImageDraw.Draw(img)

        for i, line in enumerate(lines):
            y = margin + i * line_height

            for item in line:
                val = item["ent"]
                t_type = item["type"]

                # 获取颜色
                if token_types is None:
                    color = cls.get_color_gradient(
                        val, min_ent_global, max_ent_global, COLOR_GLOBAL_LOW, COLOR_GLOBAL_HIGH
                    )
                else:
                    if t_type == 0:
                        color = cls.get_color_gradient(val, min_ent_0, max_ent_0, COLOR_TYPE0_LOW, COLOR_TYPE0_HIGH)
                    else:
                        color = cls.get_color_gradient(val, min_ent_1, max_ent_1, COLOR_TYPE1_LOW, COLOR_TYPE1_HIGH)

                # 直接绘制文字 (去掉了所有 rect/offset 逻辑)
                draw.text((item["x"], y), item["token"], font=font, fill=color)

        # 7. 绘制图例 (Legend)
        cb_x = margin
        cb_y = img_height - legend_height + 20
        cb_w = 300
        cb_h = 15

        def draw_gradient_bar(label, start_c, end_c, offset_y, v_min, v_max):
            draw.text(
                (cb_x, cb_y + offset_y - 25),
                f"{label}: Low Ent ({v_min:.2f}) -> High Ent ({v_max:.2f})",
                font=font,
                fill=(0, 0, 0),
            )
            for k in range(cb_w):
                ratio = k / cb_w
                r = int(start_c[0] + ratio * (end_c[0] - start_c[0]))
                g = int(start_c[1] + ratio * (end_c[1] - start_c[1]))
                b = int(start_c[2] + ratio * (end_c[2] - start_c[2]))
                draw.line([(cb_x + k, cb_y + offset_y), (cb_x + k, cb_y + offset_y + cb_h)], fill=(r, g, b))

        if token_types is None:
            draw_gradient_bar("Entropy (All)", COLOR_GLOBAL_LOW, COLOR_GLOBAL_HIGH, 0, min_ent_global, max_ent_global)
        else:
            draw_gradient_bar("Latent (0)", COLOR_TYPE0_LOW, COLOR_TYPE0_HIGH, 0, min_ent_0, max_ent_0)
            draw_gradient_bar("Explicit (1)", COLOR_TYPE1_LOW, COLOR_TYPE1_HIGH, 50, min_ent_1, max_ent_1)

        if solution and solution != "None":
            sol_text = f"Solution: {solution}"
            if len(sol_text) > 100:
                sol_text = sol_text[:100] + "..."
            draw.text((cb_x + cb_w + 50, cb_y), sol_text, font=font, fill=(0, 0, 0))

        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, cv2_img)

    @classmethod
    def save_experiment_data(
        cls, batch_start: int, experiment_data: Dict, experiment_data_dir: str, SAVE_STATES=True, DRAW_ATTENTION=True
    ):
        """
        保存实验数据并触发可视化。
        修改点：移除了提前的 Padding 逻辑，直接将原始的变长 Attention List 传给可视化函数。
        """
        os.makedirs(experiment_data_dir, exist_ok=True)

        if SAVE_STATES:
            save_path = f"{experiment_data_dir}/batch_{batch_start}_data.pt"
            torch.save(experiment_data, save_path)
            logger.info(f"Saved detailed experiment data to {save_path}")

        if DRAW_ATTENTION:
            batch_size = len(experiment_data["texts"])
            all_attentions = experiment_data["attentions"]  # 形状: batch_size 个 List[Tensor(num_layers, k_len)]
            type_masks = experiment_data["type_masks"]  # 形状: batch_size 个 List[int]

            # 2. 遍历 Batch 逐个绘制注意力转移图
            for b in range(batch_size):
                sample_attns = all_attentions[b]
                sample_mask = type_masks[b]

                if sample_attns and len(sample_attns) > 0:
                    vis_path = f"{experiment_data_dir}/sample_{batch_start + b}_attention.png"
                    try:
                        # 直接传入 List，由可视化函数内部处理变长序列的对齐
                        cls.visualize_attention_transition(
                            attention_list=sample_attns, type_mask=sample_mask, save_path=vis_path
                        )
                        logger.info(f"Successfully generated attention visualization for sample {batch_start + b}")
                    except Exception as e:
                        logger.error(f"Failed to visualize sample {batch_start + b}: {e}")

    @classmethod
    def visualize_attention_transition(cls, attention_list: List[torch.Tensor], type_mask: List[int], save_path: str):
        """
        生成符合科研论文规范的注意力转移图
        特色: 动态重构 X 轴，精准展现 Latent 内部、Explicit 对 Prompt/KVCache、以及 Explicit 内部的注意力分布。
        """

        num_layers = attention_list[0].shape[0]
        total_steps = len(attention_list)
        max_vis_steps = min(500, total_steps)

        # =================================================================
        # 1. 自动侦测模型的 Phase 切换边界，并处理 KV Cache 的物理重置状态
        # =================================================================
        phases = []
        current_phase_start = 0
        last_type = type_mask[0]

        for i in range(1, max_vis_steps):
            curr_k_len = attention_list[i].shape[1]
            prev_k_len = attention_list[i - 1].shape[1]

            # 切换条件: Type改变，或者 Key长度发生非自然跳变（注入新Prompt或重置）
            if type_mask[i] != last_type or curr_k_len != prev_k_len + 1:
                phases.append((current_phase_start, i))
                current_phase_start = i
                last_type = type_mask[i]
        phases.append((current_phase_start, max_vis_steps))

        # =================================================================
        # 2. 计算精准的物理坐标映射 (避免任何错位)
        # =================================================================
        phase_boxes = []
        current_x = 0

        for phase_idx, (start, end) in enumerate(phases):
            first_k = attention_list[start].shape[1]
            p_type = type_mask[start]

            if phase_idx == 0:
                # 初始 Latent 阶段：屏蔽其初始的超长系统 Prompt
                prompt_len = 0
                cut_left = first_k - 1
                kv_reset = False
            else:
                prev_k = attention_list[start - 1].shape[1]
                # 严格判断是否发生了 KV Cache 清空
                if first_k <= start + 1 or first_k < prev_k:
                    # 发生了重置 (is_expicit_cot逻辑): 新 k_len 仅包含 prompt_len + 1
                    prompt_len = first_k - 1
                    cut_left = 0
                    kv_reset = True
                else:
                    # 未重置，继承了历史 (is_expicit_next逻辑)
                    prompt_len = first_k - prev_k - 1
                    cut_left = attention_list[0].shape[1] - 1
                    kv_reset = False

            prompt_x_start = current_x
            current_x += prompt_len
            gen_x_start = current_x

            # 为当前阶段生成的 Token 在 X 轴预留空间
            for i in range(start, end):
                current_x += 1

            phase_boxes.append(
                {
                    "y_start": start,
                    "y_end": end,
                    "x_prompt_start": prompt_x_start,
                    "x_gen_start": gen_x_start,
                    "x_end": current_x,
                    "prompt_len": prompt_len,
                    "type": p_type,
                    "kv_reset": kv_reset,
                    "cut_left": cut_left,
                }
            )

        total_x_steps = current_x
        total_y_steps = max_vis_steps

        # =================================================================
        # 3. 核心拼接算法：构建 非对称 纯净矩阵
        # =================================================================
        clean_matrix = torch.zeros((num_layers, total_y_steps, total_x_steps), dtype=torch.float32)

        for phase in phase_boxes:
            start, end, cut_left = phase["y_start"], phase["y_end"], phase["cut_left"]

            # 决定注意力投射的 X 轴起点：如果继承了 KV Cache，就能回看历史阶段；否则从新 Prompt 开始
            if start == 0 or not phase["kv_reset"]:
                place_x_start = 0
            else:
                place_x_start = phase["x_prompt_start"]

            for i in range(start, end):
                attn = attention_list[i]
                # 切除不需要的初始系统 Prompt
                sliced_attn = attn[:, cut_left:].float()

                l_slice = sliced_attn.shape[1]
                end_x = place_x_start + l_slice

                # 边界保护
                if end_x > total_x_steps:
                    l_slice = total_x_steps - place_x_start
                    sliced_attn = sliced_attn[:, :l_slice]
                    end_x = total_x_steps

                clean_matrix[:, i, place_x_start:end_x] = sliced_attn

        # =================================================================
        # 4. 绘图逻辑：添加三种不同颜色的学术边框
        # =================================================================
        fig, axes = plt.subplots(1, num_layers, figsize=(6 * num_layers, 6))
        if num_layers == 1:
            axes = [axes]

        cmap = "magma"
        layer_titles = ["Layer 0 (Bottom)", "Layer Q1", "Layer Mid", "Layer Q3", "Layer Top"]

        for i in range(num_layers):
            ax = axes[i]
            sns.heatmap(
                clean_matrix[i].numpy(),
                cmap=cmap,
                cbar=(i == num_layers - 1),
                ax=ax,
                xticklabels=False,
                yticklabels=False,
                vmin=0.0,
                vmax=torch.max(clean_matrix[i]).item() * 0.8,
            )

            # 在热力图上绘制三区标注框
            for phase in phase_boxes:
                y_s = phase["y_start"]
                h = phase["y_end"] - y_s

                # 情况1: Latent Phase
                if phase["type"] == 0:
                    x_s = phase["x_gen_start"]
                    w = phase["x_end"] - x_s
                    if w > 0:
                        ax.add_patch(
                            patches.Rectangle(
                                (x_s, y_s), w, h, linewidth=2.5, edgecolor="cyan", facecolor="none", linestyle="--"
                            )
                        )

                # 情况2 & 3: Explicit Phase
                else:
                    # --- Box 2: Explicit 模型回看 Prompt / KV Cache 的区域 ---
                    # 动态判断：如果 KV Cache 继承了，橘色框会从 X=0 框起，涵盖 Latent 历史 + 新 Prompt！
                    attn_start_x = 0 if not phase["kv_reset"] else phase["x_prompt_start"]
                    prompt_area_w = phase["x_gen_start"] - attn_start_x
                    if prompt_area_w > 0:
                        ax.add_patch(
                            patches.Rectangle(
                                (attn_start_x, y_s),
                                prompt_area_w,
                                h,
                                linewidth=2.5,
                                edgecolor="orange",
                                facecolor="none",
                                linestyle="--",
                            )
                        )

                    # --- Box 3: Explicit 新生成的 Token 相互之间的注意力区域 ---
                    gen_x_s = phase["x_gen_start"]
                    gen_w = phase["x_end"] - gen_x_s
                    if gen_w > 0:
                        ax.add_patch(
                            patches.Rectangle(
                                (gen_x_s, y_s),
                                gen_w,
                                h,
                                linewidth=2.5,
                                edgecolor="lime",
                                facecolor="none",
                                linestyle="--",
                            )
                        )

            ax.set_title(layer_titles[i] if i < len(layer_titles) else f"Layer {i}", fontsize=14, pad=10)
            ax.set_xlabel("Attended Tokens (Keys = Prompt + Generated)", fontsize=12)
            if i == 0:
                ax.set_ylabel("Generation Steps (Queries)", fontsize=12)

        # 构建匹配三种框颜色的图例
        custom_lines = [
            Line2D([0], [0], color="cyan", lw=2.5, linestyle="--"),
            Line2D([0], [0], color="orange", lw=2.5, linestyle="--"),
            Line2D([0], [0], color="lime", lw=2.5, linestyle="--"),
        ]
        fig.legend(
            custom_lines,
            [
                "1. Latent Reasoning (Gen-to-Gen)",
                "2. Explicit Attn to KV Cache & Prompt",
                "3. Explicit Reasoning (Gen-to-Gen)",
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=3,
            frameon=False,
            fontsize=13,
        )

        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # logger.debug(f"Saved attention transition visualization to {save_path}")
        # plt.savefig(save_path.replace(".png", ".pdf"), bbox_inches="tight")
        plt.close()
