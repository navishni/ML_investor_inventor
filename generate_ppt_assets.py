from __future__ import annotations

import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from recommender import PlatformRecommender


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "ppt_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

BG = (8, 19, 29)
PANEL = (16, 36, 53)
PANEL_2 = (20, 42, 62)
LINE = (39, 65, 90)
TEXT = (238, 247, 255)
MUTED = (157, 181, 201)
PRIMARY = (55, 213, 255)
SECONDARY = (255, 177, 74)
SUCCESS = (131, 217, 120)
DANGER = (255, 112, 133)
ACCENT = (43, 141, 255)


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def new_canvas(width: int = 1600, height: int = 900, color: tuple[int, int, int] = BG):
    image = Image.new("RGB", (width, height), color)
    return image, ImageDraw.Draw(image)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_text(draw: ImageDraw.ImageDraw, xy, text: str, font, fill, spacing: int = 6):
    draw.multiline_text(xy, text, font=font, fill=fill, spacing=spacing)


def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, font, fill, xy, max_width: int, spacing: int = 6):
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    draw.multiline_text(xy, "\n".join(lines), font=font, fill=fill, spacing=spacing)
    line_h = text_size(draw, "Ag", font)[1] + spacing
    return line_h * max(1, len(lines))


def pill(draw: ImageDraw.ImageDraw, xy, text: str, fill, text_fill=(255, 255, 255)):
    x, y, w, h = xy
    draw.rounded_rectangle((x, y, x + w, y + h), radius=16, fill=fill)
    font = load_font(22, bold=True)
    tw, th = text_size(draw, text, font)
    draw.text((x + (w - tw) / 2, y + (h - th) / 2 - 2), text, font=font, fill=text_fill)


def save(img: Image.Image, filename: str) -> Path:
    path = ASSET_DIR / filename
    img.save(path)
    return path


def draw_performance_table(comparison):
    img, draw = new_canvas()
    title_font = load_font(44, bold=True)
    sub_font = load_font(24, bold=False)
    head_font = load_font(20, bold=True)
    body_font = load_font(20, bold=False)
    small_font = load_font(18, bold=False)

    draw_text(draw, (70, 40), "MODEL COMPARISON", head_font, PRIMARY)
    draw_text(draw, (70, 88), "Performance table", title_font, TEXT)
    draw_text(
        draw,
        (70, 142),
        "The best model is chosen using a balanced decision score built from all six metrics.",
        sub_font,
        MUTED,
    )

    left = 60
    top = 210
    width = 1480
    header_h = 56
    row_h = 122
    cols = [
        ("Model", 250),
        ("Accuracy", 120),
        ("Precision", 120),
        ("Recall", 120),
        ("F1", 110),
        ("ROC AUC", 120),
        ("PR AUC", 120),
        ("Decision", 120),
    ]
    table_w = sum(w for _, w in cols)
    x = left
    draw.rounded_rectangle((left, top, left + table_w, top + header_h + row_h * len(comparison)), radius=18, fill=PANEL, outline=LINE, width=2)
    for name, w in cols:
        draw.text((x + 16, top + 16), name, font=head_font, fill=MUTED)
        x += w
    draw.line((left, top + header_h, left + table_w, top + header_h), fill=LINE, width=2)

    for index, row in enumerate(comparison):
        y = top + header_h + index * row_h
        is_best = index == 0
        fill = PANEL_2 if is_best else (13, 27, 40)
        draw.rectangle((left, y, left + table_w, y + row_h), fill=fill, outline=(35, 61, 85))
        if is_best:
            pill(draw, (left + 16, y + 16, 72, 34), "Best", PRIMARY, (4, 14, 22))
        x = left
        values = [
            row["name"],
            f'{row["accuracy"] * 100:.2f}%',
            f'{row["precision"] * 100:.2f}%',
            f'{row["recall"] * 100:.2f}%',
            f'{row["f1"] * 100:.2f}%',
            f'{row["roc_auc"] * 100:.2f}%',
            f'{row["pr_auc"] * 100:.2f}%',
            f'{row["decision_score"] * 100:.2f}%',
        ]
        for col_index, ((_, w), value) in enumerate(zip(cols, values)):
            font = body_font if col_index == 0 else head_font
            fill_text = TEXT if col_index == 0 else TEXT
            draw.text((x + 16, y + 44), value, font=font, fill=fill_text)
            x += w
        if is_best:
            draw.line((left, y + row_h - 2, left + table_w, y + row_h - 2), fill=PRIMARY, width=3)

    note = "Decision score = average of Accuracy, Precision, Recall, F1, ROC AUC, and PR AUC."
    draw_text(draw, (70, 780), note, small_font, MUTED)
    return save(img, "performance_table.png")


def draw_split_diagram():
    img, draw = new_canvas()
    title_font = load_font(44, bold=True)
    sub_font = load_font(24, bold=False)
    box_font = load_font(24, bold=True)
    small_font = load_font(18, bold=False)

    draw_text(draw, (70, 40), "TRAIN / TEST SPLIT", load_font(22, bold=True), PRIMARY)
    draw_text(draw, (70, 88), "How the dataset was divided", title_font, TEXT)
    draw_text(draw, (70, 144), "The model learns from the training set and is evaluated only on unseen test data.", sub_font, MUTED)

    def box(x, y, w, h, title, desc, fill):
        draw.rounded_rectangle((x, y, x + w, y + h), radius=22, fill=fill, outline=LINE, width=2)
        tw, th = text_size(draw, title, box_font)
        draw.text((x + (w - tw) / 2, y + 30), title, font=box_font, fill=TEXT)
        draw_text(draw, (x + 26, y + 92), desc, small_font, MUTED)

    box(140, 280, 1320, 110, "Full Dataset", "All investor-inventor interaction rows after cleaning and feature engineering.", PANEL)
    box(210, 470, 500, 150, "Training Set 75%", "Used to fit the model and learn matching patterns.", PANEL_2)
    box(890, 470, 500, 150, "Test Set 25%", "Kept unseen until evaluation so the score is fair.", PANEL_2)
    draw.line((800, 390, 800, 470), fill=PRIMARY, width=5)
    draw.polygon([(790, 458), (810, 458), (800, 476)], fill=PRIMARY)
    draw.line((500, 620, 500, 690), fill=PRIMARY, width=5)
    draw.line((1180, 620, 1180, 690), fill=PRIMARY, width=5)
    draw_text(draw, (420, 690), "Model Training", load_font(24, bold=True), TEXT)
    draw_text(draw, (1080, 690), "Model Evaluation", load_font(24, bold=True), TEXT)
    draw_text(draw, (70, 790), "Same split is used for every model to keep the comparison fair.", small_font, MUTED)
    return save(img, "train_test_split.png")


def draw_confusion_matrices(comparison):
    img, draw = new_canvas()
    title_font = load_font(44, bold=True)
    head_font = load_font(24, bold=True)
    small_font = load_font(18, bold=False)

    draw_text(draw, (70, 40), "CONFUSION MATRICES", load_font(22, bold=True), PRIMARY)
    draw_text(draw, (70, 88), "Where the model was correct and where it made mistakes", title_font, TEXT)
    draw_text(draw, (70, 144), "High TP and TN are good. Low FP and FN mean fewer wrong recommendations.", small_font, MUTED)

    card_w = 700
    card_h = 250
    positions = [(70, 220), (830, 220), (70, 500), (830, 500)]
    cell_labels = [("TN", "FP"), ("FN", "TP")]
    cell_colors = {
        "TN": (29, 78, 59),
        "TP": (31, 94, 88),
        "FP": (116, 46, 62),
        "FN": (132, 84, 42),
    }
    for (x, y), row in zip(positions, comparison[:4]):
        draw.rounded_rectangle((x, y, x + card_w, y + card_h), radius=20, fill=PANEL, outline=LINE, width=2)
        draw.text((x + 20, y + 18), row["name"], font=head_font, fill=TEXT)
        counts = {
            "TN": row["confusion"]["tn"],
            "FP": row["confusion"]["fp"],
            "FN": row["confusion"]["fn"],
            "TP": row["confusion"]["tp"],
        }
        start_x = x + 28
        start_y = y + 72
        cell_w = 300
        cell_h = 70
        for r in range(2):
            for c in range(2):
                label = cell_labels[r][c]
                cx = start_x + c * (cell_w + 24)
                cy = start_y + r * (cell_h + 18)
                draw.rounded_rectangle((cx, cy, cx + cell_w, cy + cell_h), radius=16, fill=cell_colors[label], outline=(255, 255, 255), width=1)
                draw.text((cx + 16, cy + 10), label, font=load_font(18, bold=True), fill=TEXT)
                count_text = str(counts[label])
                tw, th = text_size(draw, count_text, load_font(26, bold=True))
                draw.text((cx + cell_w - tw - 18, cy + 26), count_text, font=load_font(26, bold=True), fill=TEXT)
    draw_text(draw, (70, 790), "Interpretation: the best model has many TP/TN and fewer FP/FN.", small_font, MUTED)
    return save(img, "confusion_matrices.png")


def _scale_points(xs, ys, plot):
    x0, y0, w, h = plot
    pts = []
    for x, y in zip(xs, ys):
        sx = x0 + x * w
        sy = y0 + (1 - y) * h
        pts.append((sx, sy))
    return pts


def _draw_dashed_line(draw, start, end, color, width=3, dash=10, gap=8):
    x1, y1 = start
    x2, y2 = end
    total = math.hypot(x2 - x1, y2 - y1)
    if total == 0:
        return
    dx = (x2 - x1) / total
    dy = (y2 - y1) / total
    traveled = 0.0
    while traveled < total:
        s = traveled
        e = min(traveled + dash, total)
        sx = x1 + dx * s
        sy = y1 + dy * s
        ex = x1 + dx * e
        ey = y1 + dy * e
        draw.line((sx, sy, ex, ey), fill=color, width=width)
        traveled += dash + gap


def draw_curve_chart(graph, series_key: str, filename: str, title: str, subtitle: str, x_label: str, y_label: str):
    img, draw = new_canvas()
    title_font = load_font(34, bold=True)
    sub_font = load_font(20, bold=False)
    axis_font = load_font(18, bold=True)
    tick_font = load_font(16, bold=False)
    legend_font = load_font(18, bold=True)
    small_font = load_font(16, bold=False)

    draw_text(draw, (70, 40), title.upper(), load_font(22, bold=True), PRIMARY)
    draw_text(draw, (70, 88), subtitle, title_font, TEXT)

    plot_left = 120
    plot_top = 260
    plot_w = 1300
    plot_h = 500
    draw.rounded_rectangle((90, 220, 1510, 820), radius=24, fill=PANEL, outline=LINE, width=2)

    for tick in [0, 0.25, 0.5, 0.75, 1.0]:
        x = plot_left + tick * plot_w
        y = plot_top + (1 - tick) * plot_h
        draw.line((plot_left, y, plot_left + plot_w, y), fill=(46, 73, 96), width=1)
        draw.line((x, plot_top, x, plot_top + plot_h), fill=(46, 73, 96), width=1)
        label = f"{int(tick * 100)}%"
        tw, th = text_size(draw, label, tick_font)
        draw.text((plot_left - 16 - tw, y - th / 2), label, font=tick_font, fill=MUTED)
        draw.text((x - tw / 2, plot_top + plot_h + 12), label, font=tick_font, fill=MUTED)

    if series_key == "roc":
        _draw_dashed_line(draw, (plot_left, plot_top + plot_h), (plot_left + plot_w, plot_top), SECONDARY, width=3, dash=14, gap=10)
    else:
        baseline_y = plot_top + (1 - float(graph["positive_rate"]) / 100.0) * plot_h
        _draw_dashed_line(draw, (plot_left, baseline_y), (plot_left + plot_w, baseline_y), SECONDARY, width=3, dash=14, gap=10)

    colors = [PRIMARY, SECONDARY, SUCCESS]
    legend_x = 110
    legend_y = 180
    for index, curve in enumerate(graph["curves"][:3]):
        color = colors[index % len(colors)]
        auc_value = curve["roc_auc"] if series_key == "roc" else curve["pr_auc"]
        draw.ellipse((legend_x, legend_y + index * 32, legend_x + 16, legend_y + 16 + index * 32), fill=color)
        draw.text((legend_x + 24, legend_y - 1 + index * 32), f'{curve["name"]}  |  {("ROC AUC" if series_key == "roc" else "PR AUC")} {auc_value:.2f}%', font=legend_font, fill=TEXT)

        if series_key == "roc":
            xs = curve["roc"]["fpr"]
            ys = curve["roc"]["tpr"]
        else:
            xs = curve["pr"]["recall"]
            ys = curve["pr"]["precision"]
        pts = _scale_points(xs, ys, (plot_left, plot_top, plot_w, plot_h))
        if len(pts) >= 2:
            draw.line(pts, fill=color, width=5, joint="curve")
            for point in pts[:: max(1, len(pts) // 10)]:
                draw.ellipse((point[0] - 3, point[1] - 3, point[0] + 3, point[1] + 3), fill=color)

    draw.text((plot_left + plot_w / 2 - 110, 845), x_label, font=axis_font, fill=TEXT)
    draw.text((18, plot_top + plot_h / 2 - 70), y_label, font=axis_font, fill=TEXT)
    draw.text((70, 860), "The dashed line is the baseline. Curves closer to the ideal corner are better.", font=small_font, fill=MUTED)
    return save(img, filename)


def main():
    recommender = PlatformRecommender()
    recommender.train()
    comparison = recommender.get_model_comparison()
    graph = recommender.get_graph_payload()
    draw_performance_table(comparison)
    draw_split_diagram()
    draw_confusion_matrices(comparison)
    draw_curve_chart(graph, "roc", "roc_curve.png", "ROC Curve", "True positive rate vs false positive rate", "False Positive Rate", "True Positive Rate")
    draw_curve_chart(graph, "pr", "pr_curve.png", "Precision-Recall Curve", "Precision vs recall across the top models", "Recall", "Precision")
    summary = {
        "best_model": recommender.best_model_name,
        "comparison": [
            {
                "name": row["name"],
                "accuracy": row["accuracy"],
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "decision_score": row["decision_score"],
                "confusion": row["confusion"],
            }
            for row in comparison
        ],
        "assets": {
            "performance_table": str(ASSET_DIR / "performance_table.png"),
            "train_test_split": str(ASSET_DIR / "train_test_split.png"),
            "confusion_matrices": str(ASSET_DIR / "confusion_matrices.png"),
            "roc_curve": str(ASSET_DIR / "roc_curve.png"),
            "pr_curve": str(ASSET_DIR / "pr_curve.png"),
        },
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
