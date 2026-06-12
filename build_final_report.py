from __future__ import annotations

import datetime as _dt
from pathlib import Path
from textwrap import fill
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUT_PATH = ROOT / "MatchTank_AI_Final_Report.docx"
ASSET_DIR = ROOT / "report_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(r"C:\Users\navis\OneDrive\Desktop\Data Science2")

SCREENSHOTS = [
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214041.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214052.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214133.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214155.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214219.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214230.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214451.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214514.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214556.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214608.png"),
    Path(r"C:\Users\navis\OneDrive\Pictures\Screenshots\Screenshot 2026-04-08 214708.png"),
]

ARCH_DIAGRAM = ASSET_DIR / "architecture.png"
WORKFLOW_DIAGRAM = ASSET_DIR / "workflow.png"
SPLIT_DIAGRAM = ASSET_DIR / "train_test_split.png"

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
PIC_NS = "http://schemas.openxmlformats.org/drawingml/2006/picture"
REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
PKG_CT_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
CP_NS = "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
DC_NS = "http://purl.org/dc/elements/1.1/"
DCTERMS_NS = "http://purl.org/dc/terms/"
DCTYPE_NS = "http://purl.org/dc/dcmitype/"
EMU_PER_INCH = 914400
TWIPS_PER_INCH = 1440


def twips(inches: float) -> int:
    return int(round(inches * TWIPS_PER_INCH))


def emu(inches: float) -> int:
    return int(round(inches * EMU_PER_INCH))


def _font_candidates() -> list[Path]:
    return [
        Path(r"C:\Windows\Fonts\arial.ttf"),
        Path(r"C:\Windows\Fonts\calibri.ttf"),
        Path(r"C:\Windows\Fonts\segoeui.ttf"),
        Path(r"C:\Windows\Fonts\times.ttf"),
    ]


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                Path(r"C:\Windows\Fonts\arialbd.ttf"),
                Path(r"C:\Windows\Fonts\calibrib.ttf"),
                Path(r"C:\Windows\Fonts\seguisb.ttf"),
                Path(r"C:\Windows\Fonts\timesbd.ttf"),
            ]
        )
    candidates.extend(_font_candidates())
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def box_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        words = paragraph.split()
        current = [words[0]]
        for word in words[1:]:
            candidate = " ".join(current + [word])
            if draw.textbbox((0, 0), candidate, font=font)[2] <= max_width:
                current.append(word)
            else:
                lines.append(" ".join(current))
                current = [word]
        lines.append(" ".join(current))
    return lines


def draw_round_box(draw: ImageDraw.ImageDraw, xy, fill, outline, radius=24, width=4):
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def draw_arrow(draw: ImageDraw.ImageDraw, start, end, color, width=8, head=18):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    import math

    angle = math.atan2(y2 - y1, x2 - x1)
    left = (
        x2 - head * math.cos(angle) + head * 0.6 * math.sin(angle),
        y2 - head * math.sin(angle) - head * 0.6 * math.cos(angle),
    )
    right = (
        x2 - head * math.cos(angle) - head * 0.6 * math.sin(angle),
        y2 - head * math.sin(angle) + head * 0.6 * math.cos(angle),
    )
    draw.polygon([end, left, right], fill=color)


def draw_box_with_title(draw, top_left, size, title, body, fill, outline, title_font, body_font, text_color="white"):
    x, y = top_left
    w, h = size
    draw_round_box(draw, (x, y, x + w, y + h), fill, outline, radius=28, width=4)
    draw.text((x + 28, y + 22), title, font=title_font, fill=text_color)
    body_lines = box_lines(draw, body, body_font, max_width=w - 56)
    current_y = y + 78
    step = getattr(body_font, "size", 24) + 10
    for line in body_lines:
        draw.text((x + 28, current_y), line, font=body_font, fill=text_color)
        current_y += step


def make_architecture_diagram(path: Path):
    img = Image.new("RGB", (2400, 2100), "#0B1F2E")
    draw = ImageDraw.Draw(img)
    draw.text((90, 50), "System Architecture", font=load_font(42, bold=True), fill="#39C5F4")
    draw.text((90, 110), "Layered design of the MatchTank AI platform", font=load_font(30, bold=True), fill="white")
    title_font = load_font(30, bold=True)
    body_font = load_font(24)
    boxes = [
        ((520, 220), (1360, 240), "Users", "Investors, inventors, and reviewers enter the platform through the login and dashboard flow."),
        ((520, 520), (1360, 250), "Presentation Layer", "HTML, CSS, and JavaScript pages render the login page, dashboards, analytics charts, and shared inbox."),
        ((520, 830), (1360, 270), "Flask Application Layer", "Authentication, profile updates, recommendation APIs, chat endpoints, equity calculator, and analytics routes."),
        ((520, 1160), (1360, 280), "Intelligence Layer", "PlatformRecommender loads data, builds features, trains models, evaluates metrics, and ranks matches."),
        ((520, 1500), (1360, 260), "Data Layer", "Original CSV files, augmented history, JSON stores for users and chats, and the cached model artifact."),
        ((520, 1820), (1360, 220), "Outputs", "Investor recommendations, inventor matches, shared chat, analytics dashboard, and deal support tools."),
    ]
    centers = []
    for idx, (pos, size, title, body) in enumerate(boxes):
        fill = ["#13334A", "#173D57", "#164A66", "#1A566D", "#143141", "#13334A"][idx]
        outline = ["#2E6F94", "#2D8DB5", "#2AA1D5", "#39C5F4", "#2E6F94", "#2E6F94"][idx]
        draw_box_with_title(draw, pos, size, title, body, fill, outline, title_font, body_font)
        x, y = pos
        w, h = size
        centers.append((x + w // 2, y + h))
    for i in range(len(centers) - 1):
        draw_arrow(draw, centers[i], (centers[i + 1][0], centers[i + 1][1] + 24), "#39C5F4", width=10, head=20)
    draw.text((90, 2050), "This diagram highlights how the platform separates UI, backend, ML, and storage concerns for maintainability.", font=load_font(24), fill="#B8D5E6")
    img.save(path)


def make_workflow_diagram(path: Path):
    img = Image.new("RGB", (2600, 1800), "#0B1F2E")
    draw = ImageDraw.Draw(img)
    draw.text((90, 50), "Workflow", font=load_font(42, bold=True), fill="#39C5F4")
    draw.text((90, 110), "End-to-end path from login to recommendations and feedback", font=load_font(30, bold=True), fill="white")
    box_w, box_h = 2150, 170
    x = 225
    y0 = 240
    steps = [
        ("1. Login or Signup", "The user enters the platform as an investor or inventor and reaches the matching dashboard."),
        ("2. Load Profile Data", "The backend loads investor, inventor, history, chat, and profile records."),
        ("3. Build Match Features", "The engine computes domain match, location match, risk gap, affordability, and text similarity."),
        ("4. Train and Test Models", "Four classifiers are trained on the same split and compared using multiple metrics."),
        ("5. Rank Recommendations", "The model with the highest composite score drives the final recommendation list."),
        ("6. Interact and Improve", "Users can shortlist, like, dislike, open chat, save achievements, and send feedback."),
    ]
    prev_center = None
    title_font = load_font(30, bold=True)
    body_font = load_font(24)
    for idx, (title, body) in enumerate(steps):
        fill = ["#13334A", "#173D57", "#164A66", "#1A566D", "#143141", "#13334A"][idx]
        outline = ["#2E6F94", "#2D8DB5", "#2AA1D5", "#39C5F4", "#2E6F94", "#2E6F94"][idx]
        draw_round_box(draw, (x, y0, x + box_w, y0 + box_h), fill, outline, radius=30, width=4)
        draw.text((x + 30, y0 + 20), title, font=title_font, fill="white")
        body_lines = box_lines(draw, body, body_font, max_width=box_w - 70)
        by = y0 + 72
        for line in body_lines:
            draw.text((x + 30, by), line, font=body_font, fill="#E8F4FA")
            by += getattr(body_font, "size", 24) + 10
        if prev_center is not None:
            draw_arrow(draw, prev_center, (x + box_w // 2, y0 - 12), "#39C5F4", width=9, head=18)
        prev_center = (x + box_w // 2, y0 + box_h)
        y0 += 235
    draw.text((90, 1670), "The workflow is intentionally simple for presentation, but the backend is structured so each stage can evolve independently.", font=load_font(24), fill="#B8D5E6")
    img.save(path)


def make_split_diagram(path: Path):
    img = Image.new("RGB", (2200, 780), "#0B1F2E")
    draw = ImageDraw.Draw(img)
    draw.text((90, 50), "Train-Test Split", font=load_font(42, bold=True), fill="#39C5F4")
    draw.text((90, 110), "The model is evaluated on unseen history to keep the score honest", font=load_font(30, bold=True), fill="white")
    boxes = [
        ((120, 250), (600, 260), "Full History Dataset", "All observed investor-idea interactions are loaded from the raw history file."),
        ((860, 250), (520, 260), "Training Set 75%", "The model learns patterns from this portion, together with weighted synthetic rows."),
        ((1560, 250), (520, 260), "Testing Set 25%", "This slice stays unseen during training and is used to compute the reported metrics."),
    ]
    for idx, (pos, size, title, body) in enumerate(boxes):
        fill = ["#173D57", "#164A66", "#143141"][idx]
        outline = ["#2D8DB5", "#2AA1D5", "#39C5F4"][idx]
        draw_box_with_title(draw, pos, size, title, body, fill, outline, load_font(30, bold=True), load_font(24))
    draw_arrow(draw, (720, 380), (840, 380), "#39C5F4", width=10, head=20)
    draw_arrow(draw, (1400, 380), (1540, 380), "#39C5F4", width=10, head=20)
    draw.text((90, 610), "A stratified split keeps positive and negative samples balanced so the evaluation remains fair.", font=load_font(24), fill="#B8D5E6")
    img.save(path)


def build_styles_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{W_NS}">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
        <w:sz w:val="22"/>
        <w:szCs w:val="22"/>
        <w:color w:val="1F1F1F"/>
      </w:rPr>
    </w:rPrDefault>
    <w:pPrDefault>
      <w:pPr>
        <w:spacing w:after="120" w:line="276" w:lineRule="auto"/>
      </w:pPr>
    </w:pPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
      <w:sz w:val="22"/>
      <w:color w:val="1F1F1F"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="0" w:after="160"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Cambria" w:hAnsi="Cambria" w:cs="Cambria"/>
      <w:b/>
      <w:sz w:val="60"/>
      <w:color w:val="0E2A47"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="Heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:keepNext/>
      <w:spacing w:before="320" w:after="140"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Cambria" w:hAnsi="Cambria" w:cs="Cambria"/>
      <w:b/>
      <w:sz w:val="36"/>
      <w:color w:val="0E2A47"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="Heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:keepNext/>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Cambria" w:hAnsi="Cambria" w:cs="Cambria"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:color w:val="144A74"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="Heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:keepNext/>
      <w:spacing w:before="200" w:after="100"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Cambria" w:hAnsi="Cambria" w:cs="Cambria"/>
      <w:b/>
      <w:sz w:val="24"/>
      <w:color w:val="1E6AA9"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="BodyText">
    <w:name w:val="Body Text"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="120" w:line="276" w:lineRule="auto"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
      <w:sz w:val="22"/>
      <w:color w:val="1F1F1F"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="80" w:after="180"/>
      <w:jc w:val="center"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
      <w:i/>
      <w:sz w:val="18"/>
      <w:color w:val="4D6478"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="60"/>
      <w:ind w:left="360" w:right="360"/>
      <w:shd w:fill="F4F7FA"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Consolas" w:hAnsi="Consolas" w:cs="Consolas"/>
      <w:sz w:val="18"/>
      <w:color w:val="1E1E1E"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableCell">
    <w:name w:val="Table Cell"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:after="0"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
      <w:sz w:val="18"/>
      <w:color w:val="1F1F1F"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="TableHeader">
    <w:name w:val="Table Header"/>
    <w:basedOn w:val="TableCell"/>
    <w:qFormat/>
    <w:rPr>
      <w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Calibri"/>
      <w:b/>
      <w:sz w:val="18"/>
      <w:color w:val="FFFFFF"/>
    </w:rPr>
  </w:style>
</w:styles>
'''


def build_settings_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="{W_NS}">
  <w:zoom w:percent="100"/>
  <w:defaultTabStop w:val="720"/>
  <w:characterSpacingControl w:val="doNotCompress"/>
  <w:compat>
    <w:compatSetting w:name="compatibilityMode" w:val="15" w:uri="http://schemas.microsoft.com/office/word"/>
  </w:compat>
</w:settings>
'''


def build_font_table_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:fonts xmlns:w="{W_NS}">
  <w:font w:name="Calibri"/>
  <w:font w:name="Cambria"/>
  <w:font w:name="Consolas"/>
  <w:font w:name="Arial"/>
</w:fonts>
'''


def build_web_settings_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:webSettings xmlns:w="{W_NS}">
  <w:optimizeForBrowser/>
</w:webSettings>
'''


def build_core_props_xml() -> str:
    now = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="{CP_NS}" xmlns:dc="{DC_NS}" xmlns:dcterms="{DCTERMS_NS}" xmlns:dcmitype="{DCTYPE_NS}" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>MatchTank AI Final Report</dc:title>
  <dc:subject>Investor-Inventor Matchmaking Platform</dc:subject>
  <dc:creator>Codex</dc:creator>
  <cp:keywords>machine learning, recommendation system, investor, inventor, MatchTank AI</cp:keywords>
  <dc:description>Final report for the MatchTank AI investor-inventor matchmaking platform.</dc:description>
  <cp:lastModifiedBy>Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{now}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{now}</dcterms:modified>
</cp:coreProperties>
'''


def build_app_props_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties" xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <HeadingPairs>
    <vt:vector size="2" baseType="variant">
      <vt:variant>
        <vt:lpstr>Title</vt:lpstr>
      </vt:variant>
      <vt:variant>
        <vt:i4>1</vt:i4>
      </vt:variant>
    </vt:vector>
  </HeadingPairs>
  <TitlesOfParts>
    <vt:vector size="1" baseType="lpstr">
      <vt:lpstr>MatchTank AI Final Report</vt:lpstr>
    </vt:vector>
  </TitlesOfParts>
  <Company>OpenAI</Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
'''


def build_content_types_xml() -> str:
    image_defaults = "".join(
        f'<Default Extension="{ext}" ContentType="{ctype}"/>'
        for ext, ctype in [
            ("png", "image/png"),
            ("jpg", "image/jpeg"),
            ("jpeg", "image/jpeg"),
        ]
    )
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="{PKG_CT_NS}">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  {image_defaults}
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/word/fontTable.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.fontTable+xml"/>
  <Override PartName="/word/webSettings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.webSettings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
'''


def build_root_rels_xml() -> str:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
'''


class ReportBuilder:
    def __init__(self):
        self.body_parts: list[str] = []
        self.media_files: list[tuple[str, bytes]] = []
        self.next_image_index = 1
        self.docpr_id = 1

    def add_paragraph(
        self,
        text: str = "",
        style: str = "BodyText",
        align: str | None = None,
        bold: bool = False,
        italic: bool = False,
        page_break_before: bool = False,
    ) -> None:
        ppr = [f'<w:pStyle w:val="{style}"/>']
        if align:
            ppr.append(f'<w:jc w:val="{align}"/>')
        if page_break_before:
            ppr.append("<w:pageBreakBefore/>")
        rpr = []
        if bold:
            rpr.append("<w:b/>")
        if italic:
            rpr.append("<w:i/>")
        run_xml = f'<w:r><w:rPr>{"".join(rpr)}</w:rPr><w:t xml:space="preserve">{escape(text)}</w:t></w:r>' if text else ""
        self.body_parts.append(f'<w:p><w:pPr>{"".join(ppr)}</w:pPr>{run_xml}</w:p>')

    def add_bullets(self, items: list[str]) -> None:
        for item in items:
            self.add_paragraph(f"• {item}", style="BodyText")

    def add_table(
        self,
        rows: list[list[str]],
        widths: list[int],
        header_fill: str = "0E2A47",
        body_fill: str = "FFFFFF",
        alignments: list[str] | None = None,
    ) -> None:
        alignments = alignments or ["left"] * len(widths)
        tbl_pr = f'''
          <w:tblPr>
            <w:tblW w:w="0" w:type="auto"/>
            <w:tblBorders>
              <w:top w:val="single" w:sz="8" w:space="0" w:color="B7C6D6"/>
              <w:left w:val="single" w:sz="8" w:space="0" w:color="B7C6D6"/>
              <w:bottom w:val="single" w:sz="8" w:space="0" w:color="B7C6D6"/>
              <w:right w:val="single" w:sz="8" w:space="0" w:color="B7C6D6"/>
              <w:insideH w:val="single" w:sz="8" w:space="0" w:color="B7C6D6"/>
              <w:insideV w:val="single" w:sz="8" w:space="0" w:color="B7C6D6"/>
            </w:tblBorders>
            <w:tblLook w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/>
          </w:tblPr>
        '''
        grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
        rows_xml = []
        for r_idx, row in enumerate(rows):
            cells = []
            for c_idx, cell in enumerate(row):
                is_header = r_idx == 0
                fill = header_fill if is_header else body_fill if r_idx % 2 else "F7FAFD"
                style = "TableHeader" if is_header else "TableCell"
                align = alignments[c_idx] if c_idx < len(alignments) else "left"
                cells.append(
                    f'''
                    <w:tc>
                      <w:tcPr>
                        <w:tcW w:w="{widths[c_idx]}" w:type="dxa"/>
                        <w:shd w:fill="{fill}"/>
                      </w:tcPr>
                      <w:p>
                        <w:pPr>
                          <w:pStyle w:val="{style}"/>
                          <w:jc w:val="{align}"/>
                        </w:pPr>
                        <w:r>
                          <w:rPr>{"<w:b/>" if is_header else ""}</w:rPr>
                          <w:t xml:space="preserve">{escape(str(cell))}</w:t>
                        </w:r>
                      </w:p>
                    </w:tc>
                    '''
                )
            rows_xml.append(f"<w:tr>{''.join(cells)}</w:tr>")
        self.body_parts.append(f'<w:tbl>{tbl_pr}<w:tblGrid>{grid}</w:tblGrid>{"".join(rows_xml)}</w:tbl>')

    def add_image(self, image_path: Path, width_inches: float = 6.4, caption: str | None = None) -> None:
        if not image_path.exists():
            self.add_paragraph(f"[Missing image: {image_path.name}]", style="Caption", italic=True)
            return
        with Image.open(image_path) as im:
            w_px, h_px = im.size
            height_inches = width_inches * (h_px / w_px)
            width_emu = emu(width_inches)
            height_emu = emu(height_inches)
            payload = image_path.read_bytes()
        media_name = f"image{self.next_image_index}.png"
        rel_id = f"rId{4 + self.next_image_index}"
        self.media_files.append((media_name, payload))
        self.body_parts.append(
            f'''
            <w:p>
              <w:pPr><w:jc w:val="center"/></w:pPr>
              <w:r>
                <w:drawing>
                  <wp:inline distT="0" distB="0" distL="0" distR="0">
                    <wp:extent cx="{width_emu}" cy="{height_emu}"/>
                    <wp:docPr id="{self.docpr_id}" name="Picture {self.docpr_id}"/>
                    <wp:cNvGraphicFramePr>
                      <a:graphicFrameLocks noChangeAspect="1"/>
                    </wp:cNvGraphicFramePr>
                    <a:graphic>
                      <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
                        <pic:pic>
                          <pic:nvPicPr>
                            <pic:cNvPr id="0" name="{escape(media_name)}"/>
                            <pic:cNvPicPr/>
                          </pic:nvPicPr>
                          <pic:blipFill>
                            <a:blip r:embed="{rel_id}"/>
                            <a:stretch><a:fillRect/></a:stretch>
                          </pic:blipFill>
                          <pic:spPr>
                            <a:xfrm>
                              <a:off x="0" y="0"/>
                              <a:ext cx="{width_emu}" cy="{height_emu}"/>
                            </a:xfrm>
                            <a:prstGeom prst="rect">
                              <a:avLst/>
                            </a:prstGeom>
                          </pic:spPr>
                        </pic:pic>
                      </a:graphicData>
                    </a:graphic>
                  </wp:inline>
                </w:drawing>
              </w:r>
            </w:p>
            '''
        )
        if caption:
            self.add_paragraph(caption, style="Caption")
        self.docpr_id += 1
        self.next_image_index += 1

    def add_page_break(self) -> None:
        self.body_parts.append('<w:p><w:r><w:br w:type="page"/></w:r></w:p>')

    def add_code_block(self, text: str) -> None:
        for line in text.strip("\n").splitlines():
            self.add_paragraph(line, style="Code")
        self.body_parts.append('<w:p><w:r><w:t xml:space="preserve"> </w:t></w:r></w:p>')

    def build_document_xml(self) -> str:
        sect_pr = '''
        <w:sectPr>
          <w:pgSz w:w="11906" w:h="16838"/>
          <w:pgMar w:top="1080" w:right="1080" w:bottom="1080" w:left="1080" w:header="720" w:footer="720" w:gutter="0"/>
          <w:cols w:space="720"/>
          <w:docGrid w:linePitch="360"/>
        </w:sectPr>
        '''
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:wp="{WP_NS}" xmlns:a="{A_NS}" xmlns:pic="{PIC_NS}">
  <w:body>
    {"".join(self.body_parts)}
    {sect_pr}
  </w:body>
</w:document>
'''

    def build_doc_rels_xml(self) -> str:
        rels = [
            ("rId1", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles", "styles.xml"),
            ("rId2", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings", "settings.xml"),
            ("rId3", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/fontTable", "fontTable.xml"),
            ("rId4", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/webSettings", "webSettings.xml"),
        ]
        for idx in range(1, len(self.media_files) + 1):
            rels.append(
                (
                    f"rId{4 + idx}",
                    "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image",
                    f"media/image{idx}.png",
                )
            )
        rel_xml = "".join(
            f'<Relationship Id="{rid}" Type="{typ}" Target="{target}"/>'
            for rid, typ, target in rels
        )
        return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{REL_NS}">
  {rel_xml}
</Relationships>
'''

    def build_package(self, out_path: Path) -> None:
        with ZipFile(out_path, "w", ZIP_DEFLATED) as zf:
            zf.writestr("[Content_Types].xml", build_content_types_xml())
            zf.writestr("_rels/.rels", build_root_rels_xml())
            zf.writestr("docProps/core.xml", build_core_props_xml())
            zf.writestr("docProps/app.xml", build_app_props_xml())
            zf.writestr("word/document.xml", self.build_document_xml())
            zf.writestr("word/styles.xml", build_styles_xml())
            zf.writestr("word/settings.xml", build_settings_xml())
            zf.writestr("word/fontTable.xml", build_font_table_xml())
            zf.writestr("word/webSettings.xml", build_web_settings_xml())
            zf.writestr("word/_rels/document.xml.rels", self.build_doc_rels_xml())
            for idx, (name, payload) in enumerate(self.media_files, start=1):
                zf.writestr(f"word/media/{name}", payload)


def dataset_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        return sum(1 for _ in f) - 1


def make_report():
    make_architecture_diagram(ARCH_DIAGRAM)
    make_workflow_diagram(WORKFLOW_DIAGRAM)
    make_split_diagram(SPLIT_DIAGRAM)

    builder = ReportBuilder()

    builder.add_paragraph("MATCHTANK AI", style="Heading3", align="center")
    builder.add_paragraph("Investor-Inventor Matchmaking Platform", style="Title", align="center")
    builder.add_paragraph("Final Project Report", style="Heading1", align="center")
    builder.add_paragraph(
        "A machine learning based platform that connects investors with inventors, explains model selection with analytics, and supports live communication through a shared inbox.",
        style="BodyText",
        align="center",
    )
    builder.add_table(
        [
            ["Project Field", "Details"],
            ["Domain", "Machine learning recommendation system"],
            ["Core Models", "Logistic Regression, Random Forest, Gradient Boosting, Decision Tree"],
            ["Best Model", "Gradient Boosting by composite performance score"],
            ["Main Stack", "Python, Flask, pandas, scikit-learn, HTML, CSS, JavaScript"],
        ],
        widths=[2800, 6200],
        alignments=["left", "left"],
    )
    builder.add_paragraph("Prepared for the final review and faculty evaluation.", style="Caption")
    builder.add_page_break()

    builder.add_paragraph("Abstract", style="Heading1")
    builder.add_paragraph(
        "MatchTank AI is an investor-inventor matchmaking platform designed to automate startup discovery and funding matching. The system reads investor profiles, inventor profiles, and historical interaction records, then predicts which inventor ideas best fit each investor. It combines a Flask web application, a machine learning recommendation engine, role-based dashboards, a shared chat inbox, and an equity calculator in one submission-ready platform.",
    )
    builder.add_paragraph(
        "The recommender compares four classification models trained on engineered pairwise features: Logistic Regression, Random Forest, Gradient Boosting, and Decision Tree. Model quality is measured using accuracy, precision, recall, F1, ROC AUC, and PR AUC. The final winner is selected using a balanced composite score, so the report can justify the model choice in a transparent and academic way.",
    )

    builder.add_paragraph("Acknowledgement", style="Heading1")
    builder.add_paragraph(
        "I would like to express my sincere gratitude to my project guide, faculty members, and everyone who supported the development of this work. This project was completed as a practical implementation of machine learning, web development, and recommendation system concepts, and it strengthened my understanding of how a real product can be designed around a model rather than around a single prediction number.",
    )

    builder.add_paragraph("Contents", style="Heading1")
    builder.add_bullets(
        [
            "Introduction and problem statement",
            "Objectives and scope",
            "Dataset overview",
            "System architecture",
            "Workflow and methodology",
            "Feature engineering and training pipeline",
            "Model comparison and selection",
            "Evaluation metrics and graphs",
            "User interface evidence",
            "Core Python code overview",
            "Platform features",
            "Limitations and future scope",
            "Conclusion",
        ]
    )

    builder.add_paragraph("1. Introduction", style="Heading1")
    builder.add_paragraph(
        "The idea behind MatchTank AI is simple: investors should not have to manually search through hundreds of startup ideas, and inventors should not have to rely on random discovery to find funding partners. The platform works as a digital middle layer between both sides, similar to a Shark Tank style matching system, but with machine learning as the decision engine.",
    )
    builder.add_paragraph(
        "Instead of returning a static sample list, the system scores the entire dataset and produces personalized rankings. It looks at domain fit, location fit, risk compatibility, affordability, and textual similarity between the investor profile and the inventor idea. Because the same data is used for model comparison and dashboard display, the report can show both technical accuracy and product usability.",
    )

    builder.add_paragraph("2. Problem Statement and Objectives", style="Heading1")
    builder.add_paragraph(
        "The problem is that startup discovery is noisy and unstructured. Investors often waste time filtering ideas that do not match their preferred sector or risk appetite, while inventors struggle to identify the investors most likely to understand and fund their idea. The platform addresses this gap by converting the match process into a structured recommendation problem.",
    )
    builder.add_bullets(
        [
            "Connect investors with suitable inventor ideas using machine learning.",
            "Provide a clean login and signup flow with role-based dashboards.",
            "Explain the recommendation engine with metrics, curves, and confusion matrices.",
            "Enable feedback, shortlisting, and shared chat so the platform behaves like a real middleware.",
            "Support deal discussion with an equity calculator and inventor profile details.",
        ]
    )

    builder.add_paragraph("3. Dataset Overview", style="Heading1")
    builder.add_paragraph(
        "The original dataset was stored in C:\\Users\\navis\\OneDrive\\Desktop\\Data Science2. It contains three core CSV files: investors.csv, inventors.csv, and history.csv. The project also creates history_augmented.csv in the application folder to provide a denser learning signal for training. The augmented rows help the model learn more stable patterns, but the final reported metrics are still based on a held-out slice of the raw history.",
    )
    builder.add_table(
        [
            ["File", "Rows", "Main Columns", "Purpose"],
            [
                "investors.csv",
                str(dataset_count(DATA_DIR / "investors.csv")),
                "investor_id, investor_name, preferred_risk_appetite, company_investment, past_investments, preferred_location, industry_focus, available_funds, focus_domain",
                "Stores investor profiles and investment preferences",
            ],
            [
                "inventors.csv",
                str(dataset_count(DATA_DIR / "inventors.csv")),
                "idea_id, idea_title, idea_text, domain, funding_st_required, f_team_size, location, technology, risk_level",
                "Stores startup ideas and inventor profile details",
            ],
            [
                "history.csv",
                str(dataset_count(DATA_DIR / "history.csv")),
                "history_id, investor_id, idea_id, year, interaction_score",
                "Stores historical investor-inventor interactions",
            ],
            [
                "history_augmented.csv",
                str(dataset_count(ROOT / "data" / "history_augmented.csv")),
                "Same core fields with synthetic rows added for training",
                "Improves the learning signal used by the recommender",
            ],
        ],
        widths=[1700, 800, 4200, 2900],
        alignments=["left", "center", "left", "left"],
    )
    builder.add_paragraph(
        "The investor file captures available funds, company investment, past investments, preferred location, industry focus, and preferred risk appetite. The inventor file captures the startup idea title, textual summary, domain, technology, required funding, team size, location, and risk level. The interaction history is used to learn which investor-idea pairs previously behaved like strong matches.",
    )

    builder.add_paragraph("4. System Architecture", style="Heading1")
    builder.add_paragraph(
        "The platform follows a layered design. The presentation layer contains the login page, investor dashboard, inventor dashboard, analytics dashboard, and shared chat components. The Flask backend manages authentication, route handling, profile persistence, recommendation APIs, chat APIs, and the equity calculator. The intelligence layer is the recommender engine, which handles feature engineering, model training, evaluation, caching, and ranked output generation. The data layer keeps the project connected to the CSV files and the JSON/SQLite stores that preserve users, profiles, messages, feedback, and cached model outputs.",
    )
    builder.add_image(ARCH_DIAGRAM, caption="Figure 1. System architecture of the MatchTank AI platform.")
    builder.add_bullets(
        [
            "Presentation layer: HTML, CSS, JavaScript, and responsive dashboard pages.",
            "Application layer: Flask routes, login/signup flow, dashboards, analytics, chat, and equity calculator.",
            "Intelligence layer: feature engineering, model training, metrics, and recommendation ranking.",
            "Data layer: investors.csv, inventors.csv, history.csv, history_augmented.csv, and JSON persistence.",
        ]
    )

    builder.add_paragraph("5. Workflow and Methodology", style="Heading1")
    builder.add_paragraph(
        "The workflow starts when a user logs in or signs up as an investor or inventor. The backend loads the relevant profile and builds the feature vectors for every investor-idea pair. Historical interactions are split into training and testing subsets so that the model can be evaluated on unseen data. Four classifiers are trained and compared using the same feature set. The best model is selected using a composite score, and the ranked recommendations are then displayed in the dashboard. Users can continue the cycle by liking, disliking, shortlisting, chatting, and updating inventor profiles.",
    )
    builder.add_image(WORKFLOW_DIAGRAM, caption="Figure 2. End-to-end workflow from login to recommendations and feedback.")
    builder.add_paragraph("Train-test split used in the evaluation pipeline:", style="Heading3")
    builder.add_image(SPLIT_DIAGRAM, caption="Figure 3. Stratified train-test split used to keep evaluation fair.")
    builder.add_bullets(
        [
            "75% of observed history is used for training.",
            "25% of observed history is kept unseen for evaluation.",
            "Synthetic rows help the model learn better patterns, but they do not decide the reported test score.",
            "Stratification keeps positive and negative matches balanced in both splits.",
        ]
    )

    builder.add_paragraph("6. Feature Engineering and Training Pipeline", style="Heading1")
    builder.add_paragraph(
        "The quality of the recommender depends on feature engineering. Each investor-idea pair is converted into a structured set of numeric and categorical features. The model does not just look at IDs; it learns from domain fit, location fit, risk compatibility, affordability, and profile text similarity. The TF-IDF layer compares investor interests with inventor descriptions so the model can capture semantic overlap, not only exact category matches.",
    )
    builder.add_bullets(
        [
            "Domain match: whether investor focus domain and inventor domain are the same.",
            "Location match: whether preferred location and idea location align.",
            "Risk gap: the absolute difference between investor risk appetite and inventor risk level.",
            "Affordability ratio: available funds divided by funding required.",
            "Text similarity: TF-IDF based overlap between the two profile texts.",
            "Heuristic score: a weighted compatibility score used for augmentation and ranking support.",
            "Historical rates: investor positive rate, idea positive rate, and domain/technology/location positive rates.",
        ]
    )
    builder.add_paragraph(
        "The training pipeline uses median imputation for numeric values, most-frequent imputation for categorical values, one-hot encoding for categories, and standard scaling for numeric features. The models compared are Logistic Regression, Random Forest, Gradient Boosting, and Decision Tree. Each model is trained on the same pairwise data and then evaluated on the same held-out test split.",
    )
    builder.add_code_block(
        """
train():
    ensure_augmented_history()
    load_data()
    prepare_text_features()
    observed_rows = build_observed_rows(history_df)
    train_rows, test_rows = train_test_split(observed_rows, stratify=label)
    synthetic_rows = build_synthetic_training_rows()
    if synthetic_rows is not empty:
        train_rows = concat(train_rows, synthetic_rows)
    for each model in model_specs:
        pipeline = build_pipeline(model)
        pipeline.fit(X_train, y_train, sample_weight=w_train)
        probabilities = pipeline.predict_proba(X_test)
        predictions = threshold(probabilities, 0.5)
        compute accuracy, precision, recall, f1, roc_auc, pr_auc
        decision_score = average(all metrics)
    best_model = max(decision_score)
        """
    )

    builder.add_paragraph("7. Model Comparison and Selection", style="Heading1")
    builder.add_paragraph(
        "The model comparison table shows how the four classifiers perform on the held-out test slice. The important point is that accuracy alone is not enough for this project. Logistic Regression slightly edges the others on raw accuracy, but Gradient Boosting gives the best overall balance when the curve-based metrics are included. That is why the final winner is selected using a composite decision score rather than a single number.",
    )
    builder.add_table(
        [
            ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC", "PR AUC", "Decision Score"],
            ["Gradient Boosting", "91.60%", "90.91%", "93.02%", "91.95%", "98.82%", "98.98%", "94.21%"],
            ["Logistic Regression", "92.00%", "92.25%", "92.25%", "92.25%", "98.05%", "98.45%", "94.21%"],
            ["Random Forest", "92.00%", "90.98%", "93.80%", "92.37%", "97.28%", "97.54%", "93.99%"],
            ["Decision Tree", "89.20%", "89.23%", "89.92%", "89.58%", "89.18%", "85.44%", "88.76%"],
        ],
        widths=[2100, 950, 950, 950, 850, 1100, 1100, 1100],
        alignments=["left", "center", "center", "center", "center", "center", "center", "center"],
    )
    builder.add_paragraph(
        "Decision score = (accuracy + precision + recall + F1 + ROC AUC + PR AUC) / 6. The same weight is used for each metric so that no single score dominates the final choice. If two models are very close, ROC AUC and PR AUC act as the practical tie-breakers because they better describe ranking quality in recommendation problems.",
    )
    builder.add_paragraph(
        "Gradient Boosting is the best model for the report because it combines the strongest ROC and PR curves with balanced classification scores. In other words, it is not only accurate, but also better at ranking strong matches near the top of the list, which is exactly what the investor-inventor recommender needs.",
    )

    builder.add_paragraph("8. Evaluation Metrics and Graphs", style="Heading1")
    builder.add_paragraph(
        "The confusion matrix helps explain the types of errors the model makes. True Positive means a correct strong match, True Negative means a correct weak match, False Positive means a bad match was recommended, and False Negative means a good match was missed. For a recommendation system, a good balance matters: too many false positives make the platform noisy, while too many false negatives cause the platform to miss good opportunities.",
    )
    builder.add_paragraph(
        "ROC curves show True Positive Rate against False Positive Rate. Curves closer to the top-left corner are stronger because they indicate that the model finds more positive cases while producing fewer false alarms. Precision-Recall curves show precision against recall and are especially useful when the positive class is important. In this project, the ROC and PR curves are used as visual proof that Gradient Boosting performs better overall than the other candidates.",
    )
    builder.add_paragraph("ROC and PR curve screenshots from the analytics dashboard:", style="Heading3")
    builder.add_image(SCREENSHOTS[8], caption="Figure 4. ROC curve and precision-recall curve for the top three models.")
    builder.add_paragraph("Metric heatmap comparison from the analytics dashboard:", style="Heading3")
    builder.add_image(SCREENSHOTS[9], caption="Figure 5. Comparison matrix showing the metrics for each model.")
    builder.add_paragraph("Model-by-model confusion matrix breakdown:", style="Heading3")
    builder.add_image(SCREENSHOTS[10], caption="Figure 6. Confusion matrices for the four compared classifiers.")
    builder.add_paragraph(
        "The interpretation is straightforward: Gradient Boosting and Logistic Regression are very close in overall performance, Random Forest remains competitive, and Decision Tree is weaker on almost every metric. The graph set makes the model choice understandable even to a non-technical reviewer.",
    )

    builder.add_paragraph("9. Login Page", style="Heading1")
    builder.add_paragraph(
        "The login page is the entry point for the platform. It introduces the project, shows the demo account information, and provides the user credentials form with a show or hide password control. The page is deliberately clean so that the review starts with a strong first impression rather than a crowded interface.",
    )
    builder.add_image(SCREENSHOTS[0], caption="Figure 7. Main login landing page with demo statistics.")
    builder.add_image(SCREENSHOTS[1], caption="Figure 8. Login form with show password button and account creation link.")

    builder.add_paragraph("10. Investor Dashboard", style="Heading1")
    builder.add_paragraph(
        "The investor dashboard is the decision-making space for the investor role. It shows the investor profile summary, current best model, the recommendation table, equity calculator, analytics shortcut, and a shared chat inbox. The design intentionally keeps the information compact so that the reviewer can see the recommendation engine, deal support, and communication tools in one place.",
    )
    builder.add_image(SCREENSHOTS[2], caption="Figure 9. Investor dashboard overview with model winner and equity calculator.")
    builder.add_paragraph(
        "The recommendation cards include the startup title, domain, technology, risk level, funding need, and a short explanatory reason such as domain fit or affordability. Each card also exposes actions such as shortlist, like, dislike, view achievements, and chat. This makes the dashboard interactive and closer to a real product than a static recommendation list.",
    )
    builder.add_image(SCREENSHOTS[4], caption="Figure 10. Investor recommendation list showing ranked inventor ideas.")
    builder.add_image(SCREENSHOTS[5], caption="Figure 11. Continued investor recommendations with action buttons and score tags.")

    builder.add_paragraph("11. Shared Chat Inbox", style="Heading1")
    builder.add_paragraph(
        "The chat system is not two isolated message boxes. It is a shared inbox that shows conversations, contacts, unread counts, and message previews on both the investor and inventor side. The search field can look through investors, inventors, ideas, and message text, so the user can discover anyone in the dataset rather than only the recommended shortlist.",
    )
    builder.add_image(SCREENSHOTS[3], caption="Figure 12. Shared inbox with search bar, conversation list, and people-to-contact panel.")
    builder.add_image(SCREENSHOTS[7], caption="Figure 13. Open chat thread showing the actual message exchange between both sides.")
    builder.add_paragraph(
        "The sender name in the inbox is shown clearly, and unread counts help the reviewer see that the system can notify the opposite side. The inbox is therefore a real communication layer, not just a decorative widget. This is useful in a faculty review because it shows that the platform supports continuous interaction after the recommendation is generated.",
    )

    builder.add_paragraph("12. Inventor Dashboard", style="Heading1")
    builder.add_paragraph(
        "The inventor dashboard is the profile and outreach workspace for the inventor role. It allows the founder to update the profile, add a project summary, record achievements, list patents, and view investor matches ranked by the model. The search and filter controls make it easier to identify which investors are suitable for the idea. This is the view that turns the platform into a middleware between inventors and investors.",
    )
    builder.add_image(SCREENSHOTS[6], caption="Figure 14. Inventor dashboard with profile editing and ranked investor matches.")

    builder.add_paragraph("13. Analytics Dashboard", style="Heading1")
    builder.add_paragraph(
        "The analytics dashboard explains why the chosen model is the winner. It displays the ROC curve, precision-recall curve, metric heatmap, and confusion matrix in a presentation-friendly layout. This page is essential for the second review because it proves that the model selection is based on measured evidence, not on guesswork.",
    )
    builder.add_image(SCREENSHOTS[8], caption="Figure 15. ROC and precision-recall curves used in the model comparison.")
    builder.add_image(SCREENSHOTS[9], caption="Figure 16. Metric comparison heatmap for all four models.")
    builder.add_image(SCREENSHOTS[10], caption="Figure 17. Confusion matrix view with true positives, true negatives, false positives, and false negatives.")

    builder.add_paragraph("14. Core Python Code Overview", style="Heading1")
    builder.add_paragraph(
        "The project is organized around two main Python files. app.py contains the Flask routes, login and signup flow, dashboard rendering, profile persistence, chat handlers, feedback APIs, analytics endpoints, and equity calculator. recommender.py contains the core machine learning logic: data loading, augmentation, feature engineering, model training, evaluation, cached comparison, and recommendation ranking. dashboard.js handles the client-side inbox, filtering, and interaction buttons.",
    )
    builder.add_table(
        [
            ["File", "Responsibility"],
            ["app.py", "Flask app, authentication, dashboards, chat, analytics, equity calculator, JSON persistence"],
            ["recommender.py", "Data loading, feature engineering, model training, evaluation, recommendation scoring"],
            ["dashboard.js", "Inbox interactions, filters, chat actions, recommendation actions"],
            ["style.css", "Responsive layout and visual theme"],
        ],
        widths=[2200, 7000],
        alignments=["left", "left"],
    )
    builder.add_paragraph("Selected training pipeline logic:", style="Heading3")
    builder.add_code_block(
        """
observed_rows = build_observed_rows(history_df)
train_rows, test_rows = train_test_split(
    observed_rows, test_size=0.25, stratify=observed_rows["label"]
)
synthetic_rows = build_synthetic_training_rows()
if synthetic_rows is not empty:
    train_rows = concat(train_rows, synthetic_rows)

for each model in model_specs:
    pipeline = build_pipeline(estimator)
    pipeline.fit(X_train, y_train, sample_weight=w_train)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = probabilities >= 0.5
    metrics = accuracy, precision, recall, f1, roc_auc, pr_auc
    decision_score = average(metrics)
best_model = model with max(decision_score)
        """
    )
    builder.add_paragraph("Selected recommendation logic:", style="Heading3")
    builder.add_code_block(
        """
candidate_pairs = all inventor ideas for one investor
scores = predict_proba(candidate_pairs) for each trained model
best_score = score from the selected best model
sort candidate pairs by best_score
return top matches with model scores and human-readable reasons
        """
    )
    builder.add_paragraph("Key API routes:", style="Heading3")
    builder.add_table(
        [
            ["Route", "Purpose"],
            ["/login", "Login page"],
            ["/signup", "Signup page"],
            ["/investor/dashboard", "Investor dashboard"],
            ["/inventor/dashboard", "Inventor dashboard"],
            ["/analytics", "Analytics page"],
            ["/api/investor/<id>/recommendations", "Investor recommendations"],
            ["/api/inventor/<idea_id>/matches", "Investor matches for an inventor"],
            ["/api/chat/<investor_id>/<idea_id>", "Fetch and post chat messages"],
            ["/api/equity-calculator", "Equity split calculations"],
        ],
        widths=[3300, 5900],
        alignments=["left", "left"],
    )

    builder.add_paragraph("15. Platform Features", style="Heading1")
    builder.add_bullets(
        [
            "Role-based login and signup for investors and inventors.",
            "Separate dashboards with recommendation, profile, analytics, and chat sections.",
            "Like, dislike, and shortlist actions for the investor side.",
            "Achievement and patent editing for inventor profiles.",
            "Shared chat inbox with search, unread badge, and conversation previews.",
            "Equity calculator for quick deal discussion.",
            "Analytics page with performance table, ROC curve, PR curve, heatmap, and confusion matrix.",
            "Model cache for faster repeated runs and a cleaner demo experience.",
        ]
    )

    builder.add_paragraph("16. Limitations and Future Scope", style="Heading1")
    builder.add_paragraph(
        "The project uses a mixture of original and synthetic data, which is excellent for a submission demo but still not the same as a live production marketplace with many years of authentic user behavior. The current recommendation quality is therefore best interpreted as a strong prototype rather than a final production-grade matching service.",
    )
    builder.add_bullets(
        [
            "Add more real interaction history to improve generalization.",
            "Replace JSON persistence with a full database-backed message store if the system scales.",
            "Add real-time WebSocket notifications for chat and unread counts.",
            "Provide explainability cards so investors can see why a match was recommended.",
            "Export analytics and deal notes as PDF reports for presentation and sharing.",
            "Add meeting scheduling and reply reminders to move from recommendation to conversion.",
        ]
    )

    builder.add_paragraph("17. Conclusion", style="Heading1")
    builder.add_paragraph(
        "MatchTank AI demonstrates how machine learning can be used to create a practical investor-inventor matchmaking system. The platform combines recommendation, communication, analytics, and deal support into one clear workflow. The final model is selected using a balanced score that considers accuracy, precision, recall, F1, ROC AUC, and PR AUC, so the explanation is technically strong and easy to defend during review. Gradient Boosting is selected as the best model because it gives the most reliable balance for this recommendation problem.",
    )
    builder.add_paragraph(
        "Overall, the project is submission-ready because it includes the dataset, the training pipeline, model comparison, dashboard evidence, evaluation charts, and the core code structure in one consistent report.",
    )

    builder.build_package(OUT_PATH)


if __name__ == "__main__":
    make_report()
    print(f"Report generated: {OUT_PATH}")
