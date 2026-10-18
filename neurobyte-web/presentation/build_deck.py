#!/usr/bin/env python
"""Build a NeuroBytes deck (16:9) that REPLICATES THE WEBSITE.

It mirrors the live site's LIGHT "tea green" theme and its sections: the hero
("See what the brain watched." + 4-stat grid), "Three seconds, five steps." +
feature grid, the Results bar chart (J1 -> C13, top-5 iris / top-1 pine), the
latency sweep, and the "Why it works" science cards. Colours, type and layout
come from neurobyte-web/src/index.css (@theme web2), App.tsx, AnalyticsView.tsx,
ScienceView.tsx and src/content/*.ts.

    python neurobyte-web/presentation/build_deck.py
    # -> neurobyte-web/presentation/NeuroBytes_overview.pptx
"""
import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn

# ---- light theme: tea green (index.css :root[data-theme="light"], src/theme.ts LIGHT) ----
# Variable names are the deck's roles; values are the site's light-mode tokens.
VOID     = RGBColor(0xFE, 0xFA, 0xE0)  # base            cornsilk.500
INK      = RGBColor(0xED, 0xF1, 0xD4)  # highlight-low   beige.600
INK_UP   = RGBColor(0xFF, 0xFD, 0xF3)  # surface (cards) cornsilk.800
LINE     = RGBColor(0xCC, 0xD5, 0xAE)  # highlight-med   tea_green.500 (gridlines, borders)
LINE_STR = RGBColor(0xAC, 0xBB, 0x7B)  # highlight-high  tea_green.400
TEXT     = RGBColor(0x2D, 0x33, 0x1A)  # text            tea_green.100
MIST     = RGBColor(0x5B, 0x66, 0x35)  # subtle          tea_green.200
MUTED    = RGBColor(0x5B, 0x66, 0x35)  # muted           tea_green.200
VIOLET   = RGBColor(0x5B, 0x66, 0x35)  # iris            tea_green.200 (primary accent)
BLUE     = RGBColor(0x88, 0x99, 0x4F)  # pine            tea_green.300 (top-1 bars)
CYAN     = RGBColor(0x5D, 0x51, 0x03)  # foam            cornsilk.100 (best / success)
AMBER    = RGBColor(0x53, 0x3E, 0x08)  # gold            papaya_whip.100 (emphasis)
LOVE     = RGBColor(0x64, 0x41, 0x1F)  # love            light_bronze.200 (chance line)
ROSE     = RGBColor(0x96, 0x62, 0x2E)  # rose            light_bronze.300
WHITE    = RGBColor(0xFE, 0xFA, 0xE0)  # on-iris         cornsilk.500
CARD_LINE = LINE                       # .card border: highlight-med
# light-mode brain hues (theme.ts BRAIN_HUES_LIGHT)
BRAIN = [RGBColor(0x5B, 0x66, 0x35), RGBColor(0x96, 0x62, 0x2E), RGBColor(0xA5, 0x7B, 0x10),
         RGBColor(0x79, 0x84, 0x2C), RGBColor(0x64, 0x41, 0x1F), RGBColor(0x5D, 0x51, 0x03)]

FONT = "Inter"          # the site's heading/body font
MONO = FONT     # numbers in Inter: JetBrains Mono is rarely installed and falls back to serif

EMU = 914400
SW, SH = Inches(13.333), Inches(7.5)


def E(x):  # inches -> EMU int
    return int(Inches(x))


def _no_shadow(shape):
    shape.shadow.inherit = False


def _solid(shape, color):
    shape.fill.solid(); shape.fill.fore_color.rgb = color


def _grad_lr(shape, stops):
    """Left->right linear gradient using the two default stops.
    stops: [(pos0, col0), (pos1, col1), ...]; only first and last are used."""
    shape.fill.gradient()
    gs = shape.fill.gradient_stops
    first, last = stops[0], stops[-1]
    gs[0].position = 0.0
    gs[0].color.rgb = first[1]
    gs[1].position = 1.0
    gs[1].color.rgb = last[1]
    try:
        shape.fill.gradient_angle = 0.0
    except Exception:
        pass


def rect(slide, x, y, w, h, fill=None, line=None, lw=1.0, radius=0.0, grad=None):
    shp = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE,
        Inches(x), Inches(y), Inches(w), Inches(h))
    if radius:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    if grad is not None:
        _grad_lr(shp, grad)
    elif fill is None:
        shp.fill.background()
    else:
        _solid(shp, fill)
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line; shp.line.width = Pt(lw)
    _no_shadow(shp)
    return shp


def txt(slide, x, y, w, h, paras, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
        space_after=4, ls=1.0):
    """paras: list of paragraphs; each is list of (text,size,bold,color,font)."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Pt(2); tf.margin_top = tf.margin_bottom = Pt(2)
    for i, para in enumerate(paras):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space_after); p.line_spacing = ls
        for run in para:
            t, size, bold, color = run[0], run[1], run[2], run[3]
            fnt = run[4] if len(run) > 4 else FONT
            r = p.add_run(); r.text = t
            r.font.size = Pt(size); r.font.bold = bold; r.font.name = fnt
            r.font.color.rgb = color
    return tb


def eyebrow(slide, x, y, label, color=MUTED):
    """site .eyebrow: short violet rule + wide-tracked muted caps."""
    rect(slide, x, y + 0.11, 0.42, 0.028, fill=VIOLET)
    txt(slide, x + 0.55, y, 8.0, 0.35,
        [[(label.upper(), 11.5, True, color)]])


def _chan(c):
    s = str(c)  # RGBColor -> 'RRGGBB'
    return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)


def _lerp(c1, c2, t):
    r1, g1, b1 = _chan(c1)
    r2, g2, b2 = _chan(c2)
    return RGBColor(int(r1 + (r2 - r1) * t), int(g1 + (g2 - g1) * t), int(b1 + (b2 - b1) * t))


def _ramp(t):
    """violet -> blue -> cyan ramp, t in [0,1], matching the site's .grad-word."""
    if t < 0.5:
        return _lerp(VIOLET, BLUE, t / 0.5)
    return _lerp(BLUE, CYAN, (t - 0.5) / 0.5)


def gradient_word(slide, x, y, w, h, word, size):
    """The site's violet->blue->cyan gradient word, as per-character coloured runs."""
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = False
    tf.margin_left = tf.margin_right = Pt(0); tf.margin_top = tf.margin_bottom = Pt(0)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]; p.alignment = PP_ALIGN.LEFT
    n = max(len(word) - 1, 1)
    for i, ch in enumerate(word):
        r = p.add_run(); r.text = ch
        r.font.size = Pt(size); r.font.bold = False; r.font.name = FONT
        # light mode: `.grad-word { background-image: none; color: var(--color-iris) }`
        r.font.color.rgb = VIOLET
    return tb


# ---- page scaffold ------------------------------------------------------------------
prs = Presentation(); prs.slide_width = SW; prs.slide_height = SH
BLANK = prs.slide_layouts[6]


def new_slide():
    s = prs.slides.add_slide(BLANK)
    rect(s, 0, 0, 13.333, 7.5, fill=VOID)  # void background
    # the site's light-mode washes: gold top-right, olive left (rgba over cornsilk, flattened)
    g1 = rect(s, 9.3, -1.4, 5.5, 4.0, fill=RGBColor(0xF3, 0xEB, 0xC7), radius=0.5)
    g2 = rect(s, -1.6, 1.6, 4.6, 3.6, fill=RGBColor(0xEE, 0xEB, 0xCF), radius=0.5)
    for g in (g1, g2):
        g.line.fill.background()
    return s


def card(slide, x, y, w, h, radius=0.08):
    """site .card: translucent surface, soft border, big radius."""
    return rect(slide, x, y, w, h, fill=INK_UP, line=CARD_LINE, lw=1.0, radius=radius)


def footer(slide, n):
    txt(slide, 0.6, 7.03, 9, 0.35,
        [[("NEUROBYTES · EEG-TO-VIDEO RETRIEVAL", 8.5, True, MUTED)]])
    txt(slide, 11.9, 7.03, 0.9, 0.35, [[(f"0{n}", 9, True, MIST, MONO)]],
        align=PP_ALIGN.RIGHT)


TITLE_X, TITLE_Y, TITLE_SIZE = 0.6, 0.92, 30


def title(slide, plain, accent="", x=TITLE_X, y=TITLE_Y, w=12.1, h=0.8, size=TITLE_SIZE,
          newline=False):
    """One text box: plain words in text colour + accent word in iris, so the two can
    never drift apart (light mode renders .grad-word as flat iris)."""
    if newline:
        paras = [[(plain, size, False, TEXT)], [(accent, size, False, VIOLET)]]
    else:
        paras = [[(plain + (" " if accent and not plain.endswith(" ") else ""), size, False, TEXT),
                  (accent, size, False, VIOLET)]]
    return txt(slide, x, y, w, h, paras, space_after=0, ls=0.95)


def section_head(slide, kicker, title_plain, grad=None):
    eyebrow(slide, 0.65, 0.55, kicker)
    txt(slide, 0.6, 0.92, 9.0, 0.8, [[(title_plain, 30, False, TEXT)]])
    if grad:
        # measure a rough x for the gradient word after the plain part
        tb = txt(slide, 0.6, 0.92, 0.1, 0.1, [[("", 1, False, TEXT)]])
        # place gradient word on its own line start under? Keep inline-ish:
        gradient_word(slide, grad[0], 0.95, grad[1], 0.66, grad[2], 30)


# ===================================================================== SLIDE 1 — HERO
s = new_slide()
rect(s, 0, 0, 13.333, 0.14, fill=VIOLET)
# logo mark (triangle + spark), like the header
tri = s.shapes.add_shape(MSO_SHAPE.ISOSCELES_TRIANGLE, Inches(0.65), Inches(0.5), Inches(0.42), Inches(0.42))
_solid(tri, VIOLET); tri.line.fill.background(); _no_shadow(tri)
txt(s, 1.2, 0.52, 6, 0.4, [[("NEUROBYTES", 14, True, TEXT)]])
txt(s, 9.0, 0.52, 3.7, 0.4,
    [[("HOME   ANALYSIS   WHY IT WORKS", 10.5, True, MUTED)]], align=PP_ALIGN.RIGHT)

eyebrow(s, 0.65, 1.55, "EEG-to-video retrieval")
title(s, "See what the brain", "watched.", y=1.95, w=8.8, h=2.1, size=56, newline=True)
txt(s, 0.6, 4.05, 7.4, 1.2,
    [[("From three seconds of EEG over the visual cortex, NeuroBytes ranks 102 candidate "
       "videos and finds the one the person was watching. No video is shown to the model at "
       "test time, only brain activity.", 15, False, MIST)]], ls=1.15)

stats = [("36.03%", "top-1 · chance 0.98%"), ("67.16%", "top-5 · chance 4.9%"),
         ("5.99", "mean rank of 102"), ("×37", "better than chance")]
sx, sw = 0.6, 1.95
for i, (v, l) in enumerate(stats):
    x = sx + i * (sw + 0.05)
    txt(s, x, 5.55, sw, 0.6, [[(v, 30, False, TEXT, MONO)]])
    txt(s, x, 6.15, sw, 0.6, [[(l.upper(), 9, True, MUTED)]], ls=1.0)
# primary pill + link
pill = rect(s, 9.9, 2.4, 2.8, 2.8, grad=None, fill=None)  # placeholder spacing
# brain hint (concentric dots) on the right, echoing the particle brain
import math
cx0, cy0 = 11.2, 3.3
for ring, (rad, n) in enumerate([(0.35, 8), (0.8, 14), (1.25, 20)]):
    for k in range(n):
        col = BRAIN[(k + ring * 2) % len(BRAIN)]
        a = 2 * math.pi * k / n + ring * 0.3
        dx = cx0 + rad * math.cos(a) - 0.035
        dy = cy0 + rad * math.sin(a) - 0.035
        d = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(dx), Inches(dy), Inches(0.07), Inches(0.07))
        _solid(d, col); d.line.fill.background(); _no_shadow(d)
txt(s, 0.6, 6.95, 9, 0.4,
    [[("Team NeuroBytes — Sushruth Shivaraju & Ayush Shetty", 10, True, MIST)]])
txt(s, 11.9, 7.03, 0.9, 0.35, [[("01", 9, True, MIST, MONO)]], align=PP_ALIGN.RIGHT)

# ===================================================================== SLIDE 2 — how it works
s = new_slide()
eyebrow(s, 0.65, 0.55, "How it works")
title(s, "Three seconds,", "five steps.")

STEPS = [
    ("01", "Record", "3 s of EEG while a clip plays"),
    ("02", "Window", "0–700 ms after the clip starts"),
    ("03", "Decode", "EEG → V-JEPA, CLIP, DINOv2, audio…"),
    ("04", "Fuse", "rank fusion of ~650 scorers"),
    ("05", "Balance", "CSLS hub removal + Sinkhorn"),
]
sy = 2.0
for i, (n, t, d) in enumerate(STEPS):
    txt(s, 0.65, sy + 0.04, 0.6, 0.4, [[(n, 13, False, VIOLET, MONO)]])
    txt(s, 1.25, sy - 0.04, 5.3, 0.4, [[(t, 18, False, TEXT)]])
    txt(s, 1.25, sy + 0.33, 5.3, 0.35, [[(d, 12, False, MIST)]])
    sy += 0.82

# features grid (right)
FEATURES = [
    ("EEG Moments data", "6 people, 17 channels over the back of the head, ~1,000 training clips each, 102 test clips."),
    ("~650 brain-to-video scorers", "Ridge decoders over latency windows × deep video features, plus a contrastive net across people."),
    ("Cognitive grounding", "The decodable signal sits in the first 800 ms — the classic visual ERP window."),
    ("Validated on held-out data", "Every weight is chosen on held-out training videos; the test set only measures the result."),
]
fx = [7.1, 10.2]; fy = [2.0, 4.35]
for i, (t, d) in enumerate(FEATURES):
    x = fx[i % 2]; y = fy[i // 2]
    txt(s, x, y, 2.9, 0.45, [[(t, 15, False, TEXT)]], ls=1.0)
    txt(s, x, y + 0.5, 2.9, 1.6, [[(d, 11, False, MIST)]], ls=1.12)
footer(s, 2)

# ===================================================================== SLIDE 3 — results chart
s = new_slide()
eyebrow(s, 0.65, 0.55, "Results")
title(s, "From 11% to", "36% top-1")
txt(s, 0.6, 1.62, 12, 0.5,
    [[("Official dev test, sub-01..04 (408 recordings, 102 candidates each).", 12.5, False, MIST)]])

SYSTEMS = [("J1", 11.0, 29.7, "ridge baseline"), ("T1", 11.5, 35.5, "time-resolved"),
           ("T2", 14.0, 38.7, "V-JEPA 2 + S3D"), ("WF1", 23.0, 52.5, "winfuse 509"),
           ("C3", 26.5, 57.4, "both pipelines"), ("C5", 30.9, 62.3, "+ CSLS"),
           ("C10", 32.6, 66.7, "+ align, Sinkhorn"), ("C11", 34.6, 66.2, "+ neural net"),
           ("C12", 35.1, 67.4, "+ 7 heads"), ("C13", 36.0, 67.2, "+ 2 neural nets")]
# chart card
cx, cy, cw, ch = 0.6, 2.25, 12.1, 4.5
card(s, cx, cy, cw, ch)
txt(s, cx + 0.3, cy + 0.2, 8, 0.4, [[("FROM BASELINE TO FINAL SYSTEM", 11, True, MIST)]])
# legend
rect(s, cx + 8.9, cy + 0.26, 0.18, 0.18, fill=VIOLET)
txt(s, cx + 9.15, cy + 0.17, 1.0, 0.35, [[("top-5", 11, False, MIST)]])
rect(s, cx + 10.0, cy + 0.26, 0.18, 0.18, fill=BLUE)
txt(s, cx + 10.25, cy + 0.17, 1.0, 0.35, [[("top-1", 11, False, MIST)]])
# plot area
plot_x, plot_y = cx + 0.7, cy + 0.75
plot_w, plot_h = cw - 1.1, ch - 1.5
maxv = 80.0
# gridlines
for v in (0, 20, 40, 60, 80):
    gy = plot_y + plot_h * (1 - v / maxv)
    ln = rect(s, plot_x, gy, plot_w, 0.012, fill=LINE)
    ln.line.fill.background()
    txt(s, cx + 0.05, gy - 0.14, 0.55, 0.3, [[(f"{v}%", 9, False, MIST, MONO)]], align=PP_ALIGN.RIGHT)
n = len(SYSTEMS); slot = plot_w / n
for i, (name, t1, t5, note) in enumerate(SYSTEMS):
    final = (name == "C13")
    bw = slot * 0.3
    x0 = plot_x + i * slot + slot * 0.1
    h5 = plot_h * (t5 / maxv); h1 = plot_h * (t1 / maxv)
    b5 = rect(s, x0, plot_y + plot_h - h5, bw, h5, fill=VIOLET)
    b1 = rect(s, x0 + bw + 0.04, plot_y + plot_h - h1, bw, h1, fill=BLUE)
    if not final and name != "C13":
        for b in (b5, b1):
            pass  # keep full opacity; final stands out via the outline below
    if final:
        rect(s, x0 - 0.08, plot_y - 0.05, bw * 2 + 0.24, plot_h + 0.55, line=AMBER, lw=1.5,
             fill=None, radius=0.1)
    txt(s, x0 - 0.1, plot_y + plot_h - h5 - 0.24, bw + 0.2, 0.22,
        [[(f"{t5:.0f}", 8.5, False, TEXT, MONO)]], align=PP_ALIGN.CENTER)
    txt(s, x0 + bw, plot_y + plot_h - h1 - 0.24, bw + 0.2, 0.22,
        [[(f"{t1:.0f}", 8.5, False, TEXT, MONO)]], align=PP_ALIGN.CENTER)
    txt(s, x0 - 0.15, plot_y + plot_h + 0.04, bw * 2 + 0.3, 0.25,
        [[(name, 11, bool(final), TEXT)]], align=PP_ALIGN.CENTER)
    txt(s, x0 - 0.25, plot_y + plot_h + 0.28, bw * 2 + 0.5, 0.25,
        [[(note, 7.5, False, MIST)]], align=PP_ALIGN.CENTER)
footer(s, 3)

# ===================================================================== SLIDE 4 — latency (where in time)
s = new_slide()
eyebrow(s, 0.65, 0.55, "The finding")
title(s, "The signal is the", "onset response")
txt(s, 0.6, 1.62, 12, 0.5,
    [[("Mean rank of the true video (of 102) when decoding from one latency window only. "
       "Lower is better; chance is 51.5.", 12.5, False, MIST)]], ls=1.1)

LAT = [("50–200", 35.5), ("50–400", 27.1), ("200–500", 32.9), ("480–830", 45.9),
       ("910–1260", 48.3), ("1340–1690", 52.2), ("1760–2110", 51.2),
       ("2190–2540", 51.0), ("2620–2970", 50.1)]
cx, cy, cw, ch = 0.6, 2.3, 12.1, 4.45
card(s, cx, cy, cw, ch)
plot_x, plot_y = cx + 0.7, cy + 0.4
plot_w, plot_h = cw - 1.1, ch - 1.25
maxv = 60.0
best = min(v for _, v in LAT)
for v in (0, 20, 40, 60):
    gy = plot_y + plot_h * (1 - v / maxv)
    ln = rect(s, plot_x, gy, plot_w, 0.012, fill=LINE); ln.line.fill.background()
    txt(s, cx + 0.05, gy - 0.14, 0.55, 0.3, [[(str(v), 9, False, MIST, MONO)]], align=PP_ALIGN.RIGHT)
# chance line (amber dashed-ish)
cy_line = plot_y + plot_h * (1 - 51.5 / maxv)
ch_ln = rect(s, plot_x, cy_line, plot_w, 0.02, fill=LOVE); ch_ln.line.fill.background()
txt(s, plot_x + 0.05, cy_line - 0.3, 1.6, 0.25, [[("chance 51.5", 9, True, LOVE)]])
n = len(LAT); slot = plot_w / n
for i, (w, v) in enumerate(LAT):
    bw = slot * 0.6
    x0 = plot_x + i * slot + slot * 0.2
    h = plot_h * (v / maxv)
    col = CYAN if v == best else (VIOLET if v < 40 else LINE_STR)
    rect(s, x0, plot_y + plot_h - h, bw, h, fill=col, radius=0.12)
    txt(s, x0 - 0.15, plot_y + plot_h - h - 0.24, bw + 0.3, 0.22,
        [[(f"{v:.0f}", 9, False, TEXT, MONO)]], align=PP_ALIGN.CENTER)
    txt(s, x0 - 0.2, plot_y + plot_h + 0.05, bw + 0.4, 0.25,
        [[(w, 8.5, False, MIST)]], align=PP_ALIGN.CENTER)
txt(s, cx + cw/2 - 2, cy + ch - 0.3, 4, 0.25,
    [[("window after clip onset (ms)", 10, False, MIST)]], align=PP_ALIGN.CENTER)
footer(s, 4)

# ===================================================================== SLIDE 5 — why it works cards
s = new_slide()
eyebrow(s, 0.65, 0.55, "Why it works")
title(s, "What the brain signal can and", "cannot tell us")

CARDS = [
    ("800 ms", "The decodable signal is the onset response",
     "Decoding is strongest 50–400 ms after the clip starts, the visual P1/N1 window. Every window after ~800 ms is at chance."),
    ("CSLS", "Hubs and balance",
     "CSLS removes each candidate's average pull; Sinkhorn makes every video win about once across the 102 recordings. Together ≈6 points of top-1."),
    ("6 brains", "Learning across people needs alignment",
     "The 17 channels mix brain sources differently in every head. A per-person spatial map lets each person borrow the others' recordings."),
    ("V-JEPA 2", "Deep video models speak the brain's language",
     "Self-supervised video models and mid-level CNN layers match the early visual response best. Interpretable attributes add nothing on top."),
    ("robust", "Robust, not random",
     "Same seeds rebuild bit-identically; five new seeds give 34.31 / 67.65 vs 36.03 / 67.16 — within one standard error (≈2.4 pts)."),
    ("r = 0.16", "Misses are noise, not hard videos",
     "Per-video difficulty barely correlates between people. The true video is usually rank 2–5; single-recording noise decides the last step."),
]
cw, chh = 3.87, 2.25
gx, gy, gap = 0.6, 1.85, 0.17
accent = [AMBER, CYAN, VIOLET, ROSE, LOVE, BLUE]
for i, (tag, title, body) in enumerate(CARDS):
    col = i % 3; row = i // 3
    x = gx + col * (cw + gap); y = gy + row * (chh + gap)
    card(s, x, y, cw, chh)
    rect(s, x, y, 0.1, chh, fill=accent[i])
    txt(s, x + 0.28, y + 0.18, cw - 0.5, 0.5, [[(tag, 20, False, accent[i], MONO)]])
    txt(s, x + 0.28, y + 0.72, cw - 0.5, 1.45,
        [[(title, 13.5, False, TEXT)], [(body, 10.5, False, MIST)]], space_after=6, ls=1.08)
footer(s, 5)

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NeuroBytes_overview.pptx")
prs.save(out)
print("wrote", out, "·", len(prs.slides._sldIdLst), "slides")
