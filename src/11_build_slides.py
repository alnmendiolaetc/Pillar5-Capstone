"""
Pillar 5 Capstone - Step 6: Build the two slide decks
Creates one deck for a technical audience and one for a business audience.

The two decks deliberately share almost nothing. A data scientist wants to know
why PR-AUC was chosen over accuracy. An executive wants to know what it costs,
what it saves, and what could go wrong. Showing either audience the other's
slides is the usual reason good analysis fails to land.

What this file produces:
    Allen_Mendiola_Credit_Card_Fraud_Detection_Technical.pptx
    Allen_Mendiola_Credit_Card_Fraud_Detection_Business.pptx

Run after 08_bias_audit.py:
    python src/11_build_slides.py
"""

import os
import json
from datetime import date

import pandas as pd
from PIL import Image as PILImage
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN


# ===== SETTINGS =====
THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))
PROJECT_FOLDER = os.path.dirname(THIS_FOLDER)
FIGURES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "figures")
TABLES_FOLDER = os.path.join(PROJECT_FOLDER, "reports", "tables")

STUDENT_NAME = "Allen Mendiola"

# Colours used across both decks.
DARK_BLUE = RGBColor(0x1F, 0x38, 0x64)
MID_BLUE = RGBColor(0x2E, 0x54, 0x96)
RED = RGBColor(0xC4, 0x4E, 0x52)
GREEN = RGBColor(0x2E, 0x8B, 0x57)
GREY = RGBColor(0x55, 0x55, 0x55)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_WIDTH = 13.333
SLIDE_HEIGHT = 7.5


def load_json(file_name):
    path = os.path.join(TABLES_FOLDER, file_name)
    handle = open(path)
    content = json.load(handle)
    handle.close()
    return content


def load_csv(file_name):
    return pd.read_csv(os.path.join(TABLES_FOLDER, file_name))


def money(amount):
    return "$" + "{:,.0f}".format(amount)


def new_presentation():
    """Start a blank 16:9 deck."""
    presentation = Presentation()
    presentation.slide_width = Inches(SLIDE_WIDTH)
    presentation.slide_height = Inches(SLIDE_HEIGHT)
    return presentation


def blank_slide(presentation):
    """Add a completely empty slide that we place our own boxes onto."""
    return presentation.slides.add_slide(presentation.slide_layouts[6])


def add_textbox(slide, text, left, top, width, height, size=18,
                bold=False, colour=GREY, align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True

    paragraph = frame.paragraphs[0]
    paragraph.alignment = align

    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = colour
    run.font.name = "Calibri"

    return box


def add_slide_title(slide, title, subtitle=None):
    """The heading that appears at the top of every content slide."""
    add_textbox(slide, title, 0.6, 0.35, 12.2, 0.9, size=30, bold=True, colour=DARK_BLUE)

    if subtitle is not None:
        add_textbox(slide, subtitle, 0.6, 1.18, 12.2, 0.5, size=15, colour=GREY)

    # A thin rule under the heading.
    line = slide.shapes.add_shape(1, Inches(0.6), Inches(1.68), Inches(12.2), Inches(0.035))
    line.fill.solid()
    line.fill.fore_color.rgb = MID_BLUE
    line.line.fill.background()
    line.shadow.inherit = False


def add_bullets(slide, bullets, left=0.75, top=2.0, width=11.9, size=17, spacing=0.62):
    """Write a list of bullet points down the slide."""
    position = top

    for item in bullets:
        # A tuple means (text, is_a_sub_bullet)
        if isinstance(item, tuple):
            text = item[0]
            is_sub = item[1]
        else:
            text = item
            is_sub = False

        if is_sub:
            marker = "–  "
            indent = left + 0.55
            # Narrow the box by the same amount we pushed it right, otherwise
            # an indented bullet runs off the right-hand edge of the slide.
            box_width = width - 0.55
            font_size = size - 3
            colour = GREY
        else:
            marker = "•  "
            indent = left
            box_width = width
            font_size = size
            colour = RGBColor(0x22, 0x22, 0x22)

        add_textbox(slide, marker + text, indent, position, box_width, 0.55,
                    size=font_size, colour=colour)
        position = position + spacing

    return position


def add_picture(slide, file_name, left, top, width, bottom_limit=6.3):
    """
    Put a chart on a slide, scaled to the width given.

    Charts come in different shapes, and a tall one scaled to a fixed width can
    run straight off the bottom of the slide. So we work out how tall it would
    be, and if that is too tall we scale it down to fit instead.
    """
    path = os.path.join(FIGURES_FOLDER, file_name)
    if not os.path.exists(path):
        print("  WARNING: missing figure", file_name)
        return

    picture_file = PILImage.open(path)
    shape_ratio = picture_file.height / float(picture_file.width)
    picture_file.close()

    height = width * shape_ratio
    room_available = bottom_limit - top

    if height > room_available:
        height = room_available
        width = height / shape_ratio

    slide.shapes.add_picture(path, Inches(left), Inches(top),
                             width=Inches(width), height=Inches(height))


def add_stat_box(slide, value, label, left, top, width=2.9, colour=MID_BLUE):
    """A large number with a caption under it, used for headline results."""
    panel = slide.shapes.add_shape(1, Inches(left), Inches(top), Inches(width), Inches(1.75))
    panel.fill.solid()
    panel.fill.fore_color.rgb = colour
    panel.line.fill.background()
    panel.shadow.inherit = False

    frame = panel.text_frame
    frame.word_wrap = True
    frame.margin_top = Inches(0.12)

    first = frame.paragraphs[0]
    first.alignment = PP_ALIGN.CENTER
    run = first.add_run()
    run.text = value
    run.font.size = Pt(34)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"

    second = frame.add_paragraph()
    second.alignment = PP_ALIGN.CENTER
    run = second.add_run()
    run.text = label
    run.font.size = Pt(12.5)
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"


def add_table(slide, frame, left, top, width, height, font_size=12):
    """Draw a pandas table onto a slide."""
    rows = len(frame) + 1
    columns = len(frame.columns)

    shape = slide.shapes.add_table(rows, columns, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    table = shape.table

    # Header row.
    # The formatting has to be set on the RUN, not just the paragraph. A
    # paragraph-level colour is only a default that PowerPoint may override with
    # the table's own theme - which would leave dark text on a dark blue header.
    for column_number in range(columns):
        cell = table.cell(0, column_number)
        cell.text = str(frame.columns[column_number])

        for run in cell.text_frame.paragraphs[0].runs:
            run.font.size = Pt(font_size)
            run.font.bold = True
            run.font.color.rgb = WHITE
            run.font.name = "Calibri"

        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_BLUE

    # Body rows.
    for row_number in range(len(frame)):
        for column_number in range(columns):
            cell = table.cell(row_number + 1, column_number)
            value = frame.iloc[row_number, column_number]

            if isinstance(value, float):
                # "%g" turns a big number into 6.018e+05, which nobody wants to
                # read on a slide. Large values get thousands separators instead.
                if abs(value) >= 1000:
                    cell.text = "{:,.0f}".format(value)
                else:
                    cell.text = "%.4g" % value
            else:
                cell.text = str(value)

            for run in cell.text_frame.paragraphs[0].runs:
                run.font.size = Pt(font_size - 0.5)
                run.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
                run.font.name = "Calibri"

            cell.fill.solid()
            if row_number % 2 == 0:
                cell.fill.fore_color.rgb = RGBColor(0xF2, 0xF5, 0xFA)
            else:
                cell.fill.fore_color.rgb = WHITE

    return table


def add_takeaway(slide, text, top=6.45, colour=GREEN):
    """A coloured strip along the bottom holding the one-line message."""
    strip = slide.shapes.add_shape(1, Inches(0.6), Inches(top), Inches(12.2), Inches(0.62))
    strip.fill.solid()
    strip.fill.fore_color.rgb = colour
    strip.line.fill.background()
    strip.shadow.inherit = False

    frame = strip.text_frame
    frame.word_wrap = True
    paragraph = frame.paragraphs[0]
    paragraph.alignment = PP_ALIGN.CENTER
    run = paragraph.add_run()
    run.text = text
    run.font.size = Pt(14.5)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"


def add_title_slide(presentation, title, subtitle, audience):
    """The opening slide of a deck."""
    slide = blank_slide(presentation)

    banner = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(SLIDE_WIDTH), Inches(2.9))
    banner.fill.solid()
    banner.fill.fore_color.rgb = DARK_BLUE
    banner.line.fill.background()
    banner.shadow.inherit = False

    frame = banner.text_frame
    frame.word_wrap = True
    frame.margin_left = Inches(0.8)
    frame.margin_top = Inches(0.75)

    first = frame.paragraphs[0]
    run = first.add_run()
    run.text = title
    run.font.size = Pt(40)
    run.font.bold = True
    run.font.color.rgb = WHITE
    run.font.name = "Calibri"

    second = frame.add_paragraph()
    run = second.add_run()
    run.text = subtitle
    run.font.size = Pt(19)
    run.font.color.rgb = RGBColor(0xC8, 0xD4, 0xE8)
    run.font.name = "Calibri"

    add_textbox(slide, audience, 0.85, 3.35, 11.5, 0.5, size=17, bold=True, colour=RED)
    add_textbox(slide, STUDENT_NAME, 0.85, 4.15, 11.5, 0.5, size=19, bold=True, colour=DARK_BLUE)
    add_textbox(slide, "Pillar 5 Capstone  |  Finance Domain  |  " + date.today().strftime("%d %B %Y"),
                0.85, 4.75, 11.5, 0.5, size=14, colour=GREY)

    return slide


# ===== Load every result =====
print("=" * 60)
print("STEP 6: BUILDING THE SLIDE DECKS")
print("=" * 60)

overview = load_json("data_overview.json")
eda = load_json("eda_findings.json")
features = load_json("feature_list.json")
models = load_json("model_results.json")
clusters = load_json("clustering_results.json")
explain = load_json("explainability_results.json")
fairness = load_json("fairness_results.json")

model_comparison = load_csv("model_comparison.csv")
imbalance_comparison = load_csv("imbalance_comparison.csv")
gender_bias = load_csv("bias_by_gender.csv")
age_bias = load_csv("bias_by_age.csv")
sensitivity = load_csv("cost_sensitivity.csv")

best = models["at_best_threshold"]
mitigation = fairness["mitigation"]
age_gaps = fairness["age_band"]
gender_gaps = fairness["gender"]
city_gaps = fairness["city_size"]

print("Loaded all result files.")


# ==========================================================================
#  DECK 1 - TECHNICAL
# ==========================================================================
print()
print("--- Building the technical deck ---")

deck = new_presentation()

# --- Slide 1: title
add_title_slide(deck,
                "Detecting Fraudulent Card Transactions",
                "An end-to-end machine learning lifecycle, from framing to fairness",
                "TECHNICAL DECK  |  For data scientists and engineers")

# --- Slide 2: the problem
slide = blank_slide(deck)
add_slide_title(slide, "The problem and why it is hard",
                "Binary classification on %s transactions" % "{:,}".format(overview["n_rows"]))
add_bullets(slide, [
    "Goal: decide whether a card payment is fraudulent, at the moment it is attempted.",
    "Supervised binary classification. Target: is_fraud.",
    "Only %s of %s transactions are fraud - %.3f%% of the data."
    % ("{:,}".format(overview["n_fraud"]), "{:,}".format(overview["n_rows"]),
       overview["fraud_rate_percent"]),
    ("One fraud for roughly every %.0f genuine payments." % models["imbalance_ratio"], True),
    "This rarity drives every decision that follows.",
    ("Accuracy becomes useless - predicting 'genuine' always scores %.2f%%."
     % ((1 - models["random_baseline_pr_auc"]) * 100), True),
    ("PR-AUC and recall are the honest metrics here.", True),
])
add_takeaway(slide, "Headline metric: PR-AUC. Random baseline is %.4f, so there is a long way to climb."
             % models["random_baseline_pr_auc"], colour=MID_BLUE)

# --- Slide 3: data and quality
slide = blank_slide(deck)
add_slide_title(slide, "The data, and two problems found in it",
                "Sparkov simulated transactions, %s to %s"
                % (overview["first_transaction"][:10], overview["last_transaction"][:10]))
add_bullets(slide, [
    "Chosen because it keeps gender, date of birth, job and city size.",
    ("The better-known ULB dataset is anonymised to V1-V28, which would make both SHAP "
     "and the fairness audit meaningless.", True),
    "Problem 1: the file has exactly %s rows - Excel's row limit."
    % "{:,}".format(overview["n_rows"]),
    ("Roughly 248,000 transactions were lost before publication.", True),
    "Problem 2: birth dates used 2-digit years, so '1/19/62' was read as 2062.",
    ("%s rows (%.1f%%) produced customers born in the future, with NEGATIVE ages."
     % ("{:,}".format(overview.get("dob_wrong_century_rows", 0)),
        overview.get("dob_wrong_century_percent", 0)), True),
    ("Caught only because a cluster reported an average age of minus 41.", True),
])
add_takeaway(slide, "The age bug would have silently corrupted the age fairness audit while appearing to work.",
             colour=RED)

# --- Slide 4: features
slide = blank_slide(deck)
add_slide_title(slide, "Feature engineering", "13 engineered features from 22 raw columns")
add_bullets(slide, [
    "amt_vs_card_avg - payment size against what that card normally spends.",
    "distance_km - haversine distance from the customer's home to the shop.",
    "is_night, hour - fraud rate at night is %.2f%% against %.2f%% by day."
    % (eda["night_fraud_rate_percent"], eda["day_fraud_rate_percent"]),
    "age - computed from date of birth, after the century fix.",
    "Leakage control: card averages and merchant frequencies are fitted on TRAIN only.",
    "Split is chronological, not random - train on the past, test on the future.",
    "Dropped: name, street, card number, postcode (privacy) and gender, job (protected).",
])
add_takeaway(slide, "A random split would let the model learn from the future. The split is by date for that reason.",
             colour=MID_BLUE)

# --- Slide 5: EDA
slide = blank_slide(deck)
add_slide_title(slide, "What the data shows", "The strongest single pattern is the time of day")
add_picture(slide, "fig03_fraud_by_hour.png", 0.7, 2.0, 6.0)
add_picture(slide, "fig02_amount_distribution.png", 6.9, 2.0, 5.9)
add_takeaway(slide, "Fraud is %.0fx more likely at night, and averages %s against %s for genuine payments."
             % (eda["night_fraud_rate_percent"] / eda["day_fraud_rate_percent"],
                money(eda["average_fraud_amount"]), money(eda["average_genuine_amount"])))

# --- Slide 6: model comparison, and which one was actually deployed
slide = blank_slide(deck)
add_slide_title(slide, "Seven models compared",
                "All measured on the held-out later time period - and the top scorer is not the one deployed")
compact = model_comparison[["model", "precision", "recall", "f1", "roc_auc", "pr_auc"]].copy()
compact.columns = ["Model", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
add_table(slide, compact, 0.75, 1.95, 11.8, 3.15, font_size=11.5)
add_bullets(slide, [
    "ROC-AUC flatters everything here; PR-AUC separates the models properly.",
    "Highest PR-AUC: %s (%.4f).  Deployed: %s (%.4f)."
    % (models["highest_pr_auc_model"], models["highest_pr_auc"],
       models["best_model"], models["best_pr_auc"]),
    "That %.4f gap is noise. Tie broken on explainability - SHAP is exact on trees."
    % (models["highest_pr_auc"] - models["best_pr_auc"]),
], top=5.2, size=14, spacing=0.4)
add_takeaway(slide, "A bank that cannot explain a declined payment cannot deploy the model, whatever its score.",
             colour=RED)

# --- Slide 8: imbalance finding
slide = blank_slide(deck)
add_slide_title(slide, "A result that contradicts the textbook",
                "Class weights and SMOTE both LOWERED PR-AUC")
weights_table = imbalance_comparison.copy()
weights_table.columns = ["Approach", "Recall", "Precision", "PR-AUC"]
add_table(slide, weights_table, 0.75, 2.05, 7.2, 1.9, font_size=13)
add_bullets(slide, [
    "Recall leaps from %.2f to %.2f, exactly as advertised."
    % (imbalance_comparison["recall"].iloc[0], imbalance_comparison["recall"].iloc[1]),
    "But PR-AUC falls, because ranking quality did not improve.",
    "These methods move the decision boundary; they do not make the model wiser.",
    "PR-AUC looks across all thresholds, so it sees through the trade.",
], top=4.3, size=16)
add_takeaway(slide, "Better approach: train on the real distribution, then tune the threshold deliberately.",
             colour=RED)

# --- Slide 9: threshold, and how much the answer depends on our assumptions
slide = blank_slide(deck)
add_slide_title(slide, "Choosing the threshold with money",
                "0.5 is a convention - and the 'best' answer moves with what we assume")
add_picture(slide, "fig33_cost_sensitivity.png", 0.7, 2.0, 5.9)
add_bullets(slide, [
    "Each alert costs %s of analyst time; each FALSE alert costs a further %s in goodwill."
    % (money(models["cost_per_review"]), money(models["cost_per_upset_customer"])),
    "Best threshold: %.2f" % best["threshold"],
    ("%s alerts, %.0f%% of them real fraud" % ("{:,}".format(best["alerts"]),
                                               best["precision"] * 100), True),
    ("%s of %s frauds caught (%.1f%%)"
     % ("{:,}".format(best["fraud_caught"]),
        "{:,}".format(best["fraud_caught"] + best["fraud_missed"]),
        best["recall"] * 100), True),
    ("Net benefit %s" % money(best["net_benefit"]), True),
    "As the assumed goodwill cost rises $0 to $250, the best threshold moves %.2f to %.2f."
    % (sensitivity["best_threshold"].iloc[0], sensitivity["best_threshold"].iloc[-1]),
    "An earlier version searched from 0.05 and 'won' at 0.05 - the range was too narrow.",
], left=6.9, top=2.0, width=6.0, size=14, spacing=0.52)
add_takeaway(slide, "This is a business judgement wearing a mathematical costume. Say so out loud.",
             colour=RED)

# --- Slide 10: explainability
slide = blank_slide(deck)
add_slide_title(slide, "Explaining the model", "SHAP, LIME, permutation importance and mutual information")
add_picture(slide, "fig24_shap_bar.png", 0.7, 1.95, 6.0)
add_bullets(slide, [
    "Top features by SHAP:",
] + [("%s" % name, True) for name in explain["top_5_shap_features"][:5]] + [
    "All behavioural - what was done, not who did it.",
    "PCA needs %d of %d components for 95%% of the variance..."
    % (explain["pca_components_for_95_percent"], explain["total_features"]),
    ("...but PCA is rejected for the final model: it destroys the explanations.", True),
], left=7.1, top=2.1, width=5.7, size=15, spacing=0.5)
add_takeaway(slide, "'40x this card's normal spend, at 2am' is actionable. 'Component 3 was high' is not.",
             colour=MID_BLUE)

# --- Slide 11: bias audit and the mitigation together
slide = blank_slide(deck)
add_slide_title(slide, "Fairness audit, and what fixing it costs",
                "Gender and job were withheld from the model. Was that enough?")
add_picture(slide, "fig32_mitigation_effect.png", 0.65, 2.0, 6.1)
add_bullets(slide, [
    "Gender disparate impact %.3f (%s); age %.3f and city size %.3f both FAIL the 80%% rule."
    % (gender_gaps["disparate_impact_ratio"],
       "passes" if gender_gaps["passes_80_percent_rule"] else "fails",
       age_gaps["disparate_impact_ratio"], city_gaps["disparate_impact_ratio"]),
    "Worst served: %s customers, only %.1f%% of their fraud caught."
    % (city_gaps["least_protected_group"], city_gaps["least_protected_tpr"]),
    "Protected traits leave fingerprints on category, city size and timing.",
    "Fix: a separate threshold per age group.",
    ("gap %.1f -> %.1f points" % (mitigation["gap_before"], mitigation["gap_after"]), True),
    ("%d more frauds caught..." % (mitigation["fraud_caught_after"]
                                   - mitigation["fraud_caught_before"]), True),
    ("...but %s more false alarms to review"
     % "{:,}".format(mitigation["false_alarms_after"]
                     - mitigation["false_alarms_before"]), True),
    "Also considered: reweighting, augmentation, post-processing.",
], left=6.95, top=2.0, width=5.9, size=14, spacing=0.5)
add_takeaway(slide, "Removing a sensitive column removes your ability to SEE its effect, not the effect itself.",
             colour=RED)

# --- Slide 12: unsupervised
slide = blank_slide(deck)
add_slide_title(slide, "Finding fraud without any labels",
                "Because in a real bank the labels arrive weeks late")
add_picture(slide, "fig17_cluster_fraud_rates.png", 0.7, 1.95, 6.1)
add_bullets(slide, [
    "K-Means: best k = %d (silhouette %.3f)" % (clusters["best_k"], clusters["best_silhouette"]),
    ("Riskiest cluster has a %.1f%% fraud rate - %.0fx the average"
     % (clusters["riskiest_cluster_fraud_rate"], clusters.get("riskiest_cluster_lift", 0)), True),
    "DBSCAN: %s noise points (%.1f%%)" % ("{:,}".format(clusters["dbscan_noise_count"]),
                                          clusters["dbscan_noise_percent"]),
    ("Noise is %.0fx more likely to be fraud than grouped points"
     % clusters.get("dbscan_noise_lift", 0), True),
    "Hierarchical silhouette %.3f, agrees with K-Means but far slower."
    % clusters["hierarchical_silhouette"],
], left=7.2, top=2.1, width=5.6, size=15, spacing=0.57)
add_takeaway(slide, "'Does not look like anything else' is a powerful fraud signal on its own.")

# --- Slide 13: limitations
slide = blank_slide(deck)
add_slide_title(slide, "Limitations and what comes next")
add_bullets(slide, [
    "The data is SIMULATED. Real fraud adapts; these numbers will not transfer.",
    "Source file truncated at Excel's row limit, losing ~248,000 transactions.",
    "No race or ethnicity field exists, so that fairness dimension is untested.",
    "City size is a weak stand-in for socioeconomic status.",
    "Next: model each card as a SEQUENCE and apply an LSTM - the formulation that "
    "would genuinely justify deep learning, unlike the flat table used here.",
    "Next: velocity features (payments per hour/day/week), the biggest missing signal.",
    "Next: adversarial testing against a fraudster who knows the model exists.",
], top=1.95, spacing=0.6)
add_takeaway(slide, "Methodology transfers. The specific numbers do not. Validate on real data before trusting any of it.",
             colour=RED)

technical_path = os.path.join(PROJECT_FOLDER,
                              "Allen_Mendiola_Credit_Card_Fraud_Detection_Technical.pptx")
deck.core_properties.author = STUDENT_NAME
deck.core_properties.last_modified_by = STUDENT_NAME
deck.core_properties.title = "Detecting Fraudulent Card Transactions - Technical"
deck.save(technical_path)
print("Saved: %s (%d slides)" % (os.path.basename(technical_path), len(deck.slides._sldIdLst)))


# ==========================================================================
#  DECK 2 - BUSINESS
# ==========================================================================
print()
print("--- Building the business deck ---")

deck = new_presentation()

# --- Slide 1: title
add_title_slide(deck,
                "Stopping Card Fraud Before It Costs Us",
                "What we can catch, what it saves, and what we must decide",
                "BUSINESS DECK  |  For executives and risk owners")

# --- Slide 2: the problem
slide = blank_slide(deck)
add_slide_title(slide, "The problem in plain terms")
add_bullets(slide, [
    "When a stolen card is used, we refund the customer and absorb the loss.",
    "Fraud is rare - about %.1f in every 1,000 payments - which makes it hard to spot."
    % (overview["fraud_rate_percent"] * 10),
    "But we cannot simply block anything suspicious:",
    ("every block costs staff time to review", True),
    ("and a wrongly blocked customer is a customer standing at a till, embarrassed", True),
    "So this is a balancing act, not a detection problem.",
    "In the period we tested, fraud was worth %s." % money(overview["total_fraud_amount"]),
])
add_takeaway(slide, "The question is not 'can we catch fraud'. It is 'how much are we willing to pay to catch it'.",
             colour=MID_BLUE)

# --- Slide 3: headline results
slide = blank_slide(deck)
add_slide_title(slide, "What the system delivers", "Measured on transactions it had never seen")
add_stat_box(slide, "%.0f%%" % (best["recall"] * 100), "of fraud caught", 0.75, 2.2, 2.9, GREEN)
add_stat_box(slide, "%.0f%%" % (best["precision"] * 100), "of alerts are real fraud", 3.9, 2.2, 2.9, MID_BLUE)
add_stat_box(slide, money(best["net_benefit"]), "net benefit in the period", 7.05, 2.2, 2.9, DARK_BLUE)
add_stat_box(slide, "{:,}".format(best["alerts"]), "alerts to review", 10.2, 2.2, 2.4, GREY)
add_bullets(slide, [
    "Out of %s frauds, the system caught %s and missed %s."
    % ("{:,}".format(best["fraud_caught"] + best["fraud_missed"]),
       "{:,}".format(best["fraud_caught"]), "{:,}".format(best["fraud_missed"])),
    "Fraud value blocked: %s. Cost of reviews and customer goodwill: %s."
    % (money(best["money_saved"]), money(best["review_cost"] + best["customer_cost"])),
], top=4.4, size=17)
add_takeaway(slide, "Roughly %s of value protected for every $1 spent running it."
             % money(best["money_saved"] / max(best["review_cost"] + best["customer_cost"], 1)))

# --- Slide 4: the dial
slide = blank_slide(deck)
add_slide_title(slide, "There is a dial, and someone must set it",
                "Catch more fraud, or annoy fewer customers. Not both.")
add_picture(slide, "fig33_cost_sensitivity.png", 1.4, 2.0, 6.4)
add_bullets(slide, [
    "Turn the dial one way:",
    ("catch nearly all fraud, but block far more innocent customers", True),
    "Turn it the other way:",
    ("almost every alert is genuine, but more fraud slips through", True),
    "Where it should sit depends on what we think a wrongly blocked customer costs us.",
    "That is a judgement for this room, not for the data team.",
], left=8.0, top=2.2, width=4.9, size=15, spacing=0.56)
add_takeaway(slide, "Our recommendation assumes a wrongly blocked customer costs %s. Challenge that number."
             % money(models["cost_per_upset_customer"]), colour=RED)

# --- Slide 5: staffing option
slide = blank_slide(deck)
add_slide_title(slide, "Two ways to run it")

option_rows = [["", "Maximum value", "Within current team capacity"]]
capacity = models.get("capacity_option")
if capacity is not None:
    option_rows.append(["Alerts to review", "{:,}".format(best["alerts"]),
                        "{:,}".format(capacity["alerts"])])
    option_rows.append(["Fraud caught", "%s (%.0f%%)" % ("{:,}".format(best["fraud_caught"]),
                                                         best["recall"] * 100),
                        "%s (%.0f%%)" % ("{:,}".format(capacity["fraud_caught"]),
                                         capacity["recall"] * 100)])
    option_rows.append(["Alerts that are real fraud", "%.0f%%" % (best["precision"] * 100),
                        "%.0f%%" % (capacity["precision"] * 100)])
    option_rows.append(["Net benefit", money(best["net_benefit"]), money(capacity["net_benefit"])])

options_table = pd.DataFrame(option_rows[1:], columns=option_rows[0])
add_table(slide, options_table, 1.2, 2.2, 10.9, 2.3, font_size=14)

add_bullets(slide, [
    "The capacity option catches less fraud but needs no extra headcount.",
    "The difference between them is a staffing decision, not a technical one.",
], top=4.9, size=17)
add_takeaway(slide, "Both options are profitable. Choose based on how many reviewers we are willing to fund.",
             colour=MID_BLUE)

# --- Slide 6: fairness risk
slide = blank_slide(deck)
add_slide_title(slide, "A risk we must not discover later",
                "The system does not protect every customer group equally")
add_bullets(slide, [
    "We deliberately never told the model anyone's gender or job.",
    "That is necessary, but it turns out not to be sufficient - the model infers them "
    "indirectly from spending patterns. What we measured:",
    ("%s customers have %.0f%% of their fraud caught, against %.0f%% for %s"
     % (gender_gaps["least_protected_group"], gender_gaps["least_protected_tpr"],
        gender_gaps["best_protected_tpr"], gender_gaps["best_protected_group"]), True),
    ("%s customers are the worst protected of all, at %.0f%%"
     % (city_gaps["least_protected_group"], city_gaps["least_protected_tpr"]), True),
    ("on the standard 80%% legal test, city size scores %.2f and age %.2f - both fail"
     % (city_gaps["disparate_impact_ratio"], age_gaps["disparate_impact_ratio"]), True),
    "We can close most of the age gap, at the cost of %s extra reviews."
    % "{:,}".format(mitigation["false_alarms_after"] - mitigation["false_alarms_before"]),
])
add_takeaway(slide, "Regulators increasingly ask for exactly this evidence. We have it, before being asked.",
             colour=RED)

# --- Slide 7: what could go wrong
slide = blank_slide(deck)
add_slide_title(slide, "What could go wrong")
add_bullets(slide, [
    "These results come from SIMULATED data. Real performance will be lower.",
    ("Nothing should go live before a shadow run against real transactions.", True),
    "Fraud adapts. A model trained on last year decays - budget for monthly retraining.",
    "Alerts must go to people, not straight to automatic blocking.",
    "Customers need a route to challenge a block and have it reversed.",
    "Watch for the feedback loop: flagging one group more creates more evidence",
    ("against that group, which teaches the next model to flag them even harder.", True),
])
add_takeaway(slide, "Monitor fairness alongside accuracy from day one, not after the first complaint.", colour=MID_BLUE)

# --- Slide 8: the ask
slide = blank_slide(deck)
add_slide_title(slide, "What we are asking you to decide")
add_bullets(slide, [
    "1.  Approve a shadow run against real transactions, scoring but not blocking.",
    "2.  Set the dial: confirm what a wrongly blocked customer is worth to us.",
    "3.  Choose the staffing level - maximum value, or current team capacity.",
    "4.  Decide how much we will pay to close the fairness gap, and record that decision.",
    "5.  Fund monthly retraining as an operating cost, not a project.",
])
add_stat_box(slide, money(best["net_benefit"]), "estimated benefit per test period", 1.6, 5.05, 4.4, GREEN)
add_stat_box(slide, "%.0f%%" % (best["recall"] * 100), "of fraud stopped", 6.6, 5.05, 4.4, DARK_BLUE)
add_takeaway(slide, "The technology works. The remaining decisions are ours, and they are about values and money.",
             top=6.6, colour=MID_BLUE)

business_path = os.path.join(PROJECT_FOLDER,
                             "Allen_Mendiola_Credit_Card_Fraud_Detection_Business.pptx")
deck.core_properties.author = STUDENT_NAME
deck.core_properties.last_modified_by = STUDENT_NAME
deck.core_properties.title = "Stopping Card Fraud Before It Costs Us - Business"
deck.save(business_path)
print("Saved: %s (%d slides)" % (os.path.basename(business_path), len(deck.slides._sldIdLst)))

print()
print("Both decks built.")
print()
print("Done. That is every deliverable.")
