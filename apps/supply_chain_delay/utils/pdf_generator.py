"""
PDF Report Generator (fpdf2)
Creates professional PDF reports for single and batch predictions.

This module is fully compatible with our new app architecture:
- Reads threshold/bands/metrics from utils.model_loader.load_metadata()
- Returns in-memory bytes for Streamlit download_button
- Supports optional SHAP-like per-feature contributions for Single reports
"""

from __future__ import annotations  # future annotations for cleaner type hints
from typing import Any, Dict, Optional, List  # typing for clarity
from datetime import datetime                 # timestamping report generation
import pandas as pd                            # tabular manipulations
from fpdf import FPDF                          # lightweight PDF generation
from utils.model_loader import load_metadata   # read final_metadata.json on demand


# =============================================================================
# Internal helpers
# =============================================================================

def _bands_from_meta(meta: Dict[str, Any]) -> Dict[str, int]:
    """Extract risk band cut points from metadata with safe defaults."""
    rb = meta.get("risk_bands", {})                                  # read bands
    return {"low_max": int(rb.get("low_max", 30)),                   # default 30
            "med_max": int(rb.get("med_max", 67))}                   # default 67


def _band_label_from_prob(prob_0_1: float, bands: Dict[str, int]) -> str:
    """Map probability (0..1) to 'Low'|'Medium'|'High' using metadata percent cutpoints."""
    pct = float(prob_0_1) * 100.0                                    # convert to percent
    if pct <= bands["low_max"]:                                       # compare to low
        return "Low"                                                  # low band
    if pct <= bands["med_max"]:                                       # compare to medium
        return "Medium"                                               # medium band
    return "High"                                                     # otherwise high


def _band_rgb(level: str) -> tuple[int, int, int]:
    """Consistent band colors (matches theme_adaptive badges)."""
    lvl = str(level).strip().lower()                                  # normalize
    if lvl == "low":
        return (46, 204, 113)                                         # green
    if lvl == "medium":
        return (243, 156, 18)                                         # orange
    return (231, 76, 60)                                              # red


def _friendly(name: str, mapping: Optional[Dict[str, str]]) -> str:
    """Return business-friendly name if provided, else echo the technical name."""
    if mapping and name in mapping:                                   # if mapping given & key exists
        return str(mapping[name])                                     # mapped label
    return str(name)                                                  # fallback: raw feature name


# =============================================================================
# PDF base class (fpdf2)
# =============================================================================

class RiskReportPDF(FPDF):
    """Minimal, clean PDF skeleton with header/footer and helper blocks."""

    def __init__(self, title: str = "Supply Chain Delay Risk Assessment"):
        super().__init__(orientation="P", unit="mm", format="A4")     # portrait A4
        self.title_text = title                                       # store title
        self.set_auto_page_break(auto=True, margin=15)                # auto page breaks with margin

    # ---- Header / Footer -----------------------------------------------------

    def header(self) -> None:
        """Top banner with title; runs on each page."""
        self.set_font("Arial", "B", 14)                               # bold title font
        self.set_text_color(44, 62, 80)                               # dark blue-gray
        self.cell(0, 10, self.title_text, border=0, ln=1, align="C")  # centered title
        self.ln(2)                                                    # small spacing

    def footer(self) -> None:
        """Bottom page number; runs on each page."""
        self.set_y(-15)                                               # 15mm from bottom
        self.set_font("Arial", "I", 8)                                # small italic font
        self.set_text_color(120, 120, 120)                            # gray
        self.cell(0, 8, f"Page {self.page_no()}", 0, 0, "C")          # centered page number

    # ---- Section helpers -----------------------------------------------------

    def section_title(self, text: str) -> None:
        """Consistent section title styling."""
        self.set_font("Arial", "B", 12)                               # bold
        self.set_text_color(52, 73, 94)                               # deep gray-blue
        self.cell(0, 8, text, border=0, ln=1, align="L")              # left-aligned
        self.ln(1)                                                    # small spacing

    def paragraph(self, text: str) -> None:
        """Body text block with automatic wrapping."""
        self.set_font("Arial", "", 10)                                # regular font
        self.set_text_color(0, 0, 0)                                  # black
        self.multi_cell(0, 5, text)                                   # wrap at page width
        self.ln(1)                                                    # spacing

    def band_badge(self, label: str, text: str) -> None:
        """Draw a full-width colored badge for the risk level."""
        r, g, b = _band_rgb(label)                                    # pick color by band
        self.set_fill_color(r, g, b)                                  # set fill color
        self.set_text_color(255, 255, 255)                            # white text
        self.set_font("Arial", "B", 12)                               # bold
        self.cell(0, 10, text, border=0, ln=1, align="C", fill=True)  # full-width cell
        self.set_text_color(0, 0, 0)                                  # restore black
        self.ln(2)                                                    # spacing

    def key_value(self, key: str, value: str) -> None:
        """Single-line key/value helper (left/right)."""
        self.set_font("Arial", "", 10)                                # regular font
        self.set_text_color(0, 0, 0)                                  # black
        self.cell(60, 6, f"{key}:", border=0, align="L")              # left key
        self.cell(0, 6, value, border=0, ln=1, align="L")             # value
        # no extra spacing to allow compact lists

    def small_table(self, rows: List[List[str]], col_widths: Optional[List[int]] = None) -> None:
        """
        Draw a simple table. Very lightweight; for short lists (e.g., top features).
        - rows: list of [col1, col2, ...]
        - col_widths: optional list of widths (mm) matching number of columns
        """
        if not rows:                                                  # nothing to draw
            return
        ncols = len(rows[0])                                          # number of columns
        usable_w = self.w - self.l_margin - self.r_margin             # page width minus margins
        if col_widths is None:                                        # if no widths given
            col_widths = [usable_w / ncols] * ncols                   # equal widths
        # Header style (first row bold)
        self.set_font("Arial", "B", 10)                               # bold header
        for i, cell in enumerate(rows[0]):
            self.cell(col_widths[i], 7, str(cell), border=1, align="L")  # header cells
        self.ln(7)                                                    # line break
        self.set_font("Arial", "", 9)                                 # body font
        for r in rows[1:]:                                            # remaining rows
            for i, cell in enumerate(r):
                txt = str(cell)
                # simple truncation if too long for the column width
                if len(txt) > 64:
                    txt = txt[:61] + "..."
                self.cell(col_widths[i], 6, txt, border=1, align="L") # body cells
            self.ln(6)                                                # line break
        self.ln(2)                                                    # spacing after table


# =============================================================================
# Public API: Single report
# =============================================================================

def generate_single_report(
    *,
    order_raw: Dict[str, Any],                         # the raw inputs a user entered
    prediction: Dict[str, Any],                        # output from predict_single(...)
    engineered_features: pd.DataFrame,                 # 1-row engineered feature vector
    shap_contributions: Optional[Dict[str, float]] = None,  # optional SHAP-like dict
    friendly_feature_names: Optional[Dict[str, str]] = None # optional mapping tech->friendly
) -> bytes:
    """
    Create a single-order PDF report.

    Returns
    -------
    bytes
        PDF file bytes suitable for Streamlit's st.download_button.
    """
    meta = load_metadata()                                              # load latest metadata
    bands = _bands_from_meta(meta)                                      # read band cutpoints
    thr = float(meta.get("optimal_threshold", 0.5))                     # operating threshold
    auc = float(meta.get("best_model_auc", 0.0))                        # key metrics
    prec = float(meta.get("best_model_precision", 0.0))
    rec = float(meta.get("best_model_recall", 0.0))
    f1 = float(meta.get("best_model_f1", 0.0))

    # Extract prob and band
    prob = float(prediction.get("score", prediction.get("probability", 0.0)))  # tolerate alt key
    band = _band_label_from_prob(prob, bands)                                   # compute band label

    # Instantiate PDF and add first page
    pdf = RiskReportPDF(title="Supply Chain Delay Risk Report")                 # create PDF
    pdf.add_page()                                                              # first page

    # --- Executive summary ----------------------------------------------------
    pdf.section_title("Executive Summary")                                      # section heading
    pdf.band_badge(band, f"RISK LEVEL: {band.upper()}  ({prob*100:.1f}%)")     # colored badge
    pdf.paragraph(                                                              # summary bullets
        "This prediction estimates the probability that this shipment will be delivered late. "
        "Use it to prioritize interventions and customer communications."
    )
    pdf.key_value("Predicted probability", f"{prob:.4f}  ({prob*100:.2f}%)")   # prob
    pdf.key_value("Operating threshold", f"{thr:.4f}  ({thr*100:.2f}%)")       # threshold
    pdf.key_value("Meets threshold", "Yes" if prob >= thr else "No")           # meets?
    pdf.key_value("Model type", str(meta.get("best_model", "LightGBM")))       # model name
    pdf.key_value("AUC-ROC", f"{auc:.4f}")                                      # metrics
    pdf.key_value("Precision / Recall / F1", f"{prec:.3f} / {rec:.3f} / {f1:.3f}")
    pdf.key_value("Report generated", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))

    pdf.ln(2)                                                                   # spacing

    # --- Order details (echo raw inputs) --------------------------------------
    pdf.section_title("Order Details (Raw Inputs)")                             # section heading
    # Render a compact list of the main raw fields the user supplied
    for k in ["order_purchase_timestamp", "estimated_delivery_date",
              "sum_price", "sum_freight", "n_items", "n_sellers",
              "payment_type", "max_installments", "mode_category",
              "customer_city", "customer_state"]:
        if k in order_raw:
            pdf.key_value(k, str(order_raw.get(k)))                             # print key/value

    # --- Top contributing factors ---------------------------------------------
    pdf.ln(3)                                                                   # spacing
    pdf.section_title("Top Contributing Factors")                               # section heading
    rows: List[List[str]] = []                                                  # table rows container

    if shap_contributions:                                                      # preferred: SHAP-like dict
        # Sort by absolute contribution and show top 10
        sorted_pairs = sorted(shap_contributions.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:10]
        rows.append(["Feature (Business Name)", "Impact (±)"])                  # header
        for name, val in sorted_pairs:
            rows.append([_friendly(name, friendly_feature_names), f"{float(val):+.3f}"])  # each row
    else:
        # Fallback: show the largest-magnitude engineered values (quick proxy)
        if isinstance(engineered_features, pd.DataFrame) and not engineered_features.empty:
            ser = engineered_features.iloc[0]                                    # first/only row
            # Only consider numerics for magnitude sorting; strings become 1/0 by non-emptiness
            proxy = []
            for name, val in ser.items():
                try:
                    mag = abs(float(val))
                except Exception:
                    mag = 1.0 if str(val).strip() else 0.0
                proxy.append((name, val, mag))
            proxy = sorted(proxy, key=lambda x: x[2], reverse=True)[:10]         # top 10 by magnitude
            rows.append(["Feature (Business Name)", "Value"])                    # header
            for name, val, _ in proxy:
                show_val = f"{float(val):.4f}" if isinstance(val, (int, float)) else str(val)
                rows.append([_friendly(name, friendly_feature_names), show_val])

    if rows:
        pdf.small_table(rows)                                                    # draw the table
    else:
        pdf.paragraph("No feature details available for this prediction.")       # fallback note

    # --- Model information -----------------------------------------------------
    pdf.section_title("Model Information")                                       # heading
    info_lines = [
        f"Model: {meta.get('best_model', 'LightGBM')}",
        f"Engineered features: {int(meta.get('n_features', 0))}",
        f"Training samples: {int(meta.get('n_samples_train', 0)):,}",
        f"Test samples: {int(meta.get('n_samples_test', 0)):,}",
        f"Training date: {meta.get('training_date', 'N/A')}",
    ]
    pdf.paragraph("\n".join(info_lines))                                         # print lines

    # --- Disclaimer ------------------------------------------------------------
    pdf.ln(2)                                                                    # spacing
    pdf.set_font("Arial", "I", 8)                                               # small italic
    pdf.set_text_color(120, 120, 120)                                           # gray text
    pdf.multi_cell(0, 4,
        "DISCLAIMER: This risk assessment is generated by a machine learning model and is intended "
        "as decision support. Actual outcomes may vary based on factors outside the model. Apply "
        "business judgment for operational decisions."
    )

    # Output bytes (fpdf2: dest='S' returns bytes)
    return pdf.output(dest="S")                                                 # return bytes
    # NOTE: Streamlit usage:
    # st.download_button("Download PDF", data=generate_single_report(...), file_name="risk_report.pdf", mime="application/pdf")


# =============================================================================
# Public API: Batch summary report
# =============================================================================

def generate_batch_summary_report(
    *,
    scored_df: pd.DataFrame,                         # output of predict_batch(...): includes score, meets_threshold, risk_band
    sample_preview_rows: int = 25                    # number of rows to include in preview table
) -> bytes:
    """
    Create a concise batch PDF summarizing a scored CSV.

    The report includes:
    - Overall counts and average score
    - Band distribution
    - Threshold summary
    - Optional preview of the first N rows (safe, small)

    Returns
    -------
    bytes
        PDF file bytes suitable for Streamlit's st.download_button.
    """
    meta = load_metadata()                                              # load metadata
    thr = float(meta.get("optimal_threshold", 0.5))                     # threshold
    bands = _bands_from_meta(meta)                                      # cutpoints

    # Defensive copies / typed columns
    df = scored_df.copy()                                               # copy so we don't mutate
    if "score" in df.columns:                                           # ensure numeric 'score'
        df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    # Summary stats
    n_rows = len(df)                                                    # total rows
    avg_score = float(df["score"].mean()) if "score" in df.columns else 0.0   # mean prob
    meets = int(df.get("meets_threshold", pd.Series([], dtype=bool)).sum()) if "meets_threshold" in df.columns else 0
    not_meets = int(n_rows - meets)                                     # count not meeting threshold

    # Band counts (if present) or derive from score
    if "risk_band" in df.columns:
        band_counts = df["risk_band"].value_counts().to_dict()          # existing mapping
    else:
        # derive labels from scores using current metadata
        band_counts = {"Low": 0, "Medium": 0, "High": 0}
        if "score" in df.columns:
            for p in df["score"]:
                band_counts[_band_label_from_prob(float(p), bands)] += 1

    # Build the PDF
    pdf = RiskReportPDF(title="Batch Prediction Summary")               # new PDF
    pdf.add_page()                                                      # first page

    # --- Overview -------------------------------------------------------------
    pdf.section_title("Overview")
    pdf.key_value("Rows scored", f"{n_rows:,}")                         # row count
    pdf.key_value("Average probability", f"{avg_score:.4f}  ({avg_score*100:.2f}%)")  # mean prob
    pdf.key_value("Operating threshold", f"{thr:.4f}  ({thr*100:.2f}%)")              # threshold
    pdf.key_value("Predictions ≥ threshold", f"{meets:,}")              # meets count
    pdf.key_value("Predictions < threshold", f"{not_meets:,}")          # not meets

    pdf.ln(1)                                                           # spacing

    # --- Band distribution ----------------------------------------------------
    pdf.section_title("Risk Band Distribution")
    # Render a small table with band counts (Low/Medium/High)
    rows = [["Band", "Count"]]
    for label in ["Low", "Medium", "High"]:
        rows.append([label, f"{int(band_counts.get(label, 0)):,}"])
    pdf.small_table(rows, col_widths=[60, 40])                          # small table

    # --- Preview table (optional) ---------------------------------------------
    if n_rows > 0:
        pdf.section_title(f"Data Preview (first {min(sample_preview_rows, n_rows)} rows)")
        # Pick a compact set of columns for preview
        show_cols = [c for c in ["order_purchase_timestamp", "estimated_delivery_date",
                                 "sum_price", "sum_freight", "n_items", "n_sellers",
                                 "payment_type", "mode_category", "customer_state",
                                 "score", "risk_band", "meets_threshold"] if c in df.columns]
        preview = df[show_cols].head(sample_preview_rows)              # slice safe preview

        # Build table rows (header + body)
        header = [str(c) for c in preview.columns]                     # header row
        body = [[str(x) for x in row] for _, row in preview.iterrows()]# body rows
        pdf.small_table([header] + body)                               # render table

    # --- Disclaimer ------------------------------------------------------------
    pdf.ln(2)
    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(120, 120, 120)
    pdf.multi_cell(0, 4,
        "NOTE: Band assignments reflect the current application metadata cut points (Low/Medium/High). "
        "Use these results to prioritize review and interventions at scale."
    )

    # Output bytes
    return pdf.output(dest="S")                                         # return bytes for download
