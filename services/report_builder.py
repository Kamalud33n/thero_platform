"""
PDF report generation (ReportLab + matplotlib chart), isolated from the WS/
MJPEG camera pipelines. build_report_sync() is a pure sync function — it is
called via loop.run_in_executor() from routers/reports.py so it never blocks
the asyncio event loop used by streaming routes.
"""
import os
import io
import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fastapi import HTTPException

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    Image as RLImage, HRFlowable,
)

from config import (
    PDF_NAVY, PDF_GREY_BORDER, PDF_GREY_BG, PDF_GREY_TEXT,
    PDF_ROW_ALT, PDF_BODY_TEXT,
)
from database import get_db
from models import Patient, SessionModel, Report
from services.helpers import calculate_recovery_score, calculate_improvement
from services.timeutils import utcnow, to_display


def _make_progress_chart(sessions: list) -> io.BytesIO:
    plot_sessions = sessions[-10:]
    idx   = list(range(1, len(plot_sessions) + 1))
    acc   = [s.accuracy_percentage for s in plot_sessions]
    rom   = [s.average_rom for s in plot_sessions]
    dates = [to_display(s.start_time).strftime("%m/%d") for s in plot_sessions]

    fig, ax1 = plt.subplots(figsize=(6.5, 2.3), dpi=150)
    fig.patch.set_facecolor("white")
    ax1.set_facecolor("white")

    ax1.plot(idx, acc, color="#1B2A4A", linewidth=2, marker="o",
              markersize=4, label="Accuracy %")
    ax2 = ax1.twinx()
    ax2.plot(idx, rom, color="#8B93A1", linewidth=1.6, linestyle="--",
              marker="s", markersize=3.5, label="ROM °")

    ax1.set_xticks(idx)
    ax1.set_xticklabels(dates, fontsize=7, color="#5A6472")
    ax1.tick_params(axis="y", labelsize=7, colors="#5A6472")
    ax2.tick_params(axis="y", labelsize=7, colors="#5A6472")
    ax1.set_ylim(0, 100)

    for spine in ("top",):
        ax1.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)
    ax1.spines["left"].set_color("#B7BEC9")
    ax1.spines["bottom"].set_color("#B7BEC9")
    ax2.spines["right"].set_color("#B7BEC9")
    ax1.grid(axis="y", color="#EEF1F5", linewidth=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=7,
               frameon=False, ncol=2, bbox_to_anchor=(0, 1.22))

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf


def _build_recommendations(sessions, avg_acc, avg_rom, rec_score, improvement) -> list:
    """
    Rule-based (non-AI) recommendation engine. Looks at every metric already
    stored per session — not just accuracy/ROM — and returns a short,
    priority-ordered list: real problems first, general coaching next,
    positive reinforcement last. Capped so the PDF section stays readable
    instead of dumping every possible sentence.
    """
    n = len(sessions)

    def _avg(attr):
        return sum((getattr(s, attr) or 0) for s in sessions) / n

    avg_stability = _avg("stability_score")
    avg_balance   = _avg("balance_score")
    avg_smooth    = _avg("movement_smoothness")
    avg_fatigue   = _avg("fatigue_estimation")
    avg_incorrect = _avg("incorrect_movements")

    total_reps_target = sum(s.total_reps for s in sessions)
    total_reps_done   = sum(s.completed_reps for s in sessions)
    rep_completion = (total_reps_done / total_reps_target * 100) if total_reps_target else 100

    # `sessions` is already ordered oldest -> newest by the caller's query,
    # so consecutive differences give real day-gaps between visits.
    gaps = [
        (sessions[i + 1].start_time - sessions[i].start_time).days
        for i in range(len(sessions) - 1)
    ]
    avg_gap_days = (sum(gaps) / len(gaps)) if gaps else None

    priority, general, positive = [], [], []

    # --- Priority: needs attention now ---
    if avg_acc < 60:
        priority.append("Accuracy is below target — slow down repetitions and prioritize correct form over speed.")
    if avg_rom < 50:
        priority.append("Range of motion is limited — add gentle stretching/mobility work before each session.")
    if avg_fatigue > 60:
        priority.append("Fatigue levels are running high during sessions — consider shorter sets with more rest between reps.")
    if avg_incorrect > 3:
        priority.append(f"An average of {avg_incorrect:.1f} incorrect movements per session were recorded — review technique with the therapist.")
    if avg_balance < 50:
        priority.append("Balance scores are low — incorporate dedicated balance/stability drills.")
    if improvement < -5:
        priority.append(f"Performance has dipped ({improvement:+.1f}%) compared to earlier sessions — worth discussing with the therapist.")

    # --- General: moderate areas still worth coaching ---
    if 60 <= avg_acc < 75:
        general.append("Accuracy is moderate — continued focus on controlled movement should improve this further.")
    if 50 <= avg_rom < 65:
        general.append("Range of motion is improving but still limited — keep up mobility exercises.")
    if avg_smooth < 60:
        general.append("Movement smoothness could improve — practicing at a slower, steadier pace may help.")
    if 50 <= avg_stability < 65:
        general.append("Stability is improving but still developing — continue balance-focused exercises.")
    if rep_completion < 80:
        general.append(f"Only {rep_completion:.0f}% of prescribed reps were completed on average — encourage finishing full sets where possible.")
    if avg_gap_days is not None and avg_gap_days > 5:
        general.append(f"Sessions are averaging {avg_gap_days:.1f} days apart — more frequent sessions (2–3x/week) would support faster recovery.")
    if n < 5:
        general.append("Still early in the program — consistent practice over at least 10 sessions is recommended before re-evaluation.")

    # --- Positive reinforcement ---
    if avg_acc >= 85 and avg_rom >= 80:
        positive.append("Excellent progress on both accuracy and range of motion — consider introducing more advanced functional exercises.")
    elif avg_acc >= 75 and avg_rom >= 70:
        positive.append("Good overall progress — maintain the current routine and gradually increase intensity.")
    if improvement > 15:
        positive.append(f"Strong improvement trend ({improvement:+.1f}%) since the first session — keep up the momentum.")
    if rec_score >= 75:
        positive.append("Recovery score indicates strong progress toward full functional recovery.")

    # Priority issues surface first, then coaching notes, then encouragement.
    # Capped at 6 so the section stays focused rather than an unfocused wall
    # of every possible sentence.
    recs = priority + general + positive
    if not recs:
        recs = ["Continue current therapy plan.", "Regular monitoring is recommended."]
    return recs[:6]


def build_report_sync(
    patient_id: str,
    report_type: str,
    range_start: datetime.datetime | None = None,
    range_end: datetime.datetime | None = None,
) -> str:
    """Pure sync function — safe to run in executor alongside async WS loop.

    report_type controls which sessions are included:
      - "weekly"       -> sessions from the last 7 days
      - "monthly"      -> sessions from the last 30 days
      - "custom"       -> sessions between range_start and range_end (inclusive)
      - "last_session" -> only the single most recent session
      - anything else (e.g. "history") -> all sessions
    """
    with get_db() as db:
        p = db.query(Patient).filter(Patient.id == patient_id).first()
        if not p:
            raise HTTPException(404, "Patient not found")

        all_sessions = (
            db.query(SessionModel)
            .filter(SessionModel.patient_id == patient_id)
            .order_by(SessionModel.start_time)
            .all()
        )
        if not all_sessions:
            raise HTTPException(404, "No sessions found for this patient")

        # now_utc drives the actual filtering — it must stay in the same
        # timezone base (UTC) as the stored start_time values. now_display
        # is only for the human-facing label text below (see timeutils.py).
        now_utc = utcnow()
        now_display = to_display(now_utc)
        if report_type == "weekly":
            cutoff = now_utc - datetime.timedelta(days=7)
            sessions = [s for s in all_sessions if s.start_time >= cutoff]
            period_label = f"Weekly Report — last 7 days (as of {now_display.strftime('%Y-%m-%d')})"
        elif report_type == "monthly":
            cutoff = now_utc - datetime.timedelta(days=30)
            sessions = [s for s in all_sessions if s.start_time >= cutoff]
            period_label = f"Monthly Report — last 30 days (as of {now_display.strftime('%Y-%m-%d')})"
        elif report_type == "custom":
            sessions = [
                s for s in all_sessions
                if range_start <= s.start_time <= range_end
            ]
            period_label = (
                f"Custom Report — {range_start.strftime('%Y-%m-%d')} "
                f"to {range_end.strftime('%Y-%m-%d')}"
            )
        elif report_type == "last_session":
            # all_sessions is ordered oldest -> newest, so the last element
            # is the most recent session.
            sessions = all_sessions[-1:]
            period_label = f"Last Session Report — {to_display(sessions[0].start_time).strftime('%Y-%m-%d')}"
        else:
            sessions = all_sessions
            period_label = "Full History Report"

        if not sessions:
            raise HTTPException(404, "No sessions found in the selected date range")

        report_dir = f"reports/{patient_id}"
        os.makedirs(report_dir, exist_ok=True)
        # UTC is fine here — this is just for filename uniqueness, not
        # something a human reads as a wall-clock time.
        ts       = utcnow().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(report_dir, f"{patient_id}_{report_type}_{ts}.pdf")

        # 28pt margins (~20-30px range) on every side, generous internal
        # whitespace is handled via Spacers between sections below.
        doc = SimpleDocTemplate(
            filepath, pagesize=A4,
            rightMargin=28, leftMargin=28, topMargin=28, bottomMargin=28,
        )

        # Paragraph styles — Helvetica family only, navy/grey/white
        title_style = ParagraphStyle(
            "PDFTitle", fontName="Helvetica-Bold", fontSize=19,
            textColor=PDF_NAVY, alignment=TA_CENTER, leading=22,
        )
        subtitle_style = ParagraphStyle(
            "PDFSubtitle", fontName="Helvetica", fontSize=9,
            textColor=PDF_GREY_TEXT, alignment=TA_CENTER, spaceAfter=4,
        )
        section_title_style = ParagraphStyle(
            "SectionTitle", fontName="Helvetica-Bold", fontSize=11,
            textColor=PDF_NAVY, leading=14,
        )
        info_label_style = ParagraphStyle(
            "InfoLabel", fontName="Helvetica-Bold", fontSize=9,
            textColor=PDF_NAVY, leading=13,
        )
        info_value_style = ParagraphStyle(
            "InfoValue", fontName="Helvetica", fontSize=9,
            textColor=PDF_BODY_TEXT, leading=13,
        )
        body_style = ParagraphStyle(
            "Body", fontName="Helvetica", fontSize=9.5,
            textColor=PDF_BODY_TEXT, leading=15,
        )
        metric_label_style = ParagraphStyle(
            "MetricLabel", fontName="Helvetica", fontSize=7.5,
            textColor=PDF_GREY_TEXT, leading=10,
        )
        metric_value_style = ParagraphStyle(
            "MetricValue", fontName="Helvetica-Bold", fontSize=16,
            textColor=PDF_NAVY, leading=19, spaceBefore=2,
        )
        table_header_style = ParagraphStyle(
            "TableHeader", fontName="Helvetica-Bold", fontSize=8.5,
            textColor=colors.white, alignment=TA_CENTER,
        )
        table_cell_style = ParagraphStyle(
            "TableCell", fontName="Helvetica", fontSize=8.5,
            textColor=PDF_BODY_TEXT, alignment=TA_CENTER,
        )

        # Layout helpers
        def section_header(text):
            """Light-grey banded section header — the only place grey fill is used."""
            tbl = Table([[Paragraph(text, section_title_style)]], colWidths=[doc.width])
            tbl.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PDF_GREY_BG),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("LINEBELOW", (0, 0), (-1, -1), 0.75, PDF_GREY_BORDER),
            ]))
            return tbl

        def metric_box(label, value, width):
            """Bordered white box for a single summary metric."""
            inner = Table(
                [[Paragraph(label.upper(), metric_label_style)],
                 [Paragraph(str(value), metric_value_style)]],
                colWidths=[width],
            )
            inner.setStyle(TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.75, PDF_GREY_BORDER),
                ("BACKGROUND", (0, 0), (-1, -1), colors.white),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
            ]))
            return inner

        def metric_row(items, gap=8):
            """Lay N metric boxes side by side with an even gap between them."""
            n_items = len(items)
            box_w   = (doc.width - gap * (n_items - 1)) / n_items
            boxes   = [metric_box(l, v, box_w) for l, v in items]
            row = Table([boxes], colWidths=[box_w] * n_items)
            row.setStyle(TableStyle([
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-2, -1), gap),
                ("RIGHTPADDING", (-1, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            return row

        # Stats
        n           = len(sessions)
        avg_acc     = sum(s.accuracy_percentage for s in sessions) / n
        avg_rom     = sum(s.average_rom for s in sessions) / n
        total_reps  = sum(s.completed_reps for s in sessions)
        rec_score   = calculate_recovery_score(sessions)
        improvement = calculate_improvement(sessions)

        # Header
        story = [
            Paragraph("Rehabilitation AI System", title_style),
            Paragraph(period_label, subtitle_style),
            Spacer(1, 14),
        ]

        # Patient info — clean two-column key/value grid, no fills
        info_rows = [
            [Paragraph("Patient", info_label_style),     Paragraph(p.name, info_value_style),
             Paragraph("Patient ID", info_label_style),   Paragraph(p.id, info_value_style)],
            [Paragraph("Age / Gender", info_label_style), Paragraph(f"{p.age} yrs · {p.gender}", info_value_style),
             Paragraph("Consultant ID", info_label_style), Paragraph(p.mednova_consultant_id or "—", info_value_style)],
            [Paragraph("Diagnosis", info_label_style),     Paragraph(p.diagnosis or "—", info_value_style),
             Paragraph("Affected Area", info_label_style), Paragraph(p.affected_body_part or "—", info_value_style)],
        ]
        label_w = 72
        info_tbl = Table(
            info_rows,
            colWidths=[label_w, doc.width / 2 - label_w, label_w + 10, doc.width / 2 - label_w - 10],
        )
        info_tbl.setStyle(TableStyle([
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ]))
        story += [
            info_tbl,
            HRFlowable(width="100%", thickness=0.75, color=PDF_GREY_BORDER, spaceBefore=6, spaceAfter=22),
        ]

        # Summary metrics — bordered boxes
        story += [
            section_header("Summary Metrics"),
            Spacer(1, 12),
            metric_row([
                ("Total Sessions", str(n)),
                ("Avg Accuracy",   f"{avg_acc:.1f}%"),
                ("Avg ROM",        f"{avg_rom:.1f}°"),
            ]),
            Spacer(1, 8),
            metric_row([
                ("Total Reps",     str(total_reps)),
                ("Recovery Score", f"{rec_score:.1f}%"),
                ("Improvement",    f"{improvement:+.1f}%"),
            ]),
            Spacer(1, 22),
        ]

        # Progress chart (matplotlib, embedded as PNG)
        chart_buf = _make_progress_chart(sessions)
        story += [
            section_header("Progress Trend"),
            Spacer(1, 10),
            RLImage(chart_buf, width=doc.width, height=doc.width * 0.34),
            Spacer(1, 22),
        ]

        # Session history — alternating row colors
        story += [section_header("Session History (Last 10)"), Spacer(1, 10)]

        tbl_header = ["Session", "Date", "Exercise", "Accuracy", "ROM", "Reps", "Stability"]
        tbl_data = [[Paragraph(h, table_header_style) for h in tbl_header]]
        for s in sessions[-10:]:
            tbl_data.append([
                Paragraph(s.id[:8], table_cell_style),
                Paragraph(to_display(s.start_time).strftime("%Y-%m-%d"), table_cell_style),
                Paragraph(s.exercise_type[:20], table_cell_style),
                Paragraph(f"{s.accuracy_percentage:.1f}%", table_cell_style),
                Paragraph(f"{s.average_rom:.1f}°", table_cell_style),
                Paragraph(str(s.completed_reps), table_cell_style),
                Paragraph(f"{s.stability_score:.1f}" if s.stability_score else "N/A", table_cell_style),
            ])

        col_w = [doc.width*0.14, doc.width*0.16, doc.width*0.22, doc.width*0.13,
                 doc.width*0.12, doc.width*0.10, doc.width*0.13]
        tbl = Table(tbl_data, colWidths=col_w, repeatRows=1)

        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), PDF_NAVY),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("GRID", (0, 0), (-1, -1), 0.5, PDF_GREY_BORDER),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]
        for i in range(1, len(tbl_data)):
            bg = colors.white if i % 2 == 1 else PDF_ROW_ALT
            style_cmds.append(("BACKGROUND", (0, i), (-1, i), bg))
        tbl.setStyle(TableStyle(style_cmds))
        story += [tbl, Spacer(1, 24)]

        # AI Recommendations
        story += [section_header("AI Recommendations"), Spacer(1, 10)]

        recs = _build_recommendations(sessions, avg_acc, avg_rom, rec_score, improvement)

        for r in recs:
            story.append(Paragraph(f"■&nbsp;&nbsp;{r}", body_style))
            story.append(Spacer(1, 5))

        # Footer / signature
        story += [
            Spacer(1, 22),
            HRFlowable(width="100%", thickness=0.75, color=PDF_GREY_BORDER, spaceAfter=14),
            Paragraph(f"Generated: {to_display(utcnow()).strftime('%Y-%m-%d %H:%M')}", body_style),
            Spacer(1, 26),
            Paragraph("Therapist Signature: _________________________", body_style),
            Spacer(1, 10),
            Paragraph("Date: _________________________", body_style),
        ]

        doc.build(story)

        db.add(Report(patient_id=patient_id, report_type=report_type, file_path=filepath))
        db.commit()

    return filepath