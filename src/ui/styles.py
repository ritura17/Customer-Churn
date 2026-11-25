"""Cohesive SaaS design system and custom CSS for Customer Churn Intelligence."""

import streamlit as st

PALETTE = {
    "primary": "#2563EB",
    "primary_dark": "#1D4ED8",
    "slate_900": "#0F172A",
    "slate_700": "#334155",
    "slate_500": "#64748B",
    "border": "#E2E8F0",
    "bg_canvas": "#F8FAFC",
    "card_bg": "#FFFFFF",
    "retain": "#059669",
    "retain_bg": "#ECFDF5",
    "medium": "#D97706",
    "medium_bg": "#FFFBEB",
    "churn": "#DC2626",
    "churn_bg": "#FEF2F2",
}


def inject_custom_css() -> None:
    """Inject clean SaaS styling, typography, card containers, and sidebar polish."""
    st.markdown(
        """
        <style>
        /* Main canvas & typography */
        .stApp {
            background-color: #F8FAFC;
            color: #0F172A;
        }
        [data-testid="stMainBlockContainer"],
        .block-container {
            max-width: 1400px !important;
            width: 100% !important;
            padding-top: 1.75rem !important;
            padding-bottom: 2.5rem !important;
            padding-left: 2.25rem !important;
            padding-right: 2.25rem !important;
        }
        h1, h2, h3 {
            color: #0F172A;
            letter-spacing: -0.02em;
        }

        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background-color: #0F172A;
            border-right: 1px solid #1E293B;
        }
        section[data-testid="stSidebar"] * {
            color: #E2E8F0;
        }
        section[data-testid="stSidebar"] .stRadio > div {
            gap: 0.35rem;
        }
        section[data-testid="stSidebar"] .stRadio label {
            background-color: transparent;
            border-radius: 8px;
            padding: 0.5rem 0.75rem;
            transition: background-color 0.15s ease;
            border: 1px solid transparent;
        }
        section[data-testid="stSidebar"] .stRadio label:hover {
            background-color: #1E293B;
        }
        section[data-testid="stSidebar"] .stRadio label:has(input:checked) {
            background-color: #2563EB;
            color: #FFFFFF !important;
            font-weight: 600;
        }

        /* Brand block in sidebar */
        .brand-box {
            padding: 0.75rem 0.75rem 1rem 0.75rem;
            border-bottom: 1px solid #1E293B;
            margin-bottom: 1rem;
        }
        .brand-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #FFFFFF !important;
            margin: 0;
            line-height: 1.3;
        }
        .brand-subtitle {
            font-size: 0.75rem;
            color: #94A3B8 !important;
            margin-top: 0.2rem;
        }

        /* Page header */
        .page-header {
            margin-bottom: 1.25rem;
        }
        .page-title {
            font-size: 1.75rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
        }
        .page-subtitle {
            font-size: 0.95rem;
            color: #64748B;
            margin-top: 0.25rem;
        }

        /* KPI Cards */
        .kpi-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 0.95rem 1.05rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
            min-height: 132px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        .kpi-label {
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: #64748B;
            margin-bottom: 0.3rem;
            white-space: nowrap;
        }
        .kpi-value {
            font-size: 1.45rem;
            font-weight: 700;
            color: #0F172A;
            line-height: 1.2;
            white-space: nowrap;
        }
        .kpi-sub {
            font-size: 0.76rem;
            color: #64748B;
            margin-top: 0.25rem;
        }

        /* Section Form Cards */
        .section-card-header {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-bottom: 2px solid #2563EB;
            border-radius: 10px 10px 0 0;
            padding: 0.75rem 1rem;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }
        .section-card-title {
            font-size: 1rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0;
        }
        .section-card-desc {
            font-size: 0.82rem;
            color: #64748B;
            margin: 0.15rem 0 0 0;
        }

        /* Prediction Result Banners */
        .result-card-churn {
            background: #FEF2F2;
            border: 1px solid #FECACA;
            border-left: 5px solid #DC2626;
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .result-card-stay {
            background: #ECFDF5;
            border: 1px solid #A7F3D0;
            border-left: 5px solid #059669;
            border-radius: 12px;
            padding: 1.25rem 1.4rem;
            margin-bottom: 1rem;
        }
        .result-badge-churn {
            display: inline-block;
            background: #DC2626;
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
        }
        .result-badge-stay {
            display: inline-block;
            background: #059669;
            color: #FFFFFF;
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            padding: 0.2rem 0.6rem;
            border-radius: 999px;
            margin-bottom: 0.5rem;
        }
        .result-heading {
            font-size: 1.35rem;
            font-weight: 700;
            color: #0F172A;
            margin: 0 0 0.25rem 0;
        }
        .result-copy {
            font-size: 0.92rem;
            color: #334155;
            margin: 0;
        }

        /* Risk Pills */
        .risk-pill {
            display: inline-block;
            padding: 0.22rem 0.65rem;
            border-radius: 999px;
            font-size: 0.8rem;
            font-weight: 600;
        }
        .risk-high { background: #FEE2E2; color: #991B1B; }
        .risk-medium { background: #FEF3C7; color: #92400E; }
        .risk-low { background: #D1FAE5; color: #065F46; }

        /* Primary Button Polish */
        div.stButton > button[kind="primary"] {
            background-color: #2563EB;
            border-color: #2563EB;
            border-radius: 8px;
            font-weight: 600;
            padding: 0.55rem 1.25rem;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #1D4ED8;
            border-color: #1D4ED8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_page_header(title: str, subtitle: str) -> None:
    """Render a consistent page title and subtitle."""
    st.markdown(
        f"""
        <div class="page-header">
            <div class="page-title">{title}</div>
            <div class="page-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(label: str, value: str, subtext: str = "") -> None:
    """Render a styled KPI card."""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, description: str) -> None:
    """Render a clean section header card for grouped form inputs."""
    st.markdown(
        f"""
        <div class="section-card-header">
            <div class="section-card-title">{title}</div>
            <div class="section-card-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
