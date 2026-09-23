"""
style.py — Custom CSS and UI helpers for the Streamlit app.
"""

import streamlit as st

# ---------------------------------------------------------------
# Palette
# ---------------------------------------------------------------
CREAM         = "#faf7f2"
CREAM_DARK    = "#f0ebe0"
CRIMSON       = "#8b1e3f"
CRIMSON_LIGHT = "#b8446b"
CHARCOAL      = "#2b2b2b"
SOFT_GREY     = "#6b6b6b"
BORDER        = "#e4dccf"


def inject_css():
    """Inject the global CSS overrides. Call once at the top of the app."""
    st.markdown(
        f"""
        <style>
        html, body, [class*="css"] {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                         Roboto, "Helvetica Neue", Arial, sans-serif;
            color: {CHARCOAL};
        }}

        h1, h2, h3 {{
            font-family: Georgia, "Times New Roman", serif;
            color: {CHARCOAL};
            letter-spacing: -0.01em;
        }}

        h1 {{ font-weight: 700; font-size: 2.4rem; margin-bottom: 0.2rem; }}
        h2 {{ font-weight: 700; font-size: 1.6rem; margin-top: 1.8rem; }}
        h3 {{
            font-weight: 600; font-size: 1.15rem; color: {CRIMSON};
            text-transform: uppercase; letter-spacing: 0.08em;
            margin-top: 1.6rem; margin-bottom: 0.6rem;
        }}

        .stButton > button,
        .stFormSubmitButton > button {{
            background-color: {CRIMSON}; color: white; border: none;
            border-radius: 4px; padding: 0.6rem 1.4rem;
            font-weight: 600; letter-spacing: 0.02em;
            transition: all 0.15s ease;
            box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        }}
        .stButton > button:hover,
        .stFormSubmitButton > button:hover {{
            background-color: {CRIMSON_LIGHT};
            box-shadow: 0 3px 8px rgba(139, 30, 63, 0.25);
            transform: translateY(-1px);
        }}

        .stSelectbox > div > div,
        .stTextInput > div > div,
        .stNumberInput > div > div {{
            border-radius: 4px; border-color: {BORDER};
        }}

        .custom-card {{
            background-color: white; border: 1px solid {BORDER};
            border-radius: 8px; padding: 1.2rem 1.5rem; margin: 1rem 0;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }}
        .custom-card h4 {{
            margin-top: 0; margin-bottom: 0.5rem;
            font-family: Georgia, serif; color: {CHARCOAL};
        }}

        .stAlert {{
            border-radius: 6px; border-left: 4px solid {CRIMSON};
        }}

        hr {{ border: none; border-top: 1px solid {BORDER}; margin: 1.5rem 0; }}

        .prediction-badge {{
            display: inline-block; padding: 0.4rem 1rem;
            border-radius: 999px; font-weight: 700; font-size: 1.1rem;
            letter-spacing: 0.02em;
        }}
        .pred-low      {{ background-color: #e8f5e9; color: #1b5e20; }}
        .pred-moderate {{ background-color: #fff8e1; color: #7c5e00; }}
        .pred-high     {{ background-color: #fdecea; color: #a32424; }}

        .app-footer {{
            color: {SOFT_GREY}; font-size: 0.85rem; text-align: center;
            margin-top: 3rem; padding-top: 1.5rem;
            border-top: 1px solid {BORDER};
        }}

        .hero-band {{
            background: linear-gradient(135deg, {CREAM_DARK} 0%, {CREAM} 100%);
            border-left: 5px solid {CRIMSON};
            padding: 1.5rem 2rem; border-radius: 6px; margin-bottom: 1.5rem;
        }}
        .hero-band .eyebrow {{
            font-size: 0.75rem; letter-spacing: 0.15em;
            text-transform: uppercase; color: {CRIMSON}; font-weight: 700;
        }}
        .hero-band h1 {{ margin: 0.2rem 0 0.4rem 0; }}
        .hero-band .subtitle {{ color: {SOFT_GREY}; font-size: 1rem; margin: 0; }}

        .section-marker {{
            display: flex; align-items: center;
            margin-top: 1.8rem; margin-bottom: 0.8rem;
        }}
        .section-marker::before {{
            content: ""; display: inline-block; width: 4px; height: 18px;
            background-color: {CRIMSON}; margin-right: 0.6rem;
            border-radius: 2px;
        }}
        .section-marker span {{
            font-family: Georgia, serif; font-size: 1.15rem;
            font-weight: 700; color: {CHARCOAL};
        }}

        .tip-card {{
            background: #ffffff; border: 1px solid {BORDER};
            border-radius: 10px; padding: 1.2rem 1.4rem;
            margin-bottom: 1rem; position: relative;
            transition: all 0.15s ease; height: 100%;
        }}
        .tip-card:hover {{
            border-color: {CRIMSON};
            box-shadow: 0 4px 12px rgba(139, 30, 63, 0.08);
            transform: translateY(-2px);
        }}
        .tip-card .tip-number {{
            position: absolute; top: 1rem; right: 1.2rem;
            font-family: Georgia, serif; font-size: 1.6rem;
            font-weight: 700; color: {BORDER}; line-height: 1;
        }}
        .tip-card .tip-icon {{
            font-size: 1.6rem; margin-bottom: 0.4rem; display: block;
        }}
        .tip-card .tip-category {{
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: {CRIMSON};
            margin-bottom: 0.5rem;
        }}
        .tip-card .tip-body {{
            font-size: 0.92rem; line-height: 1.5;
            color: {CHARCOAL}; margin: 0;
        }}
        .tip-card .tip-action {{
            margin-top: 0.7rem; font-size: 0.85rem;
            color: {CRIMSON}; font-weight: 600;
        }}

        .priority-banner {{
            background: linear-gradient(135deg, {CREAM_DARK} 0%, {CREAM} 100%);
            border-radius: 10px; padding: 1.5rem 1.8rem;
            margin: 1rem 0 1.5rem 0;
            border-left: 5px solid {CRIMSON};
        }}
        .priority-banner .pb-label {{
            font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
            text-transform: uppercase; color: {CRIMSON};
            margin-bottom: 0.4rem;
        }}
        .priority-banner .pb-title {{
            font-family: Georgia, serif; font-size: 1.4rem;
            font-weight: 700; color: {CHARCOAL}; margin: 0 0 0.5rem 0;
        }}
        .priority-banner .pb-desc {{
            color: #4a4a4a; font-size: 0.95rem; line-height: 1.55; margin: 0;
        }}

        .habit-card {{
            background: #ffffff; border: 1px solid {BORDER};
            border-radius: 10px; padding: 1.1rem 1.3rem;
            margin-bottom: 0.9rem; height: 100%;
        }}
        .habit-card .habit-icon {{
            font-size: 1.4rem; margin-right: 0.5rem; vertical-align: middle;
        }}
        .habit-card .habit-title {{
            display: inline-block; font-family: Georgia, serif;
            font-size: 1.05rem; font-weight: 700;
            color: {CHARCOAL}; vertical-align: middle;
        }}
        .habit-card .habit-body {{
            font-size: 0.9rem; line-height: 1.5;
            color: #4a4a4a; margin: 0.5rem 0 0 0;
        }}

        .help-card {{
            background: {CHARCOAL}; color: #ffffff;
            border-radius: 10px; padding: 1.6rem 2rem; margin: 1rem 0;
        }}
        .help-card .help-label {{
            font-size: 0.7rem; letter-spacing: 0.15em;
            text-transform: uppercase; color: {CRIMSON_LIGHT};
            font-weight: 700; margin-bottom: 0.5rem;
        }}
        .help-card .help-title {{
            font-family: Georgia, serif; font-size: 1.35rem;
            font-weight: 700; margin: 0 0 0.3rem 0; color: #ffffff;
        }}
        .help-card .help-number {{
            font-family: Georgia, serif; font-size: 2rem;
            font-weight: 700; color: #ffffff; margin: 0.3rem 0 0.6rem 0;
        }}
        .help-card .help-desc {{
            color: #cccccc; font-size: 0.9rem; line-height: 1.55; margin: 0;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, eyebrow: str = ""):
    eyebrow_html = f'<div class="eyebrow">{eyebrow}</div>' if eyebrow else ""
    st.markdown(
        f"""
        <div class="hero-band">
            {eyebrow_html}
            <h1>{title}</h1>
            <p class="subtitle">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str):
    st.markdown(
        f'<div class="section-marker"><span>{title}</span></div>',
        unsafe_allow_html=True,
    )


def card(body_html: str):
    st.markdown(f'<div class="custom-card">{body_html}</div>', unsafe_allow_html=True)


def prediction_badge(level: str) -> str:
    cls = {"Low": "pred-low", "Moderate": "pred-moderate", "High": "pred-high"}.get(level, "")
    return f'<span class="prediction-badge {cls}">{level}</span>'


def render_priority_banner(label: str, title: str, description: str):
    st.markdown(
        f"""
        <div class="priority-banner">
            <div class="pb-label">{label}</div>
            <div class="pb-title">{title}</div>
            <p class="pb-desc">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tip_card(number: int, icon: str, category: str, body: str, action: str = ""):
    action_html = f'<div class="tip-action">→ {action}</div>' if action else ""
    st.markdown(
        f"""
        <div class="tip-card">
            <div class="tip-number">{number:02d}</div>
            <span class="tip-icon">{icon}</span>
            <div class="tip-category">{category}</div>
            <p class="tip-body">{body}</p>
            {action_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_habit_card(icon: str, title: str, body: str):
    st.markdown(
        f"""
        <div class="habit-card">
            <span class="habit-icon">{icon}</span>
            <span class="habit-title">{title}</span>
            <p class="habit-body">{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_help_card():
    st.markdown(
        """
        <div class="help-card">
            <div class="help-label">If things feel overwhelming</div>
            <h3 class="help-title">Talk to someone you trust</h3>
            <div class="help-number">1166</div>
            <p class="help-desc">
                Nepal Mental Health Helpline · toll-free · 24/7.<br>
                You can also reach out to a campus counselor, a mentor,
                a close friend, or a family member.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def footer():
    st.markdown(
        """
        <div class="app-footer">
            Saroj Lamichhane · Santosh Shrestha · BCA Final-Year Project · Tribhuvan University <br>
            <a href="https://github.com/santoshh007/academic-stress-prediction-ml"
               target="_blank" style="color:#8b1e3f;">View source on GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )