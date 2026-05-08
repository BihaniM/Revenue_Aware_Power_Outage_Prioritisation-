from __future__ import annotations
import re
import requests
from sqlalchemy.orm import Session
from app.config import ENABLE_OLLAMA, OLLAMA_MODEL, OLLAMA_URL
from app.models import RevenuePriority
from app.services.analytics import site_display_name, STATE_WEIGHT

NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

def _extract_resource_count(prompt: str) -> int:
    text = prompt.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b\s+(generator|generators|vehicle|vehicles|teams|resources)", text):
            return value
    m = re.search(r"\b(\d+)\s+(generator|generators|vehicle|vehicles|teams|resources)", text)
    return int(m.group(1)) if m else 2

def _extract_candidate_sites(prompt: str, db: Session) -> list[RevenuePriority]:
    text = prompt.lower()
    all_sites = db.query(RevenuePriority).all()
    matched = []
    for site in all_sites:
        short = site_display_name(site.site_name).lower()
        compact = re.sub(r"[^a-z0-9]", "", short)
        text_compact = re.sub(r"[^a-z0-9]", "", text)
        full_compact = re.sub(r"[^a-z0-9]", "", site.site_name.lower())
        tokens = [t for t in re.split(r"\s+", short) if len(t) >= 4]
        if compact and compact in text_compact:
            matched.append(site)
        elif full_compact and full_compact in text_compact:
            matched.append(site)
        elif tokens and any(t in text for t in tokens):
            matched.append(site)
    seen = set()
    unique = []
    for site in matched:
        if site.site_name not in seen:
            unique.append(site)
            seen.add(site.site_name)
    return unique

def _priority_score(site: RevenuePriority, prompt: str) -> float:
    score = float(site.total_revenue_hour or 0) + STATE_WEIGHT.get(site.alarm_state, 0) * 10000
    if "vvip" in prompt.lower() and site_display_name(site.site_name).lower() in prompt.lower():
        score += 100000
    return score

def _fallback_answer(prompt: str, db: Session) -> str:
    resources = _extract_resource_count(prompt)
    sites = _extract_candidate_sites(prompt, db)
    if not sites:
        sites = db.query(RevenuePriority).order_by(RevenuePriority.total_revenue_hour.desc()).limit(10).all()
    ranked = sorted(sites, key=lambda s: _priority_score(s, prompt), reverse=True)
    selected = ranked[:resources]
    lines = [
        "Recommended resource allocation order:",
        "",
    ]
    for idx, site in enumerate(ranked, start=1):
        marker = "✅ Allocate now" if site in selected else "⏳ Keep in queue"
        lines.append(
            f"{idx}. {site_display_name(site.site_name)} ({site.region}) - {marker} | "
            f"Hourly revenue: Rs. {site.total_revenue_hour:,.2f} | Alarm state: {site.alarm_state}"
        )
    lines.append("")
    lines.append(f"Available resources detected from prompt: {resources}. Priority is based on revenue generation capacity, active alarm stage, and VVIP mention when available.")
    return "\n".join(lines)

def answer_prompt(prompt: str, db: Session) -> str:
    deterministic_context = _fallback_answer(prompt, db)
    if not ENABLE_OLLAMA:
        return deterministic_context
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": (
                "You are a telecom corrective-maintenance assistant. Rewrite the following recommendation in clear operational English. "
                "Do not change the order or numbers.\n\n"
                f"User request:\n{prompt}\n\nComputed database result:\n{deterministic_context}"
            ),
            "stream": False,
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("response", deterministic_context).strip() or deterministic_context
    except Exception:
        return deterministic_context
