# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
import time
import random
from datetime import datetime
from io import StringIO
from typing import Tuple, Dict, Any, Optional

import requests
import pandas as pd

# -------------------------
# URLs
# -------------------------
# USD/AED from stable profile pages
USD_PROFILE_URL = "https://www.tgju.org/profile/price_dollar_rl"
AED_PROFILE_URL = "https://www.tgju.org/profile/price_aed"

# USDT from fast-updating crypto table
CRYPTO_URL = "https://www.tgju.org/crypto"

AED_TO_USD_REF = 3.672

CSV_PATH = "live_fx_snapshots.csv"
STATE_PATH = "state.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.7,en;q=0.6",
    "Connection": "keep-alive",
    "Referer": "https://www.tgju.org/",
    "DNT": "1",
}

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


# -------------------------
# Normalization & parsing
# -------------------------
def norm(s: str) -> str:
    """
    Normalize Persian text for robust matching.
    - Converts Persian digits to English digits
    - Removes kashida 'ـ'
    - Normalizes Arabic y/k to Persian ی/ک
    - Removes zero-width characters
    - Collapses whitespace
    """
    if s is None:
        return ""
    s = str(s).strip().translate(PERSIAN_DIGITS)

    # remove kashida (very common in TGJU headings e.g. ریـال, تتـر)
    s = s.replace("ـ", "")

    # normalize Arabic chars
    s = s.replace("ي", "ی").replace("ك", "ک")

    # remove zero-width marks
    s = s.replace("\u200c", " ").replace("\u200f", " ").replace("\u200e", " ")
    s = re.sub(r"\s+", " ", s)
    return s


def to_number(x) -> float | None:
    if x is None:
        return None
    s = norm(x).replace(",", "").replace("٫", ".")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def fmt_int(n: float | int) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except Exception:
        return str(n)


# -------------------------
# Jalali (Shamsi) without extra libs
# -------------------------
def gregorian_to_jalali(gy: int, gm: int, gd: int) -> Tuple[int, int, int]:
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621

    gy2 = gy + 1 if gm > 2 else gy
    days = (
        (365 * gy)
        + ((gy2 + 3) // 4)
        - ((gy2 + 99) // 100)
        + ((gy2 + 399) // 400)
        - 80
        + gd
        + g_d_m[gm - 1]
    )
    jy += 33 * (days // 12053)
    days %= 12053
    jy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        jy += (days - 1) // 365
        days = (days - 1) % 365

    if days < 186:
        jm = 1 + (days // 31)
        jd = 1 + (days % 31)
    else:
        jm = 7 + ((days - 186) // 30)
        jd = 1 + ((days - 186) % 30)

    return jy, jm, jd


def jalali_now_str() -> Tuple[str, str]:
    now = datetime.now()
    jy, jm, jd = gregorian_to_jalali(now.year, now.month, now.day)
    return f"{jy:04d}/{jm:02d}/{jd:02d}", now.strftime("%H:%M:%S")


# -------------------------
# HTTP helpers (retry + backoff + jitter)
# -------------------------
RETRYABLE_STATUS = {403, 429, 500, 502, 503, 504}


def _compute_backoff(attempt: int, base: float = 1.2, cap: float = 25.0) -> float:
    expo = min(cap, base * (2 ** (attempt - 1)))
    jitter = random.uniform(0.0, 0.35 * expo)
    return expo + jitter


def http_get_text(url: str, timeout: int = 25, retries: int = 5) -> str:
    last_err = None
    sess = requests.Session()

    for attempt in range(1, retries + 1):
        try:
            r = sess.get(url, headers=HEADERS, timeout=timeout)

            if r.status_code in RETRYABLE_STATUS:
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    sleep_s = int(retry_after)
                else:
                    sleep_s = _compute_backoff(attempt)

                if attempt < retries:
                    print(f"⚠️ HTTP {r.status_code} برای {url} | retry in {sleep_s:.1f}s (attempt {attempt}/{retries})")
                    time.sleep(sleep_s)
                    continue

            r.raise_for_status()
            return r.text

        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            sleep_s = _compute_backoff(attempt)
            print(f"⚠️ خطای دریافت {url}: {e} | retry in {sleep_s:.1f}s (attempt {attempt}/{retries})")
            time.sleep(sleep_s)

    raise RuntimeError(f"خطا در دریافت صفحه: {url} | {last_err}")


# -------------------------
# Robust HTML extraction for profile pages
# -------------------------
def extract_value_after_label(html: str, label: str) -> Optional[str]:
    if not html:
        return None
    h = html.translate(PERSIAN_DIGITS)
    label_pat = re.escape(label)

    m = re.search(
        rf"{label_pat}[\s:：]*?(?:</[^>]+>\s*)*([0-9][0-9,\.\s]*)",
        h,
        flags=re.IGNORECASE,
    )
    if m:
        return m.group(1).strip()

    m = re.search(
        rf"{label_pat}(.{{0,250}}?)",
        h,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        chunk = m.group(1)
        n = re.search(r"([0-9][0-9,\.\s]*)", chunk)
        if n:
            return n.group(1).strip()

    return None


def fetch_profile_kv_table(url: str) -> Tuple[Dict[str, str], str]:
    html = http_get_text(url)
    kv: Dict[str, str] = {}

    try:
        tables = pd.read_html(StringIO(html))
    except Exception:
        tables = []

    for t in tables:
        if t is None or t.empty:
            continue
        if t.shape[1] < 2:
            continue

        df = t.copy().iloc[:, :2]
        df.columns = ["k", "v"]
        df["k"] = df["k"].astype(str).map(norm)
        df["v"] = df["v"].astype(str).map(norm)

        keys = set(df["k"].tolist())
        if ("نرخ فعلی" in keys) or ("قیمت ریالی" in keys) or ("زمان ثبت آخرین نرخ" in keys):
            for _, row in df.iterrows():
                k = row["k"]
                v = row["v"]
                if k and v:
                    kv[k] = v
            if kv:
                break

    return kv, html


def get_rate_from_profile(url: str, label: str) -> Tuple[float, str]:
    kv, html = fetch_profile_kv_table(url)

    raw = kv.get(label)
    if not raw:
        raw = extract_value_after_label(html, label)

    num = to_number(raw)
    if num is None or num <= 0:
        raise RuntimeError(f"نتوانستم '{label}' را از صفحه استخراج کنم: {url} | raw={raw}")

    return float(num), (raw or "")


def fetch_usd_aed_from_profiles() -> Tuple[float, float, str, str]:
    usd, usd_raw = get_rate_from_profile(USD_PROFILE_URL, "نرخ فعلی")
    aed, aed_raw = get_rate_from_profile(AED_PROFILE_URL, "نرخ فعلی")
    return usd, aed, usd_raw, aed_raw


# -------------------------
# USDT from /crypto table (fast update)
# -------------------------
def fetch_usdt_from_crypto() -> Tuple[float, str]:
    """
    Extract USDT price in Rial from TGJU /crypto table.
    We pick the row where 'نماد' == USDT and read the column that contains both 'قیمت' and 'ریال'.
    Returns (value_float, raw_string).
    """
    html = http_get_text(CRYPTO_URL)

    tables = pd.read_html(StringIO(html))
    if not tables:
        raise RuntimeError("هیچ جدولی از صفحه crypto استخراج نشد (ممکن است سایت موقتاً محدود کرده باشد).")

    for df in tables:
        df2 = df.copy()
        df2.columns = [norm(c) for c in df2.columns]

        if "نماد" not in df2.columns:
            continue

        # find rial price column (handles ریـال/ریال)
        rial_col = None
        for c in df2.columns:
            cn = norm(c)
            if ("قیمت" in cn) and ("ریال" in cn):
                rial_col = c
                break
        if rial_col is None:
            continue

        sym = df2["نماد"].astype(str).map(norm).str.upper()
        row = df2[sym == "USDT"]

        if row.empty:
            # optional fallback: match by name column containing "تتر"
            name_col = None
            for c in df2.columns:
                cn = norm(c)
                if ("ارزهای دیجیتال" in cn) or ("عنوان" in cn) or ("نام" in cn):
                    name_col = c
                    break
            if name_col is not None:
                name = df2[name_col].astype(str).map(norm)
                row = df2[name.str.contains("تتر", na=False) | name.str.contains("USDT", na=False)]

        if row.empty:
            continue

        raw = row.iloc[0][rial_col]
        num = to_number(raw)
        if num is None or num <= 0:
            raise RuntimeError(f"قیمت USDT از ستون ریالی قابل تبدیل نبود: {raw}")

        return float(num), str(raw)

    raise RuntimeError("جدول/ستون مناسب یا ردیف USDT در صفحه crypto پیدا نشد (احتمال تغییر ساختار صفحه).")


# -------------------------
# Bubble logic
# -------------------------
def bubble_state_from_diff(diff: float) -> str:
    if diff > 0:
        return "positive"
    elif diff < 0:
        return "negative"
    else:
        return "neutral"


def bubble_label_and_suggestion(diff: float) -> Tuple[str, str]:
    if diff > 0:
        return "حباب مثبت دلار", "پیشنهاد: فروش دلار"
    elif diff < 0:
        return "حباب منفی دلار", "پیشنهاد: خرید دلار"
    else:
        return "بدون حباب", "پیشنهاد: خنثی"


def usdt_bubble_label(diff_usdt_minus_usd: float) -> str:
    if diff_usdt_minus_usd > 0:
        return "حباب مثبت تتر/دلار"
    elif diff_usdt_minus_usd < 0:
        return "حباب منفی تتر/دلار"
    else:
        return "بدون حباب تتر/دلار"


def usdt_trade_matrix(usd_aed_state: str, usdt_usd_state: str) -> str:
    if usd_aed_state not in {"positive", "negative"} or usdt_usd_state not in {"positive", "negative"}:
        return "⚪️ وضعیت خنثی/نامشخص (یکی از حباب‌ها نزدیک صفر است)"

    if usd_aed_state == "negative" and usdt_usd_state == "negative":
        return "🟢 زمان قطعی خرید تتر"
    if usd_aed_state == "positive" and usdt_usd_state == "negative":
        return "🟡 زمان فروش محتاطانه تتر"
    if usd_aed_state == "negative" and usdt_usd_state == "positive":
        return "🟡 زمان خرید محتاطانه تتر"
    if usd_aed_state == "positive" and usdt_usd_state == "positive":
        return "🔴 زمان قطعی فروش تتر"

    return "⚪️ نامشخص"


# -------------------------
# State / Storage / Telegram
# -------------------------
def load_prev_state() -> Optional[Dict[str, Any]]:
    if not os.path.exists(STATE_PATH):
        return None
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def append_csv(record: Dict[str, Any]) -> int:
    df_new = pd.DataFrame([record])
    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return len(df_all)


def send_telegram(text: str):
    required = os.environ.get("TELEGRAM_REQUIRED", "").strip() == "1"
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        msg = "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID تنظیم نشده‌اند."
        if required:
            raise RuntimeError(msg + " (GitHub Secrets)")
        print("⚠️ " + msg + " => ارسال تلگرام انجام نشد، فقط print شد.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    rr = requests.post(url, json=payload, timeout=20)
    rr.raise_for_status()


# -------------------------
# Messages
# -------------------------
def build_main_message(
    date_sh: str,
    time_sh: str,
    usd: float,
    aed: float,
    implied: float,
    diff_usd: float,
    pct_usd: float,
    bubble_usd_label: str,
    suggestion_usd: str,
    usdt: float,
    diff_usdt: float,
    pct_usdt: float,
    usdt_label: str,
    final_usdt_signal: str,
    rows_total: int,
) -> str:
    arrow_usd = "📈" if diff_usd > 0 else ("📉" if diff_usd < 0 else "➖")
    sign_usd = "➕" if diff_usd > 0 else ("➖" if diff_usd < 0 else "➖")

    arrow_usdt = "📈" if diff_usdt > 0 else ("📉" if diff_usdt < 0 else "➖")
    sign_usdt = "➕" if diff_usdt > 0 else ("➖" if diff_usdt < 0 else "➖")

    return (
        "📊 TGJU Bubble Monitor\n"
        f"🗓 {date_sh}  ⏰ {time_sh}\n\n"
        "🟦 USD/AED (Bubble for USD)\n"
        f"💵 USD: {fmt_int(usd)}\n"
        f"🇦🇪 AED: {fmt_int(aed)}\n"
        f"🔗 Implied USD (AED×{AED_TO_USD_REF}): {fmt_int(implied)}\n"
        f"{arrow_usd} Bubble Diff: {sign_usd} {fmt_int(abs(diff_usd))} تومان   ({pct_usd:+.4f}%)\n"
        f"⚠️ {bubble_usd_label}\n"
        f"✅ {suggestion_usd}\n\n"
        "🟩 USDT/USD (Bubble for USDT vs USD)\n"
        f"🪙 USDT (IRR): {fmt_int(usdt)}\n"
        f"{arrow_usdt} USDT-USD Diff: {sign_usdt} {fmt_int(abs(diff_usdt))} تومان   ({pct_usdt:+.4f}%)\n"
        f"⚠️ {usdt_label}\n\n"
        f"🎯 سیگنال نهایی تتر: {final_usdt_signal}\n\n"
        f"🧾 Rows stored: {rows_total}"
    )


def build_alert_change_message(
    title: str,
    prev_state: str,
    new_state: str,
    date_sh: str,
    time_sh: str,
    diff: float,
    pct: float
) -> str:
    mapping = {"positive": "مثبت", "negative": "منفی", "neutral": "خنثی"}
    return (
        f"🚨 هشدار تغییر وضعیت ({title})\n"
        f"🔄 {mapping.get(prev_state, prev_state)} ➜ {mapping.get(new_state, new_state)}\n"
        f"🗓 {date_sh}  ⏰ {time_sh}\n"
        f"Diff: {diff:+.2f} تومان | {pct:+.4f}%"
    )


# -------------------------
# Main
# -------------------------
def main():
    date_sh, time_sh = jalali_now_str()

    # USD + AED (profiles)
    usd, aed, usd_raw, aed_raw = fetch_usd_aed_from_profiles()
    implied_usd = aed * AED_TO_USD_REF
    diff_usd = usd - implied_usd
    pct_usd = (diff_usd / implied_usd * 100) if implied_usd else 0.0
    bubble_usd_label, suggestion_usd = bubble_label_and_suggestion(diff_usd)
    usd_aed_state = bubble_state_from_diff(diff_usd)

    # USDT (crypto table)
    usdt, usdt_raw = fetch_usdt_from_crypto()

    diff_usdt = usdt - usd
    pct_usdt = (diff_usdt / usd * 100) if usd else 0.0
    usdt_state = bubble_state_from_diff(diff_usdt)
    usdt_label = usdt_bubble_label(diff_usdt)

    final_usdt_signal = usdt_trade_matrix(usd_aed_state, usdt_state)

    # round for storage
    usd_r = round(usd, 2)
    aed_r = round(aed, 2)
    implied_r = round(implied_usd, 2)
    diff_usd_r = round(diff_usd, 2)
    pct_usd_r = round(pct_usd, 6)

    usdt_r = round(usdt, 2)
    diff_usdt_r = round(diff_usdt, 2)
    pct_usdt_r = round(pct_usdt, 6)

    record = {
        "date_shamsi": date_sh,
        "time": time_sh,

        "usd": usd_r,
        "usd_raw": norm(usd_raw),
        "aed": aed_r,
        "aed_raw": norm(aed_raw),

        "implied_usd_from_aed": implied_r,
        "diff_usd_minus_implied": diff_usd_r,
        "bubble_percent": pct_usd_r,
        "bubble_state": usd_aed_state,
        "bubble_usd": bubble_usd_label,
        "suggestion": suggestion_usd,
        "usd_source": USD_PROFILE_URL,
        "aed_source": AED_PROFILE_URL,

        "usdt": usdt_r,
        "usdt_raw": norm(usdt_raw),
        "diff_usdt_minus_usd": diff_usdt_r,
        "usdt_bubble_percent": pct_usdt_r,
        "usdt_bubble_state": usdt_state,
        "usdt_bubble_label": usdt_label,
        "usdt_final_signal": final_usdt_signal,
        "usdt_source": CRYPTO_URL,
    }

    rows_total = append_csv(record)

    main_msg = build_main_message(
        date_sh, time_sh,
        usd_r, aed_r, implied_r,
        diff_usd_r, pct_usd_r,
        bubble_usd_label, suggestion_usd,
        usdt_r, diff_usdt_r, pct_usdt_r,
        usdt_label, final_usdt_signal,
        rows_total
    )

    send_telegram(main_msg)

    prev = load_prev_state() or {}
    prev_usd_state = prev.get("bubble_state")
    prev_usdt_state = prev.get("usdt_bubble_state")

    if prev_usd_state and prev_usd_state != usd_aed_state:
        send_telegram(build_alert_change_message(
            "USD/AED", prev_usd_state, usd_aed_state, date_sh, time_sh, diff_usd_r, pct_usd_r
        ))

    if prev_usdt_state and prev_usdt_state != usdt_state:
        send_telegram(build_alert_change_message(
            "USDT/USD", prev_usdt_state, usdt_state, date_sh, time_sh, diff_usdt_r, pct_usdt_r
        ))

    save_state({
        "bubble_state": usd_aed_state,
        "usdt_bubble_state": usdt_state,
        "last_raw": {
            "usd_raw": norm(usd_raw),
            "aed_raw": norm(aed_raw),
            "usdt_raw": norm(usdt_raw),
        },
        "updated_at": f"{date_sh} {time_sh}",
    })

    print(main_msg)


if __name__ == "__main__":
    main()
