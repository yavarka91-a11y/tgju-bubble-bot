# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import re
from datetime import datetime
from io import StringIO
from typing import Tuple, Dict, Any

import requests
import pandas as pd

CURRENCY_URL = "https://www.tgju.org/currency"
AED_TO_USD_REF = 3.672

CSV_PATH = "live_fx_snapshots.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Connection": "keep-alive",
}

PERSIAN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")


def norm(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().translate(PERSIAN_DIGITS)
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


# --- Jalali (Shamsi) without extra libs
def gregorian_to_jalali(gy: int, gm: int, gd: int) -> Tuple[int, int, int]:
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    if gy > 1600:
        jy = 979
        gy -= 1600
    else:
        jy = 0
        gy -= 621

    gy2 = gy + 1 if gm > 2 else gy
    days = (365 * gy) + ((gy2 + 3) // 4) - ((gy2 + 99) // 100) + ((gy2 + 399) // 400) - 80 + gd + g_d_m[gm - 1]
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


def fetch_live_usd_aed_from_currency() -> Tuple[float, float]:
    r = requests.get(CURRENCY_URL, headers=HEADERS, timeout=25)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    target = None

    for t in tables:
        cols = [norm(c) for c in list(t.columns)]
        if ("عنوان" in cols) and ("قیمت زنده" in cols):
            target = t.copy()
            target.columns = cols
            break

    if target is None:
        raise RuntimeError("جدول 'عنوان/قیمت زنده' در صفحه پیدا نشد (احتمالاً ساختار صفحه تغییر کرده).")

    target["عنوان"] = target["عنوان"].apply(norm)

    def get_price(title: str) -> float:
        row = target[target["عنوان"] == title]
        if row.empty:
            raise RuntimeError(f"ردیف '{title}' در جدول پیدا نشد.")
        val = row.iloc[0]["قیمت زنده"]
        num = to_number(val)
        if num is None:
            raise RuntimeError(f"قیمت '{title}' قابل تبدیل به عدد نیست: {val}")
        return float(num)

    usd = get_price("دلار")
    aed = get_price("درهم امارات")
    return usd, aed


def classify_usd_bubble(diff: float) -> Tuple[str, str]:
    if diff > 0:
        return "حباب مثبت دلار", "پیشنهاد: فروش دلار"
    elif diff < 0:
        return "حباب منفی دلار", "پیشنهاد: خرید دلار"
    else:
        return "بدون حباب", "پیشنهاد: خنثی"


def append_csv(record: Dict[str, Any]) -> pd.DataFrame:
    df_new = pd.DataFrame([record])
    if os.path.exists(CSV_PATH):
        df_old = pd.read_csv(CSV_PATH)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new
    df_all.to_csv(CSV_PATH, index=False, encoding="utf-8-sig")
    return df_all


def send_telegram(text: str):
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID تنظیم نشده‌اند (در GitHub Secrets).")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()


def main():
    date_sh, time_sh = jalali_now_str()
    usd, aed = fetch_live_usd_aed_from_currency()

    implied_usd = aed * AED_TO_USD_REF
    diff = usd - implied_usd
    bubble_percent = (diff / implied_usd * 100) if implied_usd else 0.0

    bubble, suggestion = classify_usd_bubble(diff)

    # Round for readability
    usd_r = round(usd, 2)
    aed_r = round(aed, 2)
    implied_r = round(implied_usd, 2)
    diff_r = round(diff, 2)
    pct_r = round(bubble_percent, 5)

    record = {
        "date_shamsi": date_sh,
        "time": time_sh,
        "usd": usd_r,
        "aed": aed_r,
        "implied_usd_from_aed": implied_r,
        "diff_usd_minus_implied": diff_r,
        "bubble_percent": pct_r,
        "bubble_usd": bubble,
        "suggestion": suggestion,
        "source": "tgju.org/currency",
    }

    df_all = append_csv(record)

    msg = (
        f"TGJU Bubble Monitor\n"
        f"{date_sh} {time_sh}\n"
        f"USD: {usd_r}\n"
        f"AED: {aed_r}\n"
        f"Implied USD (AED*{AED_TO_USD_REF}): {implied_r}\n"
        f"Diff (USD - implied): {diff_r}  |  {pct_r}%\n"
        f"{bubble}\n"
        f"{suggestion}\n"
        f"Rows stored: {len(df_all)}"
    )

    send_telegram(msg)
    print(msg)


if __name__ == "__main__":
    main()
