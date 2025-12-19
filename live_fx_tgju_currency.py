# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from datetime import datetime
from io import StringIO
from typing import Tuple, Dict, Any, Optional

import requests
import pandas as pd

CURRENCY_URL = "https://www.tgju.org/currency"
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


def fmt_int(n: float | int) -> str:
    # 1,316,300
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


# -------------------------
# TGJU currency (USD, AED)
# -------------------------
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
        raise RuntimeError("جدول 'عنوان/قیمت زنده' در صفحه currency پیدا نشد (احتمالاً ساختار صفحه تغییر کرده).")

    target["عنوان"] = target["عنوان"].apply(norm)

    def get_price(title: str) -> float:
        row = target[target["عنوان"] == title]
        if row.empty:
            raise RuntimeError(f"ردیف '{title}' در جدول currency پیدا نشد.")
        val = row.iloc[0]["قیمت زنده"]
        num = to_number(val)
        if num is None:
            raise RuntimeError(f"قیمت '{title}' قابل تبدیل به عدد نیست: {val}")
        return float(num)

    usd = get_price("دلار")
    aed = get_price("درهم امارات")
    return usd, aed


# -------------------------
# TGJU crypto (USDT)
# -------------------------
def fetch_live_usdt_from_crypto() -> float:
    r = requests.get(CRYPTO_URL, headers=HEADERS, timeout=25)
    r.raise_for_status()

    tables = pd.read_html(StringIO(r.text))
    # جدول/ستون‌ها در crypto ممکنه متفاوت باشه؛ ما انعطاف‌پذیر می‌گردیم:
    candidates = []
    for t in tables:
        cols = [norm(c) for c in list(t.columns)]
        if "عنوان" in cols:
            candidates.append((t.copy(), cols))

    if not candidates:
        raise RuntimeError("هیچ جدولی با ستون 'عنوان' در صفحه crypto پیدا نشد (احتمالاً ساختار صفحه تغییر کرده).")

    # پیدا کردن ستون قیمت: یکی از این‌ها
    price_col_names = {"قیمت", "قیمت زنده", "قیمت لحظه ای", "آخرین قیمت", "آخرین"}
    for df, cols in candidates:
        df.columns = [norm(c) for c in df.columns]
        df["عنوان"] = df["عنوان"].apply(norm)

        # ستون قیمت را پیدا کن
        price_col = None
        for c in df.columns:
            if norm(c) in price_col_names:
                price_col = c
                break
        # اگر اسم دقیق نبود، یک ستون عددیِ محتمل را حدس بزن
        if price_col is None:
            # اولین ستونی که غیر از "عنوان" هست و عدد می‌دهد
            for c in df.columns:
                if c == "عنوان":
                    continue
                sample = df[c].dropna().head(5).astype(str).tolist()
                if any(to_number(x) is not None for x in sample):
                    price_col = c
                    break

        if price_col is None:
            continue

        # ردیف تتر
        row = df[df["عنوان"] == "تتر"]
        if row.empty:
            # بعضی وقت‌ها "تتر (USDT)" یا مشابه
            row = df[df["عنوان"].str.contains("تتر", na=False)]
        if row.empty:
            continue

        val = row.iloc[0][price_col]
        num = to_number(val)
        if num is not None:
            return float(num)

    raise RuntimeError("قیمت تتر در صفحه crypto پیدا نشد. (ممکنه عنوان/ستون‌ها تغییر کرده باشند.)")


# -------------------------
# Bubble logic (USD from AED, USDT from USD)
# -------------------------
def bubble_state_from_diff(diff: float) -> str:
    if diff > 0:
        return "positive"
    elif diff < 0:
        return "negative"
    else:
        return "neutral"


def bubble_label_and_suggestion(diff: float) -> Tuple[str, str]:
    # برای دلار/درهم (همان قبلی)
    if diff > 0:
        return "حباب مثبت دلار", "پیشنهاد: فروش دلار"
    elif diff < 0:
        return "حباب منفی دلار", "پیشنهاد: خرید دلار"
    else:
        return "بدون حباب", "پیشنهاد: خنثی"


def usdt_bubble_label(diff_usdt_minus_usd: float) -> str:
    # تتر/دلار: اگر USDT > USD => حباب مثبت تتر/دلار
    if diff_usdt_minus_usd > 0:
        return "حباب مثبت تتر/دلار"
    elif diff_usdt_minus_usd < 0:
        return "حباب منفی تتر/دلار"
    else:
        return "بدون حباب تتر/دلار"


def usdt_trade_matrix(usd_aed_state: str, usdt_usd_state: str) -> str:
    """
    4 حالت طبق گفته‌ی شما (با positive/negative).
    اگر neutral رخ داد، خروجی احتیاط/نامشخص می‌ده.
    """
    if usd_aed_state not in {"positive", "negative"} or usdt_usd_state not in {"positive", "negative"}:
        return "⚪️ وضعیت خنثی/نامشخص: تصمیم‌گیری احتیاطی (یکی از حباب‌ها نزدیک صفر است)"

    # 1) منفی/منفی => زمان قطعی خرید تتر
    if usd_aed_state == "negative" and usdt_usd_state == "negative":
        return "🟢 زمان قطعی خرید تتر"

    # 2) مثبت/منفی => زمان فروش محتاطانه تتر
    if usd_aed_state == "positive" and usdt_usd_state == "negative":
        return "🟡 زمان فروش محتاطانه تتر"

    # 3) منفی/مثبت => زمان خرید محتاطانه تتر
    if usd_aed_state == "negative" and usdt_usd_state == "positive":
        return "🟡 زمان خرید محتاطانه تتر"

    # 4) مثبت/مثبت => زمان قطعی فروش تتر
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
    token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID تنظیم نشده‌اند (GitHub Secrets).")

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text}
    r = requests.post(url, json=payload, timeout=20)
    r.raise_for_status()


# -------------------------
# Message builders (keep previous parts + add new parts)
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
    # بخش قبلی (دلار/درهم) دست نخورده + فقط زیباسازی
    arrow_usd = "📈" if diff_usd > 0 else ("📉" if diff_usd < 0 else "➖")
    sign_usd = "➕" if diff_usd > 0 else ("➖" if diff_usd < 0 else "➖")

    # بخش جدید (تتر/دلار)
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
        f"🪙 USDT: {fmt_int(usdt)}\n"
        f"{arrow_usdt} USDT-USD Diff: {sign_usdt} {fmt_int(abs(diff_usdt))} تومان   ({pct_usdt:+.4f}%)\n"
        f"⚠️ {usdt_label}\n\n"
        f"🎯 سیگنال نهایی تتر: {final_usdt_signal}\n\n"
        f"🧾 Rows stored: {rows_total}"
    )


def build_alert_change_message(title: str, prev_state: str, new_state: str, date_sh: str, time_sh: str, diff: float, pct: float) -> str:
    mapping = {
        "positive": "حباب مثبت",
        "negative": "حباب منفی",
        "neutral": "خنثی",
    }
    return (
        f"🚨 هشدار تغییر وضعیت ({title})\n"
        f"🔄 {mapping.get(prev_state, prev_state)}  ➜  {mapping.get(new_state, new_state)}\n"
        f"🗓 {date_sh}  ⏰ {time_sh}\n"
        f"Diff: {diff:+.2f} تومان | {pct:+.4f}%"
    )


# -------------------------
# Main
# -------------------------
def main():
    date_sh, time_sh = jalali_now_str()

    # 1) USD + AED from currency
    usd, aed = fetch_live_usd_aed_from_currency()

    implied_usd = aed * AED_TO_USD_REF
    diff_usd = usd - implied_usd
    pct_usd = (diff_usd / implied_usd * 100) if implied_usd else 0.0

    bubble_usd_label, suggestion_usd = bubble_label_and_suggestion(diff_usd)
    usd_aed_state = bubble_state_from_diff(diff_usd)

    # 2) USDT from crypto, compare with USD
    usdt = fetch_live_usdt_from_crypto()
    diff_usdt = usdt - usd
    pct_usdt = (diff_usdt / usd * 100) if usd else 0.0
    usdt_state = bubble_state_from_diff(diff_usdt)
    usdt_label = usdt_bubble_label(diff_usdt)

    # 3) Final matrix signal for USDT trading
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
        # قبلی‌ها
        "date_shamsi": date_sh,
        "time": time_sh,
        "usd": usd_r,
        "aed": aed_r,
        "implied_usd_from_aed": implied_r,
        "diff_usd_minus_implied": diff_usd_r,
        "bubble_percent": pct_usd_r,
        "bubble_state": usd_aed_state,
        "bubble_usd": bubble_usd_label,
        "suggestion": suggestion_usd,
        "source": "tgju.org/currency",

        # جدیدها (بدون دست زدن به قبلی‌ها)
        "usdt": usdt_r,
        "diff_usdt_minus_usd": diff_usdt_r,
        "usdt_bubble_percent": pct_usdt_r,
        "usdt_bubble_state": usdt_state,
        "usdt_bubble_label": usdt_label,
        "usdt_final_signal": final_usdt_signal,
        "usdt_source": "tgju.org/crypto",
    }

    rows_total = append_csv(record)

    main_msg = build_main_message(
        date_sh=date_sh,
        time_sh=time_sh,
        usd=usd_r,
        aed=aed_r,
        implied=implied_r,
        diff_usd=diff_usd_r,
        pct_usd=pct_usd_r,
        bubble_usd_label=bubble_usd_label,
        suggestion_usd=suggestion_usd,
        usdt=usdt_r,
        diff_usdt=diff_usdt_r,
        pct_usdt=pct_usdt_r,
        usdt_label=usdt_label,
        final_usdt_signal=final_usdt_signal,
        rows_total=rows_total,
    )

    # پیام اصلی: همیشه
    send_telegram(main_msg)

    # هشدارها: تغییر وضعیت دلار/درهم (قبلی) + (اضافه) تغییر وضعیت تتر/دلار
    prev = load_prev_state() or {}
    prev_usd_state = prev.get("bubble_state")
    prev_usdt_state = prev.get("usdt_bubble_state")

    if prev_usd_state and prev_usd_state != usd_aed_state:
        alert = build_alert_change_message(
            title="USD/AED",
            prev_state=prev_usd_state,
            new_state=usd_aed_state,
            date_sh=date_sh,
            time_sh=time_sh,
            diff=diff_usd_r,
            pct=pct_usd_r,
        )
        send_telegram(alert)

    if prev_usdt_state and prev_usdt_state != usdt_state:
        alert = build_alert_change_message(
            title="USDT/USD",
            prev_state=prev_usdt_state,
            new_state=usdt_state,
            date_sh=date_sh,
            time_sh=time_sh,
            diff=diff_usdt_r,
            pct=pct_usdt_r,
        )
        send_telegram(alert)

    # ذخیره state برای اجرای بعد
    save_state({
        # قبلی
        "bubble_state": usd_aed_state,
        "last_date_shamsi": date_sh,
        "last_time": time_sh,
        "last_diff": diff_usd_r,
        "last_pct": pct_usd_r,

        # جدید
        "usdt_bubble_state": usdt_state,
        "last_usdt_diff": diff_usdt_r,
        "last_usdt_pct": pct_usdt_r,
        "last_usdt_signal": final_usdt_signal,
    })

    print(main_msg)


if __name__ == "__main__":
    main()
