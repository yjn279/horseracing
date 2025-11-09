"""Scrape race results from db.netkeiba.com for Tokyo Racecourse."""

from __future__ import annotations

import argparse
import datetime as dt
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://db.netkeiba.com"
SEARCH_URL = f"{BASE_URL}/"
RACE_URL = f"{BASE_URL}/race/"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
TOKYO_CODE = "05"
RESULT_ID_PATTERN = re.compile(r"/race/(\d{12})/")
COURSE_PATTERN = re.compile(r"(東京|札幌|函館|福島|新潟|中山|中京|京都|阪神|小倉)")
DATE_PATTERN = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")


def create_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT, "Referer": BASE_URL})
    return session


def build_search_params(start: dt.date, end: dt.date, page: int) -> list[tuple[str, str]]:
    return [
        ("pid", "race_search_detail"),
        ("start_year", str(start.year)),
        ("start_mon", str(start.month)),
        ("start_day", str(start.day)),
        ("end_year", str(end.year)),
        ("end_mon", str(end.month)),
        ("end_day", str(end.day)),
        ("jyo[]", TOKYO_CODE),
        ("list", "100"),
        ("page", str(page)),
        ("sort", "20"),
    ]


def fetch_race_ids(
    session: requests.Session,
    start: dt.date,
    end: dt.date,
    pause_seconds: float,
) -> list[str]:
    race_ids: list[str] = []
    page = 1
    while True:
        response = session.get(SEARCH_URL, params=build_search_params(start, end, page), timeout=30)
        response.raise_for_status()
        response.encoding = "EUC-JP"
        page_ids = list(dict.fromkeys(RESULT_ID_PATTERN.findall(response.text)))
        page_ids = [race_id for race_id in page_ids if race_id not in race_ids]
        if not page_ids:
            break
        race_ids.extend(page_ids)
        if "NK_PAGINATION__ITEM--next is-disable" in response.text:
            break
        page += 1
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return race_ids


def to_snake_case(label: str) -> str:
    text = label.strip()
    text = re.sub(r"[\s\u3000]+", "_", text)
    text = re.sub(r"[^0-9A-Za-z_]+", "", text)
    return text.lower()


def parse_course_and_date(soup: BeautifulSoup) -> tuple[Optional[str], Optional[dt.date]]:
    intro = soup.select_one("div.data_intro p")
    if not intro:
        return None, None
    text = intro.get_text(" ", strip=True)
    course_match = COURSE_PATTERN.search(text)
    date_match = DATE_PATTERN.search(text)
    course = course_match.group(1) if course_match else None
    if not date_match:
        return course, None
    year, month, day = map(int, date_match.groups())
    return course, dt.date(year, month, day)


def parse_race_name(soup: BeautifulSoup) -> Optional[str]:
    title = soup.select_one("div.race_data h1") or soup.select_one("div.race_name h1")
    return title.get_text(strip=True) if title else None


def fetch_race_result(session: requests.Session, race_id: str) -> pd.DataFrame:
    response = session.get(f"{RACE_URL}{race_id}/", timeout=30)
    response.raise_for_status()
    response.encoding = "EUC-JP"
    soup = BeautifulSoup(response.text, "lxml")
    tables = pd.read_html(response.text)
    if not tables:
        raise ValueError(f"No race result table found for {race_id}")
    frame = tables[0]
    frame.columns = [to_snake_case(str(column)) for column in frame.columns]
    race_name = parse_race_name(soup)
    course, race_date = parse_course_and_date(soup)
    frame["race_id"] = race_id
    if race_name:
        frame["race_name"] = race_name
    if race_date:
        frame["race_date"] = race_date.isoformat()
    if course:
        frame["course"] = course
    return frame


def collect_race_results(
    session: requests.Session,
    race_ids: Iterable[str],
    pause_seconds: float,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for race_id in race_ids:
        frames.append(fetch_race_result(session, race_id))
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def default_dates() -> tuple[dt.date, dt.date]:
    end = dt.date.today()
    return end - dt.timedelta(days=365), end


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    start, end = default_dates()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start",
        type=lambda value: dt.datetime.strptime(value, "%Y-%m-%d").date(),
        default=start,
        help="Start date (YYYY-MM-DD). Defaults to one year ago.",
    )
    parser.add_argument(
        "--end",
        type=lambda value: dt.datetime.strptime(value, "%Y-%m-%d").date(),
        default=end,
        help="End date (YYYY-MM-DD). Defaults to today.",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Pause between requests in seconds.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tokyo_races.csv"),
        help="Destination CSV path.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    if args.start > args.end:
        raise ValueError("start date must be on or before end date")
    session = create_session()
    race_ids = fetch_race_ids(session, args.start, args.end, args.pause)
    data = collect_race_results(session, race_ids, args.pause)
    if data.empty:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
