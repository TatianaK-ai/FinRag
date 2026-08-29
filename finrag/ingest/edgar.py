"""SEC EDGAR fetch, rate-limited and cached."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from ..config import CORPUS, FORMS, P, PER_FORM, SEC_USER_AGENT

# SEC fair-access policy: <=10 req/s and a User-Agent identifying a real contact.
SPACING_S = 0.15
_last = 0.0


def _sec_get(url: str) -> requests.Response:
    global _last
    wait = max(0.0, _last + SPACING_S - time.time())
    if wait:
        time.sleep(wait)
    _last = time.time()
    r = requests.get(url, headers={"User-Agent": SEC_USER_AGENT,
                                   "Accept-Encoding": "gzip, deflate"}, timeout=60)
    r.raise_for_status()
    return r


def list_filings(company) -> list[dict]:
    j = _sec_get(f"https://data.sec.gov/submissions/CIK{company.cik}.json").json()
    r = j["filings"]["recent"]
    counts: dict[str, int] = {}
    out: list[dict] = []
    for i, form in enumerate(r["form"]):
        if form not in FORMS:
            continue
        counts[form] = counts.get(form, 0) + 1
        if counts[form] > PER_FORM.get(form, 1):
            continue
        acc = r["accessionNumber"][i]
        cik_num = str(int(company.cik))          # archive paths drop leading zeros
        out.append({
            "ticker": company.ticker, "company": company.name, "form": form,
            "filingDate": r["filingDate"][i],
            "periodOfReport": (r.get("reportDate") or [None] * len(r["form"]))[i],
            "accession": acc, "primaryDocument": r["primaryDocument"][i],
            "sourceUrl": f"https://www.sec.gov/Archives/edgar/data/{cik_num}/"
                         f"{acc.replace('-', '')}/{r['primaryDocument'][i]}",
            "docId": f"{company.ticker}_{form.replace('-', '')}_{r['filingDate'][i]}",
        })
    return out


def download_filing(f: dict) -> dict:
    P.raw.mkdir(parents=True, exist_ok=True)
    dest: Path = P.raw / f"{f['docId']}.htm"
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached  {f['docId']}")
        return {**f, "rawPath": str(dest)}
    html = _sec_get(f["sourceUrl"]).text
    dest.write_text(html, encoding="utf-8")
    print(f"  fetched {f['docId']}  {len(html) / 1024 / 1024:.2f} MB")
    return {**f, "rawPath": str(dest)}


def fetch_corpus() -> list[dict]:
    manifest: list[dict] = []
    for company in CORPUS:
        for f in list_filings(company):
            manifest.append(download_filing(f))
    return manifest
