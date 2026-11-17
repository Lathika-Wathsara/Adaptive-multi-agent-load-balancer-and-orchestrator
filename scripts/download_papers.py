"""Verify data/papers.csv against arXiv and download PDFs + LaTeX sources (B1).

    python3 scripts/download_papers.py --verify     # fill licence, check titles
    python3 scripts/download_papers.py --download   # PDFs -> data/papers/pdf, sources -> data/papers/src
    python3 scripts/download_papers.py              # both

Everything lands under data/papers/, which is git-ignored. Be polite to arXiv: one
request every 3 s, and set CONTACT_EMAIL so the User-Agent identifies you.
Run it where the network allows arxiv.org (it does not need a GPU).
"""
import argparse
import csv
import difflib
import html
import os
import re
import sys
import time
from pathlib import Path

import httpx

OAI = [  # current endpoint first, legacy one as fallback
    "https://oaipmh.arxiv.org/oai?verb=GetRecord&identifier=oai:arXiv.org:{aid}&metadataPrefix=arXiv",
    "https://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:{aid}&metadataPrefix=arXiv",
]
PDF = "https://export.arxiv.org/pdf/{aid}"
SRC = "https://export.arxiv.org/e-print/{aid}"
DELAY = 3.0
HEADERS = {"User-Agent": f"paper-reviewer-course-project (mailto:{os.environ.get('CONTACT_EMAIL', 'unset')})"}


def fetch(client: httpx.Client, url: str, tries: int = 4) -> httpx.Response:
    for k in range(tries):
        r = client.get(url, headers=HEADERS, follow_redirects=True, timeout=120)
        if r.status_code in (429, 503) and k < tries - 1:
            time.sleep(float(r.headers.get("Retry-After", 10)))
            continue
        r.raise_for_status()
        time.sleep(DELAY)
        return r
    raise RuntimeError(f"giving up on {url}")


def licence_name(url: str) -> str:
    if not url:
        return "arXiv non-exclusive"
    m = re.search(r"/licenses/([a-z\-]+)/([\d.]+)", url)
    if m:
        return f"CC {m.group(1).upper()} {m.group(2)}"
    if "publicdomain/zero" in url:
        return "CC0"
    return url


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


def verify(rows, client) -> None:
    for r in rows:
        xml = ""
        for tmpl in OAI:
            try:
                xml = fetch(client, tmpl.format(aid=r["arxiv_id"])).text
            except httpx.HTTPError:
                continue
            if "<OAI-PMH" in xml:
                break
        t = re.search(r"<title>(.*?)</title>", xml, re.S)
        lic = re.search(r"<license>(.*?)</license>", xml, re.S)
        if "idDoesNotExist" in xml or not t:
            r["verified"] = "NOT FOUND"
            print(f"[{r['id']}] {r['arxiv_id']}: NOT FOUND on arXiv")
            continue
        real = html.unescape(re.sub(r"\s+", " ", t.group(1))).strip()
        ratio = difflib.SequenceMatcher(None, norm(real), norm(r["title"])).ratio()
        r["licence"] = licence_name(lic.group(1).strip() if lic else "")
        r["verified"] = "ok" if ratio >= 0.85 else "TITLE MISMATCH"
        flag = "" if ratio >= 0.85 else f"\n      csv:   {r['title']}\n      arxiv: {real}"
        print(f"[{r['id']}] {r['arxiv_id']}  {r['licence']:22s} {r['verified']}{flag}")


def download(rows, client, pdf_dir: Path, src_dir: Path) -> None:
    pdf_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    for r in rows:
        pdf, src = pdf_dir / f"{r['id']}.pdf", src_dir / f"{r['id']}.tar.gz"
        if not pdf.exists():
            data = fetch(client, PDF.format(aid=r["arxiv_id"])).content
            if data[:5] != b"%PDF-":
                print(f"[{r['id']}] not a PDF, skipped")
            else:
                pdf.write_bytes(data)
                print(f"[{r['id']}] pdf  {len(data) // 1024} KB")
        if not src.exists():
            try:
                data = fetch(client, SRC.format(aid=r["arxiv_id"])).content
                if data[:5] == b"%PDF-":                      # PDF-only submission: no LaTeX source
                    print(f"[{r['id']}] no LaTeX source (PDF-only)")
                else:
                    src.write_bytes(data)
                    print(f"[{r['id']}] src  {len(data) // 1024} KB")
            except httpx.HTTPError as exc:
                print(f"[{r['id']}] source unavailable: {exc}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/papers.csv")
    ap.add_argument("--out", default="data/papers")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--ids", nargs="*")
    args = ap.parse_args()
    do_verify = args.verify or not (args.verify or args.download)
    do_download = args.download or not (args.verify or args.download)

    with open(args.csv, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)
    sel = [r for r in rows if not args.ids or r["id"] in args.ids]
    with httpx.Client() as client:
        if do_verify:
            verify(sel, client)
            with open(args.csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            bad = [r["id"] for r in sel if r["verified"] != "ok"]
            print(f"verified {len(sel) - len(bad)}/{len(sel)}" + (f"; fix: {bad}" if bad else ""))
        if do_download:
            download(sel, client, Path(args.out) / "pdf", Path(args.out) / "src")
    return 0


if __name__ == "__main__":
    sys.exit(main())
