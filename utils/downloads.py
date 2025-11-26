import os
import json
import time
import requests
from Bio import Entrez
from bs4 import BeautifulSoup
import re
from bs4 import BeautifulSoup, NavigableString


# ---- NEW: text extraction helpers ------------------------------------------

def _norm_ws(s: str) -> str:
    # collapse runs of whitespace but preserve paragraph breaks we add
    return re.sub(r"[ \t\f\v]+", " ", s).strip()


def _node_text(node) -> str:
    # get visible text without reference anchors/superscripts
    # (drop <xref>, <sup>, <sub> tags but keep their text if meaningful)
    for bad in node.find_all(["xref", "sup", "sub"]):
        bad.unwrap()
    return node.get_text(" ", strip=True)


def _join_blocks(blocks):
    # join with blank lines between logical blocks
    return "\n\n".join([_norm_ws(b) for b in blocks if b and _norm_ws(b)])


def extract_structured_abstract(article) -> str:
    """
    Prefer the true structured abstract (Importance/Objectives/Design/Interventions/
    Main Outcomes and Measures/Results/Conclusions) over Key Points.
    If multiple <abstract> tags exist, score them and pick the best.
    """
    abstracts = article.find_all("abstract")
    if not abstracts:
        return ""

    candidates = []
    for a in abstracts:
        parts = []

        # Prefer top-level <sec> children with <title> and <p>
        secs = a.find_all("sec", recursive=False)
        if secs:
            for sec in secs:
                title_tag = sec.find("title")
                title = title_tag.get_text(" ", strip=True) if title_tag else ""
                # paragraphs directly under this sec
                paras = sec.find_all("p", recursive=False)
                text = " ".join(_node_text(p) for p in paras) if paras else _node_text(sec)
                if title and text:
                    parts.append(f"{title}: {text}")
                elif text:
                    parts.append(text)
        else:
            # Fallback: paragraphs directly under <abstract>
            paras = a.find_all("p", recursive=False)
            if paras:
                parts.append(" ".join(_node_text(p) for p in paras))
            else:
                parts.append(_node_text(a))

        text = _join_blocks(parts)

        # score the candidate
        score = 0
        atype = (a.get("abstract-type") or "").lower()
        if "abstract" in atype:  # e.g., abstract-type="abstract" or "structured-abstract"
            score += 2
        # positive signals for structured abstract
        if any(h in text for h in [
            "Importance", "Objective", "Objectives", "Design",
            "Interventions", "Main Outcomes and Measures", "Results",
            "Conclusions", "Conclusions and Relevance"
        ]):
            score += 3
        # downweight Key Points
        if any(h in text for h in ["Question", "Findings", "Meaning", "Key Points"]):
            score -= 2

        candidates.append((score, len(text), text))

    # choose highest score, then longest
    candidates.sort(reverse=True)
    return candidates[0][2]


def extract_body_fulltext(article) -> str:
    """
    Build a readable plain-text body from <body>, keeping section titles and paragraphs.
    Skip Key Points, Abstract, References, Acknowledgments, Supplementary material,
    tables, figures, and ref-lists.
    """
    body = article.find("body")
    if not body:
        return ""

    SKIP_SEC_TYPES = {"abstract", "key-points", "graphical-abstract"}
    SKIP_TITLES = {
        "abstract", "key points", "key point", "references",
        "acknowledgments", "acknowledgements", "supplement", "supplementary material"
    }
    SKIP_TAGS = {"ref-list", "table-wrap", "fig", "fig-group", "table-wrap-foot", "disp-formula"}

    def collect_sec(sec):
        blocks = []
        # skip structural wrappers we don't want
        stype = (sec.get("sec-type") or "").lower()
        title_tag = sec.find("title", recursive=False)
        title_txt = title_tag.get_text(" ", strip=True).lower() if title_tag else ""

        if stype in SKIP_SEC_TYPES:
            return ""
        if title_txt in SKIP_TITLES:
            return ""

        # section title
        if title_tag:
            blocks.append(title_tag.get_text(" ", strip=True))

        # direct paragraphs
        for p in sec.find_all("p", recursive=False):
            blocks.append(_node_text(p))

        # child sections
        for child in sec.find_all("sec", recursive=False):
            child_text = collect_sec(child)
            if child_text:
                blocks.append(child_text)

        return _join_blocks(blocks)

    # Remove unwanted structures inside body to reduce noise
    for t in body.find_all(list(SKIP_TAGS)):
        t.decompose()

    top_blocks = []
    for top_sec in body.find_all("sec", recursive=False):
        txt = collect_sec(top_sec)
        if txt:
            top_blocks.append(txt)

    return _join_blocks(top_blocks)


# ---- END helpers ------------------------------------------------------------


# Identify yourself to Entrez (important for NCBI rate limits)
Entrez.email = "obafunsho.abiola@gmail.com"

# Directory for storing article metadata
OUTPUT_DIR = "data/original"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_pmids(file_path):
    """Reads a list of PMIDs (one per line) from a text file."""
    with open(file_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def retry_failed_pmids(failed_json_path="data/failed_pmids.json"):
    """
    Retry fetching articles that previously failed.
    Reads from data/failed_pmids.json and attempts to fetch them again.
    """
    if not os.path.exists(failed_json_path):
        print(f"❌ No failed PMIDs file found at {failed_json_path}")
        return

    with open(failed_json_path, "r") as f:
        failed_data = json.load(f)

    pmids_to_retry = [item["pmid"] for item in failed_data]

    if not pmids_to_retry:
        print("✅ No failed PMIDs to retry!")
        return

    print(f"🔄 Retrying {len(pmids_to_retry)} failed PMIDs...")

    # Create temporary file with failed PMIDs
    temp_file = "data/temp_retry_pmids.txt"
    with open(temp_file, "w") as f:
        for pmid in pmids_to_retry:
            f.write(f"{pmid}\n")

    # Fetch using main function
    fetch_pmc_from_pmids(temp_file)

    # Clean up temp file
    os.remove(temp_file)
    print("✅ Retry complete!")


def download_pmc_pdf(pmcid, output_folder="data/pdfs"):
    """Attempts to download a PDF version of a PMC article, if available."""
    os.makedirs(output_folder, exist_ok=True)
    pdf_path = os.path.join(output_folder, f"{pmcid}.pdf")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/115.0.0.0 Safari/537.36"
        )
    }

    base_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/PMC{pmcid}/"
    if os.path.exists(pdf_path):
        print(f"✅ PDF already exists for PMC{pmcid}")
        return {
            "pdf_url": f"{base_url}pdf/",
            "pdf_downloaded": True,
            "pdf_embedded_viewer": False,
        }

    try:
        page = requests.get(base_url, headers=headers, timeout=10)
        soup = BeautifulSoup(page.text, "html.parser")
        pdf_tag = soup.find("a", href=lambda h: h and "pdf" in h.lower())

        if not pdf_tag:
            print(f"❓ No PDF link found for PMC{pmcid}")
            return {"pdf_url": "", "pdf_downloaded": False, "pdf_embedded_viewer": False}

        href = pdf_tag["href"]
        pdf_url = href if href.startswith("http") else f"https://www.ncbi.nlm.nih.gov{href}"

        print(f"🔗 Trying PDF URL: {pdf_url}")
        response = requests.get(pdf_url, headers=headers, timeout=15)

        if response.status_code == 200 and "application/pdf" in response.headers.get("Content-Type", "").lower():
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"📄 PDF downloaded for PMC{pmcid}")
            return {"pdf_url": pdf_url, "pdf_downloaded": True, "pdf_embedded_viewer": False}
        else:
            print(f"⚠️ Could not download PDF for PMC{pmcid}")
            return {"pdf_url": pdf_url, "pdf_downloaded": False, "pdf_embedded_viewer": True}
    except Exception as e:
        print(f"❌ Exception for PMC{pmcid}: {e}")
        return {"pdf_url": "", "pdf_downloaded": False, "pdf_embedded_viewer": False}


def fetch_pmc_from_pmids(pmid_file, max_articles=None):
    """Fetch abstracts and metadata for PMIDs, map to PMCIDs, and save to JSON."""
    pmids = read_pmids(pmid_file)
    if max_articles:
        pmids = pmids[:max_articles]

    print(f"📋 Loaded {len(pmids)} PMIDs from file.")

    failed_pmids = []  # Track failed PMIDs for summary

    for idx, pmid in enumerate(pmids, start=1):
        try:
            print(f"\n🔍 ({idx}/{len(pmids)}) Fetching PMID {pmid}...")

            # Step 1: Map PMID → PMCID (via ELink)
            link = Entrez.elink(dbfrom="pubmed", db="pmc", id=pmid)
            record = Entrez.read(link)
            link.close()

            # Some records return multiple LinkSetDb blocks; pick the pmc one if present
            pmcid_list = []
            if record and record[0].get("LinkSetDb"):
                for ldb in record[0]["LinkSetDb"]:
                    if ldb.get("DbTo", "").lower() == "pmc" and ldb.get("Link"):
                        pmcid_list = ldb["Link"]
                        break

            if not pmcid_list:
                print(f"⚠️ No PMC link for PMID {pmid}. Skipping.")
                failed_pmids.append({"pmid": pmid, "pmcid": None, "reason": "No PMC version available"})
                continue

            pmcid = pmcid_list[0]["Id"]

            # Step 2: Fetch full article XML
            fetch = Entrez.efetch(db="pmc", id=pmcid, rettype="full", retmode="xml")
            xml_data = fetch.read()
            fetch.close()

            # Parse XML with multiple fallback strategies
            soup = None
            article = None

            # Strategy 1: Try lxml-xml parser (default, fastest but strictest)
            try:
                soup = BeautifulSoup(xml_data, "lxml-xml")
                article = soup.find("article")
            except Exception as parse_err:
                print(f"⚠️ lxml-xml parser failed for PMC{pmcid}: {parse_err}")
                soup = None

            # Strategy 2: Try html.parser (more forgiving)
            if soup is None or article is None:
                try:
                    print(f"   Trying html.parser fallback...")
                    soup = BeautifulSoup(xml_data, "html.parser")
                    article = soup.find("article")
                except Exception as parse_err:
                    print(f"⚠️ html.parser also failed: {parse_err}")
                    soup = None

            # Strategy 3: Try xml.parser (Python's built-in, most forgiving)
            if soup is None or article is None:
                try:
                    print(f"   Trying xml parser fallback...")
                    soup = BeautifulSoup(xml_data, "xml")
                    article = soup.find("article")
                except Exception as parse_err:
                    print(f"⚠️ xml parser also failed: {parse_err}")
                    soup = None

            # Strategy 4: Fetch abstract-only version from PubMed as last resort
            if soup is None or article is None:
                print(f"   All XML parsers failed. Trying PubMed abstract-only fallback...")
                try:
                    pubmed_fetch = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml")
                    pubmed_xml = pubmed_fetch.read()
                    pubmed_fetch.close()

                    soup = BeautifulSoup(pubmed_xml, "lxml-xml")
                    pubmed_article = soup.find("PubmedArticle")

                    if pubmed_article:
                        # Extract basic info from PubMed
                        title_elem = pubmed_article.find("ArticleTitle")
                        title = title_elem.get_text(" ", strip=True) if title_elem else ""

                        abstract_elem = pubmed_article.find("AbstractText")
                        abstract = abstract_elem.get_text(" ", strip=True) if abstract_elem else ""

                        journal_elem = pubmed_article.find("Title")  # Journal title
                        journal = journal_elem.text if journal_elem else ""

                        year_elem = pubmed_article.find("PubDate")
                        pub_year = ""
                        if year_elem:
                            y = year_elem.find("Year")
                            if y:
                                pub_year = y.text.strip()

                        authors = []
                        for author in pubmed_article.find_all("Author"):
                            fname = author.find("ForeName")
                            lname = author.find("LastName")
                            if fname and lname:
                                authors.append(f"{fname.text} {lname.text}".strip())

                        # Save minimal version with warning
                        structured = {
                            "pmcid": pmcid,
                            "pmid": pmid,
                            "title": title.strip(),
                            "abstract": abstract.strip(),
                            "authors": authors,
                            "journal": journal.strip(),
                            "year": pub_year,
                            "full_text": "",  # Not available in PubMed fallback
                            "pdf_url": "",
                            "pdf_downloaded": False,
                            "pdf_embedded_viewer": False,
                            "data_source": "pubmed_fallback",
                            "warning": "Full text not available - XML parsing failed, used PubMed abstract only"
                        }

                        out_path = os.path.join(OUTPUT_DIR, f"{pmcid}.json")
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(structured, f, indent=2, ensure_ascii=False)

                        print(f"⚠️ Saved PMC{pmcid}.json with PubMed abstract only (no full text)")
                        time.sleep(0.4)
                        continue  # Move to next PMID

                except Exception as pubmed_err:
                    print(f"❌ PubMed fallback also failed: {pubmed_err}")

                print(f"❌ All fetch strategies failed for PMC{pmcid}. Skipping.")
                failed_pmids.append({"pmid": pmid, "pmcid": pmcid, "reason": "All XML parsers failed"})
                continue

            if not article:
                print(f"⚠️ No <article> element found for PMC{pmcid} after all attempts")
                continue

            print(f"✅ Successfully parsed XML for PMC{pmcid}")

            title = article.find("article-title").get_text(" ", strip=True) if article.find("article-title") else ""
            abstract = extract_structured_abstract(article)
            journal = article.find("journal-title").text if article.find("journal-title") else ""

            # Be tolerant of multiple pub-date blocks
            pub_year = ""
            for pd in article.find_all("pub-date"):
                y = pd.find("year")
                if y and y.text:
                    pub_year = y.text.strip()
                    break

            authors = []
            for contrib in article.find_all("contrib", {"contrib-type": "author"}):
                name = contrib.find("name")
                if name:
                    forename = name.find("given-names").text if name.find("given-names") else ""
                    surname = name.find("surname").text if name.find("surname") else ""
                    authors.append(f"{forename} {surname}".strip())

            sections = article.find_all("sec")
            full_text = extract_body_fulltext(article)

            pdf_info = download_pmc_pdf(pmcid)

            structured = {
                "pmcid": pmcid,
                "pmid": pmid,
                "title": title.strip(),
                "abstract": abstract.strip(),
                "authors": authors,
                "journal": journal.strip(),
                "year": pub_year,
                "full_text": full_text.strip(),
                **pdf_info,
            }

            out_path = os.path.join(OUTPUT_DIR, f"{pmcid}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(structured, f, indent=2, ensure_ascii=False)

            print(f"✅ Saved PMC{pmcid}.json ({journal}, {pub_year})")
            time.sleep(0.4)

        except Exception as e:
            print(f"❌ Failed for PMID {pmid}: {e}")
            failed_pmids.append({"pmid": pmid, "pmcid": pmcid if 'pmcid' in locals() else None, "reason": str(e)})
            continue  # Continue with next PMID instead of crashing

    # Print summary
    print("\n" + "=" * 60)
    print(f"📊 FETCH SUMMARY")
    print("=" * 60)
    total = len(pmids)
    successful = total - len(failed_pmids)
    print(f"✅ Successfully fetched: {successful}/{total}")
    print(f"❌ Failed: {len(failed_pmids)}/{total}")

    if failed_pmids:
        print(f"\n⚠️ Failed PMIDs saved to: data/failed_pmids.json")
        # Save failed PMIDs for later retry
        failed_path = os.path.join("data", "failed_pmids.json")
        os.makedirs("data", exist_ok=True)
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_pmids, f, indent=2)

        print("\nFailed PMIDs breakdown:")
        reasons = {}
        for item in failed_pmids:
            reason = item["reason"]
            if reason not in reasons:
                reasons[reason] = []
            reasons[reason].append(item["pmid"])

        for reason, pmids_list in reasons.items():
            print(f"  • {reason}: {len(pmids_list)} articles")

    print("=" * 60)