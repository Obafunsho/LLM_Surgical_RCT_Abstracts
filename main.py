# main.py - Example with retry functionality

import os
import hashlib
import datetime

from datetime import datetime, timezone

# Local helper functions for file I/O and PMC processing
from utils.helpers import load_json, save_json, list_json_files
from utils.downloads import fetch_pmc_from_pmids, retry_failed_pmids
from utils.scoring import score_readability, score_consort
from utils.rewriting import rewrite_abstract

# Configuration values: file paths
from config import (
    INPUT_DIR_ORIGINAL,
    INPUT_DIR_REWRITTEN,
    OUTPUT_DIR_ORIGINAL,
    OUTPUT_DIR_REWRITTEN,
)


# Compute SHA-256 hash of a given text for provenance
def compute_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Return current UTC timestamp in ISO format
def utc_timestamp():
    return datetime.now(timezone.utc).isoformat()


# Main function to handle abstract scoring or rewriting
def process_directory(input_dir, output_dir, rewrite=False):
    """
    Loop through each JSON file in a directory, optionally rewrite it,
    and/or score using readability and CONSORT-based metrics.

    Parameters:
    - input_dir (str): Path to folder containing input JSON files
    - output_dir (str or None): Path to save scores (None if rewrite=True)
    - rewrite (bool): If True, run GPT rewrite and save versions
    """

    # Iterate through each abstract JSON file in the directory
    for filename in list_json_files(input_dir):
        # Full file path to the input abstract
        input_path = os.path.join(input_dir, filename)

        # Load the JSON data (should contain at least abstract text)
        abstract_data = load_json(input_path)

        # Extract the text of the abstract
        original_abstract = abstract_data.get("abstract", "")

        # Use provided pmcid or infer it from the filename
        pmcid = abstract_data.get("pmcid", filename.replace(".json", ""))

        # --- REWRITING MODE ---
        if rewrite:
            # Use GPT to rewrite abstract into 250-word and 300-word versions
            full_text = abstract_data.get("full_text", "")

            # Check if this is a PubMed fallback (no full text)
            if abstract_data.get("data_source") == "pubmed_fallback":
                print(f"⚠️ Skipping rewrite for {pmcid} - PubMed fallback (no full text available)")
                continue

            rewritten_250, rewritten_300 = rewrite_abstract(full_text, original_abstract)

            # Save each rewritten version separately in rewritten/ folder
            for version, content in [("250", rewritten_250), ("300", rewritten_300)]:
                version_path = os.path.join(INPUT_DIR_REWRITTEN, f"{pmcid}_{version}.json")
                word_count = len(content.split())
                save_json({"pmcid": pmcid, "abstract": content, "word_count": word_count}, version_path)
                print(f"✅ Saved rewritten abstract {pmcid}_{version}.json ({word_count} words)")

        # --- SCORING MODE ---
        else:
            # Abstract to be scored (original or rewritten)
            abstract_text = original_abstract

            # Compute SHA-256 and current UTC timestamp for traceability
            sha = compute_sha256(abstract_text)
            timestamp = utc_timestamp()

            # Get readability scores (Flesch, word count, etc.)
            readability = score_readability(abstract_text)

            # Get CONSORT checklist scores using GPT + fallback model
            consort_scores, model_used = score_consort(abstract_text, return_model=True)

            # Combine all results into a single output dictionary
            result = {
                "pmcid": pmcid,
                "sha256": sha,
                "timestamp_utc": timestamp,
                "model_used": model_used,
                "readability": readability,
                "consort_scores": consort_scores
            }

            # Add flag if this was from PubMed fallback
            if abstract_data.get("data_source") == "pubmed_fallback":
                result["data_source"] = "pubmed_fallback"
                result["note"] = "Scored from PubMed abstract only (no full text)"

            # Save result as JSON in appropriate scoring directory
            output_path = os.path.join(output_dir, f"{pmcid}_scores.json")
            save_json(result, output_path)
            print(f"📊 Scored {pmcid} → saved to {output_path}")


# Entry point: only runs if this file is executed directly
if __name__ == "__main__":
    # STEP 1: Fetch abstracts from PMC and save to data/original/
    print("\n" + "=" * 60)
    print("STEP 1: FETCHING ARTICLES FROM PMC")
    print("=" * 60)
    fetch_pmc_from_pmids("pmid-SurgicalPr-set.txt", max_articles=None)

    # OPTIONAL: Retry any failed PMIDs from previous runs
    print("\n" + "=" * 60)
    print("STEP 1b: RETRYING PREVIOUSLY FAILED ARTICLES (if any)")
    print("=" * 60)
    retry_failed_pmids()

    # STEP 2: Score the original abstracts (no rewriting)
    print("\n" + "=" * 60)
    print("STEP 2: SCORING ORIGINAL ABSTRACTS")
    print("=" * 60)
    process_directory(
        input_dir=INPUT_DIR_ORIGINAL,
        output_dir=OUTPUT_DIR_ORIGINAL,
        rewrite=False
    )

    # STEP 3: Rewrite each abstract into two versions using GPT
    print("\n" + "=" * 60)
    print("STEP 3: REWRITING ABSTRACTS (250 & 300 words)")
    print("=" * 60)
    process_directory(
        input_dir=INPUT_DIR_ORIGINAL,
        output_dir=None,  # Not needed here since we're just saving rewrites
        rewrite=True
    )

    # STEP 4: Score rewritten abstracts (250-word and 300-word separately)
    print("\n" + "=" * 60)
    print("STEP 4: SCORING REWRITTEN ABSTRACTS")
    print("=" * 60)
    for version in ["250", "300"]:
        print(f"\n📝 Processing {version}-word versions...")

        # Folder containing all rewritten abstracts
        rewritten_dir = INPUT_DIR_REWRITTEN

        # Output subdirectory for that version's scores
        scored_output_dir = os.path.join(OUTPUT_DIR_REWRITTEN, version)
        os.makedirs(scored_output_dir, exist_ok=True)


        # List only JSONs matching that version suffix
        def is_version_file(f):
            return f.endswith(f"_{version}.json")


        # Filter relevant rewritten files and score them
        files = [f for f in list_json_files(rewritten_dir) if is_version_file(f)]
        for file in files:
            input_path = os.path.join(rewritten_dir, file)
            abstract_data = load_json(input_path)
            abstract_text = abstract_data.get("abstract", "")
            pmcid = abstract_data.get("pmcid", file.replace(".json", ""))

            sha = compute_sha256(abstract_text)
            timestamp = utc_timestamp()
            readability = score_readability(abstract_text)
            consort_scores, model_used = score_consort(abstract_text, return_model=True)

            result = {
                "pmcid": pmcid,
                "sha256": sha,
                "timestamp_utc": timestamp,
                "model_used": model_used,
                "readability": readability,
                "consort_scores": consort_scores
            }

            output_path = os.path.join(scored_output_dir, f"{pmcid}_scores.json")
            save_json(result, output_path)
            print(f"📊 Scored {pmcid} → saved to {output_path}")

    print("\n" + "=" * 60)
    print("✅ ALL STEPS COMPLETED!")
    print("=" * 60)
    print("\n📁 Output locations:")
    print(f"  - Original abstracts: {INPUT_DIR_ORIGINAL}")
    print(f"  - Original scores: {OUTPUT_DIR_ORIGINAL}")
    print(f"  - Rewritten abstracts: {INPUT_DIR_REWRITTEN}")
    print(f"  - Rewritten scores: {OUTPUT_DIR_REWRITTEN}")
    print(f"  - Failed PMIDs: data/failed_pmids.json (if any)")
