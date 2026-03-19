#!/usr/bin/env python3
"""
Document Pre-processing Script for Knowledge Tuning

This script processes multiple markdown files and generates seed data for SDG-Hub knowledge pipeline.

It performs the following steps:
1. Reads all markdown files from a specified directory
2. Chunks each document while preserving markdown structure
3. Creates seed data with ICL (In-Context Learning) templates
4. Saves the dataset as a JSONL file

Usage:
    uv run scripts/generate_seed_data.py \
      --input-dir data/research_papers_md/ \
      --output data/seed_data.jsonl \
      --max-tokens 5000 \
      --overlap 1000 \
      --domain "Machine Learning Research"
"""

import argparse
import glob
from pathlib import Path
from typing import List, Dict, Any

import datasets
from loguru import logger
from markdown_it import MarkdownIt


def chunk_markdown(text: str, max_tokens: int = 200, overlap: int = 50) -> List[str]:
    """
    Splits Markdown text into chunks at block-level elements
    (headings, paragraphs, lists, tables, code, blockquotes).
    Adds overlap (in words) between all consecutive chunks.

    Args:
        text: The markdown text to be chunked
        max_tokens: Maximum number of words per chunk
        overlap: Number of overlapping words between consecutive chunks

    Returns:
        List of text chunks with specified overlap
    """
    # Initialize markdown parser to understand document structure
    md = MarkdownIt()
    tokens = md.parse(text)

    # Group tokens into block-level segments to preserve markdown structure
    # This ensures we don't split in the middle of headings, lists, etc.
    blocks = []
    buf = []
    for tok in tokens:
        if tok.block and tok.type.endswith("_open"):
            buf = []
        elif tok.block and tok.type.endswith("_close"):
            if buf:
                blocks.append("\n".join(buf).strip())
                buf = []
        elif tok.content:
            buf.append(tok.content)
    if buf:
        blocks.append("\n".join(buf).strip())

    # Split blocks into chunks with overlap to maintain context continuity
    chunks = []
    current_words = []
    for block in blocks:
        words = block.split()
        for w in words:
            current_words.append(w)
            if len(current_words) >= max_tokens:
                # Emit a complete chunk
                chunks.append(" ".join(current_words))
                # Prepare next buffer with overlap from the end of this chunk
                # This ensures context continuity between chunks
                current_words = current_words[-overlap:] if overlap > 0 else []

    # Add any remaining words as the final chunk
    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def process_markdown_files(
    input_dir: str,
    max_tokens: int = 5000,
    overlap: int = 1000,
) -> List[Dict[str, str]]:
    """
    Process all markdown files in the input directory and chunk them.

    Args:
        input_dir: Directory containing markdown files
        max_tokens: Maximum number of words per chunk
        overlap: Number of overlapping words between consecutive chunks

    Returns:
        List of dictionaries containing document chunks and metadata
    """
    input_path = Path(input_dir)
    md_files = list(input_path.glob("*.md"))

    if not md_files:
        logger.warning(f"No markdown files found in {input_dir}")
        return []

    logger.info(f"Found {len(md_files)} markdown files to process")

    all_chunks = []
    for md_file in md_files:
        logger.info(f"Processing {md_file.name}...")

        with open(md_file, "r", encoding="utf-8") as f:
            text = f.read()

        chunks = chunk_markdown(text, max_tokens=max_tokens, overlap=overlap)

        # Add metadata to each chunk
        for chunk in chunks:
            all_chunks.append({
                "document": chunk,
                "source_file": md_file.name,
            })

        logger.info(f"  Created {len(chunks)} chunks from {md_file.name}")

    logger.info(f"Total chunks created: {len(all_chunks)}")
    return all_chunks


def create_seed_data(
    chunks: List[Dict[str, str]],
    icl_config: Dict[str, str],
) -> datasets.Dataset:
    """
    Create seed data for SDG-Hub knowledge pipeline.

    Args:
        chunks: List of document chunks with metadata
        icl_config: ICL (In-Context Learning) configuration containing:
            - document_outline: Concise title/summary
            - icl_document: Representative sample extract
            - icl_query_1, icl_query_2, icl_query_3: Example queries
            - domain: Subject area of the document

    Returns:
        HuggingFace Dataset with seed data
    """
    # Create dataset from chunks
    seed_data = datasets.Dataset.from_list(chunks)

    # Add ICL fields to each entry
    seed_data = seed_data.map(lambda x: icl_config)

    return seed_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate seed data for knowledge tuning from markdown documents"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/research_papers_md/",
        help="Directory containing markdown files (default: data/research_papers_md/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seed_data.jsonl",
        help="Output JSONL file path (default: seed_data.jsonl)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum number of words per chunk (default: 5000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=1000,
        help="Number of overlapping words between chunks (default: 1000)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        default="Machine Learning Research",
        help="Domain of the documents (default: Machine Learning Research)",
    )

    args = parser.parse_args()

    logger.info("Starting document preprocessing pipeline")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output file: {args.output}")
    logger.info(f"Chunk size: {args.max_tokens} tokens with {args.overlap} overlap")

    # Process markdown files
    chunks = process_markdown_files(
        input_dir=args.input_dir,
        max_tokens=args.max_tokens,
        overlap=args.overlap,
    )

    if not chunks:
        logger.error("No chunks created. Exiting.")
        return

    # Define ICL configuration for seed data
    # TODO: Customize this based on your specific use case
    icl_config = {
        "document_outline": "Collection of machine learning and AI research papers covering various topics including reasoning, reinforcement learning, model training, and agent systems",
        "icl_document": """## Overview

This document contains research papers and technical reports on advanced machine learning topics.

Topics covered include:
- Large Language Model training and post-training
- Reinforcement Learning techniques and algorithms
- Multi-agent systems and collaboration
- Reasoning and knowledge grounding
- Model efficiency and optimization

The papers represent cutting-edge research in artificial intelligence and machine learning,
providing insights into state-of-the-art techniques and methodologies.""",
        "icl_query_1": "What are the key techniques for post-training large language models?",
        "icl_query_2": "How do reinforcement learning methods improve LLM reasoning capabilities?",
        "icl_query_3": "What are the challenges in building multi-agent systems with language models?",
        "domain": args.domain,
    }

    logger.info("Creating seed data with ICL templates")
    seed_data = create_seed_data(chunks, icl_config)

    # Save to JSONL
    logger.info(f"Saving seed data to {args.output}")
    seed_data.to_json(args.output, orient="records", lines=True)

    logger.success(f"Successfully created seed data with {len(seed_data)} entries")
    logger.info(f"Dataset features: {seed_data.features}")


if __name__ == "__main__":
    main()
