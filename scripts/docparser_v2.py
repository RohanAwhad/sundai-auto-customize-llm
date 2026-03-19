#!/usr/bin/env python3
"""
PDF Document Processing and Export Tool

This script processes PDF documents using the docling library, performing OCR,
table detection, and exporting to multiple formats. Configuration is handled
through a YAML file, allowing flexible control over processing options.

Example Usage:
    # Using defaults
    python docparser_v2.py -i ./pdfs -o ./output

    # Using custom config
    python docparser_v2.py -i ./pdfs -o ./output -c config.yaml

See README.md for detailed configuration options and examples.
"""

# Standard
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import multiprocessing as mp
from pathlib import Path
import json
import time
from typing import Any, Optional
import yaml

# Third Party
from docling.datamodel.accelerator_options import (
    AcceleratorDevice,
    AcceleratorOptions,
)
from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    OcrAutoOptions,
    RapidOcrOptions,
    ThreadedPdfPipelineOptions,
)
from docling.datamodel.settings import settings
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.threaded_standard_pdf_pipeline import ThreadedStandardPdfPipeline
from loguru import logger
import click


# Constants and type definitions
EXPORT_FORMATS = {
    "json": ("json", "export_to_dict"),  # Deep Search JSON format
    "text": ("txt", "export_to_text"),  # Plain text
    "markdown": ("md", "export_to_markdown"),  # Markdown with structure
    "html": ("html", "export_to_html"),  # HTML with styling
    "doctags": ("doctags", "export_to_document_tokens"),  # Document tokens
}

DEFAULT_CONFIG = {
    "pipeline": {
        "ocr": {
            "enabled": True,  # Enable/disable OCR processing
            "languages": ["es"],  # List of language codes (e.g., eng, fra, deu)
            "engine": "rapidocr",  # OCR engine selection (rapidocr, easyocr, auto)
            "backend": "torch",  # RapidOCR backend
        },
        "tables": {
            "enabled": True,  # Enable/disable table detection
            "cell_matching": True,  # Enable/disable cell matching in tables
        },
        "performance": {
            "threads": 4,  # Number of processing threads
            "device": "auto",  # Default device when not sharding across GPUs
            "devices": None,  # Optional list like ["cuda:0", "cuda:1"] or "all"
            "page_batch_size": 32,  # Concurrent page processing inside Docling
            "layout_batch_size": 32,  # GPU batch size for layout inference
            "ocr_batch_size": 16,  # OCR batch size
            "table_batch_size": 4,  # Table detection batch size
        },
    },
    "export": {
        "formats": {
            "json": True,  # Deep Search JSON format
            "text": True,  # Plain text
            "markdown": True,  # Markdown with structure
            "html": True,  # HTML with styling
            "doctags": True,  # Document tokens
        }
    },
}


def deep_merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries while preserving nested defaults."""
    merged = deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value

    return merged


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load configuration from file or return defaults."""
    if not config_path:
        return deepcopy(DEFAULT_CONFIG)

    try:
        with config_path.open("r") as f:
            user_config = yaml.safe_load(f) or {}
            return deep_merge_dicts(DEFAULT_CONFIG, user_config)
    except Exception as e:
        logger.warning(f"Failed to load config file: {e}. Using defaults.")
        return deepcopy(DEFAULT_CONFIG)


def detect_cuda_device_count() -> int:
    """Return the number of visible CUDA devices."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0

        return torch.cuda.device_count()
    except Exception as exc:
        logger.warning(f"Unable to detect CUDA devices via torch: {exc}")
        return 0


def resolve_devices(config: dict) -> list[str]:
    """Resolve the configured device list for parallel document conversion."""
    pipeline_config = config["pipeline"]
    performance_config = pipeline_config["performance"]
    requested_devices = performance_config.get("devices")
    fallback_device = str(performance_config.get("device", "auto")).lower()

    if requested_devices is None:
        return [fallback_device]

    if isinstance(requested_devices, str):
        if requested_devices.lower() == "all":
            gpu_count = detect_cuda_device_count()
            if gpu_count == 0:
                fallback = fallback_device if fallback_device != "cuda" else "auto"
                logger.warning(
                    "Requested all GPUs but no CUDA devices were detected. "
                    f"Falling back to {fallback}."
                )
                return [fallback]

            return [f"cuda:{gpu_idx}" for gpu_idx in range(gpu_count)]

        return [requested_devices.lower()]

    return [str(device).lower() for device in requested_devices]


def build_ocr_options(config: dict):
    """Build OCR options from configuration."""
    ocr_config = config["pipeline"]["ocr"]
    languages = ocr_config["languages"]
    engine = str(ocr_config.get("engine", "rapidocr")).lower()

    if engine == "rapidocr":
        return RapidOcrOptions(
            lang=languages,
            backend=ocr_config.get("backend", "torch"),
        )

    if engine == "easyocr":
        return EasyOcrOptions(
            lang=languages,
            use_gpu=ocr_config.get("use_gpu"),
        )

    if engine == "auto":
        if languages:
            logger.warning(
                "OCR engine 'auto' ignores explicit language selection. "
                "Use 'rapidocr' or 'easyocr' to enforce OCR languages."
            )
        return OcrAutoOptions()

    raise ValueError(
        f"Unsupported OCR engine '{engine}'. Use one of: rapidocr, easyocr, auto."
    )


def setup_pipeline_options(config: dict, device: str) -> ThreadedPdfPipelineOptions:
    """Configure threaded pipeline options from config dictionary."""
    pipeline_config = config["pipeline"]
    performance_config = pipeline_config["performance"]

    pipeline_options = ThreadedPdfPipelineOptions(
        ocr_batch_size=performance_config["ocr_batch_size"],
        layout_batch_size=performance_config["layout_batch_size"],
        table_batch_size=performance_config["table_batch_size"],
    )
    pipeline_options.do_ocr = pipeline_config["ocr"]["enabled"]
    pipeline_options.do_table_structure = pipeline_config["tables"]["enabled"]
    pipeline_options.table_structure_options.do_cell_matching = pipeline_config[
        "tables"
    ]["cell_matching"]

    if pipeline_options.do_ocr:
        pipeline_options.ocr_options = build_ocr_options(config)

    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=performance_config["threads"],
        device=device,
    )

    settings.perf.page_batch_size = performance_config["page_batch_size"]

    return pipeline_options


def create_converter(config: dict, device: str) -> DocumentConverter:
    """Create a document converter for a specific device."""
    pipeline_options = setup_pipeline_options(config, device)

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_cls=ThreadedStandardPdfPipeline,
                pipeline_options=pipeline_options,
            )
        }
    )


def shard_file_paths(file_paths: list[Path], shard_count: int) -> list[list[Path]]:
    """Split file paths into balanced shards for parallel workers."""
    return [file_paths[index::shard_count] for index in range(shard_count)]


def convert_batch_on_device(
    device: str,
    file_paths: list[Path],
    output_dir: Path,
    config: dict,
) -> tuple[str, int, int]:
    """Convert a batch of PDFs on a single device and export the results."""
    if not file_paths:
        return device, 0, 0

    logger.info(f"[{device}] Starting conversion for {len(file_paths)} PDFs")
    doc_converter = create_converter(config, device)
    doc_converter.initialize_pipeline(InputFormat.PDF)

    success_count = failure_count = 0
    conversion_results = doc_converter.convert_all(file_paths, raises_on_error=False)

    for conv_result in conversion_results:
        file_path = Path(conv_result.input.file)

        if conv_result.status in {
            ConversionStatus.SUCCESS,
            ConversionStatus.PARTIAL_SUCCESS,
        } and conv_result.document is not None:
            try:
                export_document(conv_result, file_path.stem, output_dir, config)
                success_count += 1
                logger.info(f"[{device}] Successfully processed {file_path}")
            except Exception as exc:
                failure_count += 1
                logger.error(f"[{device}] Failed to export {file_path}: {exc}")
        else:
            failure_count += 1
            logger.error(f"[{device}] Failed to process {file_path}")

    return device, success_count, failure_count


def export_document(
    conv_result, doc_filename: str, output_dir: Path, config: dict
) -> None:
    """Export document in configured formats."""
    enabled_formats = {
        k: v
        for k, v in EXPORT_FORMATS.items()
        if config["export"]["formats"].get(k, True)
    }

    for format_name, (extension, export_method) in enabled_formats.items():
        try:
            content = getattr(conv_result.document, export_method)()
            output_path = output_dir / f"{doc_filename}.{extension}"

            with output_path.open("w", encoding="utf-8") as fp:
                if isinstance(content, (dict, list)):
                    json.dump(content, fp, ensure_ascii=False, indent=2)
                else:
                    fp.write(content)

            logger.debug(f"Successfully exported {format_name} format to {output_path}")

        except Exception as e:
            logger.error(f"Failed to export {format_name} format: {str(e)}")
            raise


@click.command()
@click.option(
    "--input-dir",
    "-i",
    type=click.Path(path_type=Path),
    help="Directory containing the documents to convert",
    required=True,
)
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    help="Directory to save the converted documents",
    required=True,
)
@click.option(
    "--config",
    "-c",
    type=click.Path(path_type=Path),
    help="Path to YAML configuration file",
    default=None,
)
def export_document_new_docling(
    input_dir: Path,
    output_dir: Path,
    config: Optional[Path],
):
    """Convert PDF documents and export them in multiple formats."""
    config_data = load_config(config)

    file_paths = sorted(input_dir.glob("*.pdf"))
    if not file_paths:
        logger.warning(f"No PDF files found in {input_dir}")
        return

    logger.info(f"Found {len(file_paths)} PDF files to process")
    devices = resolve_devices(config_data)
    logger.info(f"Using devices: {devices}")

    output_dir.mkdir(parents=True, exist_ok=True)
    success_count = failure_count = 0
    start_time = time.time()

    if len(devices) == 1:
        _, success_count, failure_count = convert_batch_on_device(
            devices[0], file_paths, output_dir, config_data
        )
    else:
        shards = shard_file_paths(file_paths, len(devices))
        worker_inputs = [
            (device, shard)
            for device, shard in zip(devices, shards, strict=False)
            if shard
        ]

        mp_context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=len(worker_inputs),
            mp_context=mp_context,
        ) as executor:
            futures = [
                executor.submit(
                    convert_batch_on_device,
                    device,
                    shard,
                    output_dir,
                    config_data,
                )
                for device, shard in worker_inputs
            ]

            for future in as_completed(futures):
                device, worker_successes, worker_failures = future.result()
                success_count += worker_successes
                failure_count += worker_failures
                logger.info(
                    f"[{device}] Finished with {worker_successes} successes "
                    f"and {worker_failures} failures"
                )

    processing_time = time.time() - start_time

    logger.info(
        f"Processed {success_count + failure_count} docs in {processing_time:.2f} seconds"
        f"\n  Successful: {success_count}"
        f"\n  Failed: {failure_count}"
    )


if __name__ == "__main__":
    try:
        export_document_new_docling()
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        raise
