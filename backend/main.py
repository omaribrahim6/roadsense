"""RoadSense CLI - road damage detection pipeline"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Tuple

import click

#add backend directory to path
backend_dir = Path(__file__).parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from config import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_FPS,
    YOLO_MODEL_PATH,
    DEDUP_RADIUS_METERS,
)
from pipeline.extractor import extract_frames, list_image_files, extract_gps_from_image
from pipeline.inference import run_inference
from pipeline.severity import calculate_severity
from pipeline.gps import simulate_gps, parse_gpx_route, DEMO_ROUTE_KUWAIT, DEMO_ROUTE_SF
from pipeline.dedup import deduplicate, EnrichedDetection
from utils.logger import setup_logger, get_logger

#initialize logger
setup_logger()
logger = get_logger("roadsense")


@click.group()
@click.version_option(version="1.0.0")
def cli():
    """RoadSense - road damage detection pipeline"""
    pass


@cli.command()
@click.option("--input", "-i", "input_path", required=True, help="video file or frames directory")
@click.option("--source-id", "-s", required=True, help="unique identifier for this run")
@click.option("--gpx", "gpx_path", default=None, help="gpx file for gps matching")
@click.option("--fps", default=DEFAULT_FPS, help=f"frame extraction rate (default: {DEFAULT_FPS})")
@click.option("--conf-threshold", default=DEFAULT_CONFIDENCE_THRESHOLD, help="min detection confidence")
@click.option("--dry-run", is_flag=True, help="output json without uploading")
@click.option("--output", "-o", "output_path", default=None, help="output file for dry run")
@click.option("--no-dedup", is_flag=True, help="skip deduplication")
@click.option("--route", type=click.Choice(["kuwait", "sf", "custom"]), default="kuwait", help="demo route")
def process(
    input_path: str,
    source_id: str,
    gpx_path: Optional[str],
    fps: int,
    conf_threshold: float,
    dry_run: bool,
    output_path: Optional[str],
    no_dedup: bool,
    route: str
):
    """process video or images for road damage detection"""
    logger.info("=" * 60)
    logger.info("RoadSense Road Damage Detection Pipeline")
    logger.info("=" * 60)
    
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)
    
    #step 1: prepare frames
    logger.info("\n[1/6] Preparing frames...")
    
    frames_with_timestamps: List[Tuple[str, float]] = []
    
    if input_path.is_file():
        #video file - extract frames
        video_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        
        if input_path.suffix.lower() in video_extensions:
            temp_dir = Path("output") / f"frames_{source_id}"
            frames_with_timestamps = extract_frames(
                str(input_path),
                str(temp_dir),
                fps=fps
            )
        else:
            #single image file
            logger.info(f"Processing single image: {input_path}")
            frames_with_timestamps = [(str(input_path), 0.0)]
    else:
        #directory of images
        logger.info(f"Scanning directory: {input_path}")
        image_files = list_image_files(str(input_path))
        
        #assign sequential timestamps
        frames_with_timestamps = [
            (path, float(i)) for i, path in enumerate(image_files)
        ]
    
    if not frames_with_timestamps:
        logger.error("No frames to process!")
        sys.exit(1)
    
    logger.info(f"  Total frames: {len(frames_with_timestamps)}")
    
    #extract paths and timestamps
    frame_paths = [f[0] for f in frames_with_timestamps]
    timestamps = [f[1] for f in frames_with_timestamps]
    video_duration = max(timestamps) if timestamps else 60.0
    
    #step 2: run yolo inference
    logger.info("\n[2/6] Running YOLO inference...")
    
    raw_detections = run_inference(
        frame_paths=frame_paths,
        conf_threshold=conf_threshold,
        timestamps=timestamps
    )
    
    if not raw_detections:
        logger.warning("No detections found!")
        if dry_run:
            _output_results([], output_path)
        sys.exit(0)
    
    logger.info(f"  Raw detections: {len(raw_detections)}")
    
    #step 3: calculate severity
    logger.info("\n[3/6] Calculating severity scores...")
    
    detections_with_severity = []
    for det in raw_detections:
        severity = calculate_severity(det)
        detections_with_severity.append((det, severity))
    
    #step 4: resolve gps coordinates
    logger.info("\n[4/6] Resolving GPS coordinates...")
    
    gps_route = None
    gps_duration = video_duration
    
    if gpx_path:
        logger.info(f"  Using GPX file: {gpx_path}")
        try:
            gps_route, gps_duration = parse_gpx_route(gpx_path)
            logger.info(f"  GPX route: {len(gps_route)} points, {gps_duration:.1f}s")
        except Exception as e:
            logger.warning(f"  Failed to parse GPX: {e}, falling back to simulation")
    
    if gps_route is None:
        if route == "sf":
            gps_route = DEMO_ROUTE_SF
        else:
            gps_route = DEMO_ROUTE_KUWAIT
        logger.info(f"  Using simulated route: {route}")
    
    #enrich detections with gps
    enriched_detections: List[EnrichedDetection] = []
    
    for det, severity in detections_with_severity:
        #try to extract gps from image exif first
        exif_gps = extract_gps_from_image(det.frame_path)
        
        if exif_gps:
            lat, lng = exif_gps
            logger.debug(f"  GPS from EXIF: {lat}, {lng}")
        else:
            #simulate gps based on timestamp
            lat, lng = simulate_gps(det.timestamp, gps_route, gps_duration)
        
        enriched = EnrichedDetection(
            frame_path=det.frame_path,
            timestamp=det.timestamp,
            damage_type=det.damage_type,
            confidence=det.confidence,
            severity=severity,
            bbox=det.bbox,
            lat=lat,
            lng=lng,
            frame_width=det.frame_width,
            frame_height=det.frame_height,
            original_class=det.original_class,
        )
        enriched_detections.append(enriched)
    
    logger.info(f"  Enriched: {len(enriched_detections)} detections with GPS")
    
    #step 5: deduplicate
    logger.info("\n[5/6] Deduplicating detections...")
    
    if no_dedup:
        final_detections = enriched_detections
        logger.info("  Deduplication skipped (--no-dedup)")
    else:
        final_detections = deduplicate(enriched_detections, DEDUP_RADIUS_METERS)
    
    logger.info(f"  Final: {len(final_detections)} unique detections")
    
    #step 6: upload or output
    logger.info("\n[6/6] Outputting results...")
    
    if dry_run:
        _output_results(final_detections, output_path)
        logger.info("  Dry run complete - no upload to Supabase")
    else:
        _upload_to_supabase(final_detections, source_id)
    
    #print summary
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    _print_summary(final_detections)


def _output_results(detections: List[EnrichedDetection], output_path: Optional[str]):
    """output detections as json"""
    results = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "count": len(detections),
        "detections": [d.to_dict() for d in detections]
    }
    
    json_str = json.dumps(results, indent=2)
    
    if output_path:
        Path(output_path).write_text(json_str)
        logger.info(f"  Results written to: {output_path}")
    else:
        print(json_str)


def _upload_to_supabase(detections: List[EnrichedDetection], source_id: str):
    """upload detections to supabase"""
    from db import (
        get_supabase_client,
        upload_batch,
        insert_batch_detections,
        ensure_bucket_exists,
    )
    
    try:
        #test connection
        client = get_supabase_client()
        
        #ensure storage bucket exists
        ensure_bucket_exists()
        
        #get unique frame paths
        unique_frames = list(set(d.frame_path for d in detections))
        
        #upload images
        logger.info(f"  Uploading {len(unique_frames)} images...")
        image_paths = upload_batch(unique_frames, source_id)
        
        #insert to database
        logger.info(f"  Inserting {len(detections)} detections...")
        inserted_ids = insert_batch_detections(detections, source_id, image_paths)
        
        logger.info(f"  Uploaded: {len(inserted_ids)} detections to Supabase")
        
    except Exception as e:
        logger.error(f"  Upload failed: {e}")
        logger.info("  Tip: Run with --dry-run to output JSON instead")


def _print_summary(detections: List[EnrichedDetection]):
    """print detection summary"""
    if not detections:
        logger.info("No detections to summarize")
        return
    
    #count by type
    by_type = {}
    total_severity = 0
    
    for d in detections:
        by_type[d.damage_type] = by_type.get(d.damage_type, 0) + 1
        total_severity += d.severity
    
    logger.info(f"Total detections: {len(detections)}")
    logger.info(f"Average severity: {total_severity / len(detections):.2f}")
    logger.info("By type:")
    for dtype, count in sorted(by_type.items()):
        logger.info(f"  - {dtype}: {count}")


@cli.command()
def test_connection():
    """test supabase connection"""
    from db.client import test_connection as test_sb
    
    logger.info("Testing Supabase connection...")
    
    if test_sb():
        logger.info("Connection successful!")
        sys.exit(0)
    else:
        logger.error("Connection failed!")
        sys.exit(1)


@cli.command()
@click.option("--source-id", "-s", required=True, help="source id to query")
def stats(source_id: str):
    """get detection statistics for a source"""
    from db.database import get_detection_stats
    
    stats = get_detection_stats(source_id)
    
    logger.info(f"Stats for source: {source_id}")
    logger.info(f"  Total: {stats['total']}")
    logger.info(f"  Avg confidence: {stats['avg_confidence']:.2f}")
    logger.info(f"  Avg severity: {stats['avg_severity']:.2f}")
    logger.info("  By type:")
    for dtype, count in stats['by_type'].items():
        logger.info(f"    - {dtype}: {count}")


if __name__ == "__main__":
    cli()
