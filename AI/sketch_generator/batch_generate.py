#!/usr/bin/env python3
"""
Batch Sketch Generator
Generates fish sketches for entries in fish_book database.

Usage:
    python batch_generate.py --limit 10
    python batch_generate.py --limit 50 --style kawaii
    python batch_generate.py --fish-id 123
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import boto3
from sqlalchemy import create_engine, text

from generator import SketchGenerator, create_generator_from_env

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("batch-sketch")


class BatchSketchGenerator:
    """Batch generate fish sketches and upload to S3."""

    def __init__(
        self,
        db_url: str,
        s3_bucket: str,
        generator: SketchGenerator,
        region: str = "ap-northeast-2",
    ):
        self.engine = create_engine(db_url)
        self.s3 = boto3.client("s3", region_name=region)
        self.bucket = s3_bucket
        self.region = region
        self.generator = generator

    def get_fish_without_sketch(self, limit: int = 50) -> List[dict]:
        """Get fish entries that don't have default_img or have placeholder."""
        query = text("""
            SELECT fish_id, name, scientific_name, default_img
            FROM fish_book
            WHERE default_img IS NULL
               OR default_img = ''
               OR default_img LIKE '%placeholder%'
            ORDER BY fish_id
            LIMIT :limit
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"limit": limit})
            return [dict(row._mapping) for row in result]

    def get_fish_by_id(self, fish_id: int) -> Optional[dict]:
        """Get a specific fish entry by ID."""
        query = text("""
            SELECT fish_id, name, scientific_name, default_img
            FROM fish_book
            WHERE fish_id = :fish_id
        """)

        with self.engine.connect() as conn:
            result = conn.execute(query, {"fish_id": fish_id})
            row = result.fetchone()
            return dict(row._mapping) if row else None

    def update_fish_sketch_url(self, fish_id: int, s3_url: str) -> None:
        """Update the fish_book entry with the new sketch URL."""
        query = text("""
            UPDATE fish_book
            SET default_img = :s3_url
            WHERE fish_id = :fish_id
        """)

        with self.engine.connect() as conn:
            conn.execute(query, {"fish_id": fish_id, "s3_url": s3_url})
            conn.commit()

    def generate_and_upload(
        self,
        fish: dict,
        style: str = "kawaii",
        dry_run: bool = False,
    ) -> Optional[str]:
        """Generate sketch for a fish and upload to S3."""
        fish_id = fish["fish_id"]
        name = fish["name"]
        scientific_name = fish.get("scientific_name")

        try:
            # Generate sketch
            image_bytes = self.generator.generate_sketch(
                name=name,
                scientific_name=scientific_name,
                style=style,
            )

            if dry_run:
                log.info(f"[DRY RUN] Would upload sketch for {name} (fish_id={fish_id})")
                return None

            # Upload to S3
            s3_key = f"fish_sketches/{fish_id}.png"
            self.s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=image_bytes,
                ContentType="image/png",
            )

            s3_url = f"https://{self.bucket}.s3.{self.region}.amazonaws.com/{s3_key}"

            # Update database
            self.update_fish_sketch_url(fish_id, s3_url)

            log.info(f"Generated and uploaded sketch for {name} (fish_id={fish_id})")
            return s3_url

        except Exception as e:
            log.error(f"Failed to generate sketch for {name} (fish_id={fish_id}): {e}")
            return None

    def run_batch(
        self,
        limit: int = 50,
        style: str = "kawaii",
        dry_run: bool = False,
    ) -> dict:
        """Run batch generation for fish without sketches."""
        fish_list = self.get_fish_without_sketch(limit)
        log.info(f"Found {len(fish_list)} fish entries without sketches")

        results = {"success": 0, "failed": 0, "skipped": 0}

        for fish in fish_list:
            url = self.generate_and_upload(fish, style=style, dry_run=dry_run)
            if url:
                results["success"] += 1
            elif dry_run:
                results["skipped"] += 1
            else:
                results["failed"] += 1

        return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch generate fish sketches")
    parser.add_argument("--limit", type=int, default=10, help="Number of fish to process")
    parser.add_argument("--fish-id", type=int, help="Generate for a specific fish ID")
    parser.add_argument("--style", default="kawaii", choices=["kawaii", "realistic", "watercolor"])
    parser.add_argument("--dry-run", action="store_true", help="Don't actually upload")
    args = parser.parse_args()

    # Environment variables
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "3306")
        db_name = os.getenv("DB_NAME", "divery")
        db_user = os.getenv("DB_USER", "root")
        db_pass = os.getenv("DB_PASSWORD", "")
        db_url = f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"

    s3_bucket = os.getenv("S3_BUCKET_NAME")
    if not s3_bucket:
        log.error("S3_BUCKET_NAME environment variable is required")
        return 1

    try:
        generator = create_generator_from_env()
    except Exception as e:
        log.error(f"Failed to create sketch generator: {e}")
        return 1

    batch = BatchSketchGenerator(
        db_url=db_url,
        s3_bucket=s3_bucket,
        generator=generator,
    )

    if args.fish_id:
        fish = batch.get_fish_by_id(args.fish_id)
        if not fish:
            log.error(f"Fish with ID {args.fish_id} not found")
            return 1
        url = batch.generate_and_upload(fish, style=args.style, dry_run=args.dry_run)
        if url:
            log.info(f"Success: {url}")
        return 0 if url else 1
    else:
        results = batch.run_batch(limit=args.limit, style=args.style, dry_run=args.dry_run)
        log.info(f"Batch complete: {results}")
        return 0 if results["failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
