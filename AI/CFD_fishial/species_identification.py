import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any


NO_MATCH_MESSAGE = "도감에 없습니다."
OPEN_DOMAIN_TOPK = 5
RESULT_DISPLAY_TOPK = 5
RANK_COLORS = {
    1: "#00d2ff",
    2: "#ffd93d",
    3: "#ff9f43",
    4: "#7bed9f",
    5: "#70a1ff",
}
RANK_LABELS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th"}


@dataclass(frozen=True)
class ClassificationRequest:
    model_choice: str
    mode: str
    custom_classes: list[str]
    apply_obis_filter: bool = False
    apply_genus_zero_shot: bool = False


@dataclass(frozen=True)
class ObisReference:
    scientific_names: set[str]
    common_name_by_scientific: dict[str, str]
    species_by_genus: dict[str, list[str]]


def normalize_scientific_name(name: str) -> str:
    if not name:
        return ""
    normalized = re.sub(r"\s+", " ", str(name).strip())
    return normalized.casefold()


def extract_scientific_name(prediction: str) -> str:
    if not prediction:
        return ""

    text = str(prediction).strip()
    text = re.split(r"\s\(|\s\[", text, maxsplit=1)[0].strip()
    matches = re.findall(r"([A-Z][a-zA-Z-]+)\s+([a-z][a-zA-Z-]+)", text)
    if matches:
        genus, species = matches[-1]
        return f"{genus} {species}"
    return text


def resolve_classification_request(
    model_choice_ui: str,
    classification_mode: str,
    zero_shot_classes: str,
) -> ClassificationRequest:
    model_choice = "bioclip-2.5-vith14" if "2.5" in model_choice_ui else "bioclip2"

    if "Open-Domain -> Genus Zero-Shot" in classification_mode:
        _ensure_open_domain_supported(model_choice, "genus zero-shot mode")
        return ClassificationRequest(
            model_choice=model_choice,
            mode="open-domain",
            custom_classes=[],
            apply_genus_zero_shot=True,
        )

    if "Open-Domain + OBIS" in classification_mode:
        _ensure_open_domain_supported(model_choice, "OBIS-filtered mode")
        return ClassificationRequest(
            model_choice=model_choice,
            mode="open-domain",
            custom_classes=[],
            apply_obis_filter=True,
        )

    if "Open-Domain" in classification_mode:
        _ensure_open_domain_supported(model_choice, "Open-Domain classification")
        return ClassificationRequest(
            model_choice=model_choice,
            mode="open-domain",
            custom_classes=[],
        )

    if "Zero-Shot (Custom List)" in classification_mode:
        custom_classes = [c.strip() for c in zero_shot_classes.split("\n") if c.strip()]
        if not custom_classes:
            raise ValueError("Please provide candidate species for Zero-Shot mode.")
        return ClassificationRequest(
            model_choice=model_choice,
            mode="zero-shot",
            custom_classes=custom_classes,
        )

    if "Zero-Shot (All Bioinfo List)" in classification_mode:
        return ClassificationRequest(
            model_choice=model_choice,
            mode="zero-shot-all",
            custom_classes=[],
        )

    if "Two-Stage" in classification_mode:
        return ClassificationRequest(
            model_choice=model_choice,
            mode="two-stage",
            custom_classes=[],
        )

    raise ValueError(f"Unsupported classification mode: {classification_mode}")


def get_result_ranks(row: dict[str, Any], max_rank: int = RESULT_DISPLAY_TOPK) -> list[int]:
    return [rank for rank in range(1, max_rank + 1) if row.get(f"Top{rank}_Species", "")]


def filter_results_to_obis_species(results: list[dict], obis_reference_path: Path, topk: int = OPEN_DOMAIN_TOPK) -> list[dict]:
    obis_reference = load_obis_reference(obis_reference_path)
    filtered_results: list[dict] = []

    for row in results:
        filtered_row = {
            "Image_Name": row.get("Image_Name", ""),
            "Inference_Time(ms)": row.get("Inference_Time(ms)", 0),
        }
        matched_candidates: list[tuple[str, float]] = []

        for rank in range(1, topk + 1):
            species_value = row.get(f"Top{rank}_Species", "")
            prob_value = row.get(f"Top{rank}_Prob", 0.0)
            scientific_name = extract_scientific_name(species_value)
            normalized = normalize_scientific_name(scientific_name)
            if normalized and normalized in obis_reference.scientific_names:
                matched_candidates.append((format_obis_result_name(scientific_name, obis_reference, species_value), prob_value))

        if not matched_candidates:
            filtered_results.append(build_no_match_result(row.get("Image_Name", ""), row.get("Inference_Time(ms)", 0.0), topk))
            continue

        for idx, (species_value, prob_value) in enumerate(matched_candidates[:topk], start=1):
            filtered_row[f"Top{idx}_Species"] = species_value
            filtered_row[f"Top{idx}_Prob"] = prob_value

        for rank in range(len(matched_candidates[:topk]) + 1, topk + 1):
            filtered_row[f"Top{rank}_Species"] = ""
            filtered_row[f"Top{rank}_Prob"] = 0.0

        filtered_results.append(filtered_row)

    return filtered_results


def run_open_domain_genus_zero_shot(
    classifier,
    best_shots_path: str,
    selected_shots: list[str],
    final_topk: int,
    obis_reference_path: Path,
) -> list[dict]:
    original_topk = classifier.topk
    try:
        classifier.topk = max(original_topk, OPEN_DOMAIN_TOPK)
        stage1_results = classifier.run(
            best_shots_path,
            mode="open-domain",
            custom_classes=[],
            target_files=selected_shots,
        )
    finally:
        classifier.topk = original_topk

    if not stage1_results:
        return []

    obis_reference = load_obis_reference(obis_reference_path)
    final_results: list[dict] = []
    print("[GenusZeroShot] ================================================")
    print(f"[GenusZeroShot] Stage-1 Open-Domain images: {len(stage1_results)}")

    for stage1_row in stage1_results:
        image_name = stage1_row.get("Image_Name", "")
        image_path = Path(best_shots_path) / image_name
        if not image_path.exists():
            continue

        print(f"[GenusZeroShot] Image: {image_name}")
        stage1_candidates = [
            extract_scientific_name(stage1_row.get(f"Top{rank}_Species", ""))
            for rank in range(1, OPEN_DOMAIN_TOPK + 1)
            if stage1_row.get(f"Top{rank}_Species", "")
        ]
        log_species_candidates("[GenusZeroShot] Stage-1 scientific candidates", stage1_candidates, limit=OPEN_DOMAIN_TOPK)

        genus_list = sorted({
            parts[0].casefold()
            for parts in (candidate.split() for candidate in stage1_candidates)
            if len(parts) >= 2
        })
        print(f"[GenusZeroShot] Genus candidates ({len(genus_list)}): {', '.join(genus_list) if genus_list else '[]'}")

        zero_shot_candidates: list[str] = []
        seen_species: set[str] = set()
        for genus_name in genus_list:
            for scientific_name in obis_reference.species_by_genus.get(genus_name, []):
                if scientific_name not in seen_species:
                    zero_shot_candidates.append(scientific_name)
                    seen_species.add(scientific_name)

        log_species_candidates("[GenusZeroShot] Stage-2 OBIS zero-shot candidates", zero_shot_candidates)

        if not zero_shot_candidates:
            print("[GenusZeroShot] No OBIS species matched extracted genus. Returning no-match result.")
            final_results.append(build_no_match_result(image_name, stage1_row.get("Inference_Time(ms)", 0.0), final_topk))
            continue

        target_txt_emb = classifier.get_txt_features(zero_shot_candidates)
        target_names = [([scientific_name], "") for scientific_name in zero_shot_candidates]

        try:
            classifier.topk = final_topk
            final_row = classifier._classify_single(image_path, target_txt_emb, target_names)
        finally:
            classifier.topk = original_topk

        if final_row is None:
            print("[GenusZeroShot] Zero-shot re-ranking failed. Returning no-match result.")
            final_results.append(build_no_match_result(image_name, stage1_row.get("Inference_Time(ms)", 0.0), final_topk))
            continue

        final_results.append(format_result_row_with_obis_names(final_row, obis_reference, final_topk))
        final_candidates = [
            final_results[-1].get(f"Top{rank}_Species", "")
            for rank in range(1, final_topk + 1)
            if final_results[-1].get(f"Top{rank}_Species", "")
        ]
        log_species_candidates("[GenusZeroShot] Final zero-shot results", final_candidates, limit=final_topk)

    print("[GenusZeroShot] Completed genus zero-shot pipeline.")
    print("[GenusZeroShot] ================================================")
    return final_results


def format_result_row_with_obis_names(row: dict[str, Any], obis_reference: ObisReference, topk: int) -> dict[str, Any]:
    formatted = dict(row)
    for rank in range(1, topk + 1):
        species_value = formatted.get(f"Top{rank}_Species", "")
        scientific_name = extract_scientific_name(species_value)
        if scientific_name:
            formatted[f"Top{rank}_Species"] = format_obis_result_name(scientific_name, obis_reference, species_value)
    return formatted


def format_obis_result_name(scientific_name: str, obis_reference: ObisReference, fallback_label: str = "") -> str:
    common_name = obis_reference.common_name_by_scientific.get(normalize_scientific_name(scientific_name), "").strip()
    scientific_name = scientific_name.strip()
    if common_name:
        return f"{scientific_name} ({common_name})"
    return scientific_name or fallback_label


def build_no_match_result(image_name: str, inference_time_ms: float = 0.0, topk: int = RESULT_DISPLAY_TOPK) -> dict[str, Any]:
    row: dict[str, Any] = {
        "Image_Name": image_name,
        "Top1_Species": NO_MATCH_MESSAGE,
        "Top1_Prob": 0.0,
        "Inference_Time(ms)": inference_time_ms,
    }
    for rank in range(2, topk + 1):
        row[f"Top{rank}_Species"] = ""
        row[f"Top{rank}_Prob"] = 0.0
    return row


def log_species_candidates(prefix: str, candidates: list[str], limit: int = 10) -> None:
    if not candidates:
        print(f"{prefix}: []")
        return

    preview = ", ".join(candidates[:limit])
    suffix = "" if len(candidates) <= limit else f", ... (+{len(candidates) - limit} more)"
    print(f"{prefix} ({len(candidates)}): {preview}{suffix}")


@lru_cache(maxsize=1)
def load_obis_reference(obis_reference_path: Path) -> ObisReference:
    if not obis_reference_path.exists():
        return ObisReference(set(), {}, {})

    with open(obis_reference_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data.get("items", []) if isinstance(data, dict) else data
    scientific_names: set[str] = set()
    common_name_by_scientific: dict[str, str] = {}
    species_by_genus: dict[str, set[str]] = {}

    for item in items:
        for scientific_name in _extract_candidate_scientific_names(item):
            normalized = normalize_scientific_name(scientific_name)
            if not normalized:
                continue
            scientific_names.add(normalized)
            genus = normalized.split()[0]
            species_by_genus.setdefault(genus, set()).add(normalized)
            common_name = _extract_common_name(item)
            if common_name:
                common_name_by_scientific.setdefault(normalized, common_name)

    return ObisReference(
        scientific_names=scientific_names,
        common_name_by_scientific=common_name_by_scientific,
        species_by_genus={genus: sorted(species) for genus, species in species_by_genus.items()},
    )


def _extract_candidate_scientific_names(item: dict[str, Any]) -> list[str]:
    candidates: list[str] = []

    for key in ("학명", "SpcScitfNm", "obis_queried_as", "obis_queried_asobis_queried_as"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            candidates.append(extract_scientific_name(value))

    genus = item.get("Genus")
    species = item.get("Species")
    if isinstance(genus, str) and isinstance(species, str) and genus.strip() and species.strip():
        candidates.append(f"{genus.split()[0].strip()} {species.split()[0].strip()}")

    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        normalized = normalize_scientific_name(candidate)
        if normalized and normalized not in seen:
            deduped.append(candidate)
            seen.add(normalized)
    return deduped


def _extract_common_name(item: dict[str, Any]) -> str:
    for key in ("국명", "CommKorNm", "援?챸"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _ensure_open_domain_supported(model_choice: str, mode_name: str) -> None:
    if model_choice != "bioclip-2.5-vith14":
        return
    raise ValueError(
        "BioCLIP-2.5 (ViT-H-14) does not currently support Open-Domain classification. "
        f"Please change the Model back to BioCLIP-2 for the {mode_name}."
    )
