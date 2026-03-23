import os
import csv
from collections import defaultdict

# Bounding Boxes for 3 Regions + 1 Fallback
REGIONS = {
    "west":  (34.0, 124.0, 39.0, 127.0),
    "south": (32.0, 124.0, 34.0, 132.0),
    "east":  (34.0, 127.0, 39.0, 132.0)
}
FALLBACK_REGION = "korea"

def get_regions(lat, lng):
    """
    Returns a list of regions that the given (lat, lng) belongs to.
    Handles overlap by returning multiple regions if on boundary.
    If no regions match, returns the fallback region.
    """
    matched = []
    for name, bbox in REGIONS.items():
        sw_lat, sw_lng, ne_lat, ne_lng = bbox
        if sw_lat <= lat <= ne_lat and sw_lng <= lng <= ne_lng:
            matched.append(name)
            
    if not matched:
        return [FALLBACK_REGION]
    return matched

def load_priors(regions, base_dir="."):
    """
    Loads priors for a list of regions.
    If multiple regions (boundary overlap), takes the MAX prior for each species.
    Returns a dictionary mapping species name (both scientific and korean) to prior score.
    """
    merged_priors = defaultdict(float)
    
    for r in regions:
        csv_path = os.path.join(base_dir, f"prior_{r}.csv")
        if not os.path.exists(csv_path):
            print(f"Warning: Prior data for '{r}' not found at {csv_path}")
            continue
            
        with open(csv_path, mode='r', encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sci_name = row["scientific_name"].strip()
                kor_name = row["korean_name"].strip()
                try:
                    prior_val = float(row["prior"])
                except ValueError:
                    prior_val = 0.0
                    
                # Merge logic: if overlapping, take the highest prior (합산 논리 중 최대값 보존)
                if sci_name:
                    merged_priors[sci_name] = max(merged_priors[sci_name], prior_val)
                if kor_name:
                    merged_priors[kor_name] = max(merged_priors[kor_name], prior_val)
                    
    return merged_priors

def rerank(predictions, priors, alpha=0.4, min_prior=0.001):
    """
    predictions: list of dicts [{'name': '...', 'score': 0.85}, ...]
    Applies formula: Final_Score = P_model * (alpha * P_prior + (1 - alpha))
    Returns sorted list.
    """
    reranked = []
    for p in predictions:
        name = p["name"]
        p_model = p["score"]
        
        # Look up prior; if not found, use min_prior
        p_prior = priors.get(name, min_prior)
        
        # Calculate weighted score
        final_score = p_model * (alpha * p_prior + (1.0 - alpha))
        
        reranked.append({
            "name": name,
            "original_score": p_model,
            "prior": p_prior,
            "final_score": final_score
        })
        
    # Sort by final score descending
    reranked.sort(key=lambda x: x["final_score"], reverse=True)
    return reranked

def print_table(results):
    """Prints the re-ranked results in a clean table format."""
    print(f"{'Rank':<5} | {'Species Name':<30} | {'Orig Score':<10} | {'Prior':<10} | {'Final Score':<10}")
    print("-" * 76)
    for i, res in enumerate(results, 1):
        print(f"{i:<5} | {res['name']:<30} | {res['original_score']:<10.4f} | {res['prior']:<10.4f} | {res['final_score']:<10.4f}")
    print()

def run_simulation():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Mock model predictions for a generic fish
    dummy_predictions = [
        {"name": "Acanthopagrus schlegelii", "score": 0.80}, # 감성돔 (Common in Korea)
        {"name": "Mola mola", "score": 0.75},                # 개복치 (Less common)
        {"name": "Unknown Alien Fish", "score": 0.70},       # Not existing in prior
        {"name": "Paralichthys olivaceus", "score": 0.65},   # 넙치 (광어)
    ]
    
    print("=== Re-ranking Simulator Started ===")
    
    test_cases = [
        {"desc": "In WEST region", "lat": 36.0, "lng": 125.5},
        {"desc": "In SOUTH region", "lat": 33.0, "lng": 128.0},
        {"desc": "Boundary Overlap (WEST & EAST)", "lat": 36.0, "lng": 127.0},
        {"desc": "Out of Bounds (Fallback KOREA)", "lat": 40.0, "lng": 129.0}
    ]
    
    for case in test_cases:
        print(f"\n--- Test Case: {case['desc']} (Lat: {case['lat']}, Lng: {case['lng']}) ---")
        regions = get_regions(case['lat'], case['lng'])
        print(f"Detected Region(s): {regions}")
        
        priors = load_priors(regions, base_dir=current_dir)
        print(f"Loaded priors for {len(priors)//2} species mappings.") # Divide by 2 because of EN/KR dups
        
        reranked = rerank(dummy_predictions, priors, alpha=0.4)
        print_table(reranked)

if __name__ == "__main__":
    run_simulation()
