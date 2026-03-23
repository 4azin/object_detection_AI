import requests
import csv
import time
import os

# iNaturalist API Endpoint for species counts
API_URL = "https://api.inaturalist.org/v1/observations/species_counts"

# Fish Taxon ID (Actinopterygii)
TAXON_ID = 47178

# Bounding Boxes for 3 + 1 Fallback Regions
# Format: (sw_lat, sw_lng, ne_lat, ne_lng)
REGIONS = {
    "west":  (34.0, 124.0, 39.0, 127.0),
    "south": (32.0, 124.0, 34.0, 132.0),
    "east":  (34.0, 127.0, 39.0, 132.0),
    "korea": (32.0, 124.0, 39.0, 132.0)   # KOREAN_PENINSULA Fallback
}

def fetch_species_counts_for_region(region_name, bbox):
    sw_lat, sw_lng, ne_lat, ne_lng = bbox
    print(f"Fetching data for {region_name.upper()} region...")
    
    page = 1
    total_results = []
    
    while True:
        params = {
            "taxon_id": TAXON_ID,
            "quality_grade": "research",
            "nelat": ne_lat,
            "nelng": ne_lng,
            "swlat": sw_lat,
            "swlng": sw_lng,
            "page": page,
            "per_page": 500
        }
        
        try:
            response = requests.get(API_URL, params=params, timeout=10)
            if response.status_code != 200:
                print(f"Error fetching page {page}: {response.status_code}")
                break
            
            data = response.json()
            results = data.get("results", [])
            
            if not results:
                break
                
            total_results.extend(results)
            print(f"  [{region_name.upper()}] Fetched page {page} ({len(results)} items)...")
            
            if len(results) < 500:
                # We reached the last page
                break
            
            page += 1
            # Rate limiting sleep (iNat API recommends ~1.5 req/sec, so ~0.7s)
            time.sleep(1)
            
        except Exception as e:
            print(f"Failed to fetch {region_name}: {e}")
            break

    return total_results

import csv

def process_and_save_priors(region_name, results, output_dir="."):
    if not results:
        print(f"No results for {region_name}.")
        return

    records = []
    max_count = 0
    for item in results:
        count = item.get("count", 0)
        taxon = item.get("taxon", {})
        taxon_id = taxon.get("id")
        name = taxon.get("name")
        kor_name = taxon.get("preferred_common_name", "")
        
        if taxon_id and name:
            records.append({
                "taxon_id": taxon_id,
                "scientific_name": name,
                "korean_name": kor_name,
                "count": count
            })
            if count > max_count:
                max_count = count
            
    # Calculate P_prior (normalized) and add to records
    for record in records:
        record["prior"] = record["count"] / max_count if max_count > 0 else 0.0
        
    # Sort by prior descending
    records.sort(key=lambda x: x["prior"], reverse=True)
    
    csv_path = os.path.join(output_dir, f"prior_{region_name}.csv")
    with open(csv_path, mode='w', newline='', encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["taxon_id", "scientific_name", "korean_name", "count", "prior"])
        writer.writeheader()
        writer.writerows(records)
        
    print(f"Saved {len(records)} species to {csv_path}\n")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    for name, bbox in REGIONS.items():
        data = fetch_species_counts_for_region(name, bbox)
        process_and_save_priors(name, data, current_dir)
        
    print("All scraping completed successfully.")
