import json

# Load results
with open('results/attack_results.json', 'r') as f:
    data = json.load(f)

print("=" * 60)
print("ATTACK RESULTS ANALYSIS")
print("=" * 60)

# Basic stats
print(f"\nTotal traces attacked: {len(data)}")

# First trace details
first = data[0]
print(f"\n--- First Trace Example ---")
print(f"Predicted KENC: {first['3DES_KENC']}")
print(f"Reference KENC: {first['reference_KENC']}")
print(f"Match: {first['3DES_KENC'] == first['reference_KENC']}")
print(f"Mean confidence: {first['mean_confidence']:.6f}")
print(f"Min confidence: {first['min_confidence']:.6f}")
print(f"Rank-0 success: {first['rank_0_success']}")
print(f"S-Box predictions: {first['sbox_predictions']}")

# Overall statistics
unique_predicted = set(r['3DES_KENC'] for r in data)
unique_reference = set(r['reference_KENC'] for r in data)

print(f"\n--- Overall Statistics ---")
print(f"Unique predicted KENC values: {len(unique_predicted)}")
print(f"Unique reference KENC values: {len(unique_reference)}")
print(f"All predictions identical: {len(unique_predicted) == 1}")

# Rank-0 success rate
rank0_count = sum(1 for r in data if r['rank_0_success'])
print(f"\nRank-0 success rate: {rank0_count}/{len(data)} = {rank0_count/len(data)*100:.1f}%")

# Confidence statistics
all_confidences = [r['mean_confidence'] for r in data]
print(f"\nConfidence Statistics:")
print(f"  Mean: {sum(all_confidences)/len(all_confidences):.6f}")
print(f"  Min: {min(all_confidences):.6f}")
print(f"  Max: {max(all_confidences):.6f}")

# Check if any predictions match reference
matches = sum(1 for r in data if r['3DES_KENC'] == r['reference_KENC'])
print(f"\nPredictions matching reference: {matches}/{len(data)} = {matches/len(data)*100:.1f}%")

# S-Box prediction consistency
sbox_preds = [tuple(r['sbox_predictions']) for r in data]
unique_sbox = set(sbox_preds)
print(f"\nUnique S-Box prediction patterns: {len(unique_sbox)}")
print(f"Most common S-Box pattern: {sbox_preds[0]}")

print("\n" + "=" * 60)
