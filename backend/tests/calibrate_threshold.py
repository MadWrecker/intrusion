import sqlite3
import json
import numpy as np
import os
import itertools

DB_PATH = 'factory_system.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def calibrate_thresholds():
    print("=========================================")
    print("   FORENSIC SIMILARITY CALIBRATOR")
    print("=========================================\n")
    
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT employee_id, name, face_embedding FROM Employees")
    rows = cursor.fetchall()
    
    identities = {}
    for r in rows:
        name = r['name']
        emp_id = r['employee_id']
        emb_data = r['face_embedding']
        if emb_data:
            try:
                emb_list = json.loads(emb_data)
                # Convert to numpy arrays
                emb_list = [np.array(e, dtype=np.float32) for e in emb_list]
                identities[f"{name} ({emp_id})"] = emb_list
            except Exception as e:
                print(f"Error parsing embeddings for {name}: {e}")
                
    if not identities:
        print("[ERROR] No embeddings found in the database.")
        print("Please ensure you have re-enrolled users after the coordinate bug fix.")
        return
        
    print(f"Loaded {len(identities)} distinct identities from database.\n")
    
    # 1. Intra-class Similarity (Same Person)
    print("--- 1. SAME PERSON SIMILARITY ---")
    same_person_scores = []
    for name, embs in identities.items():
        if len(embs) < 2:
            print(f"[{name}] Only 1 embedding enrolled. Needs >=2 to compute self-similarity.")
            continue
            
        scores = []
        for e1, e2 in itertools.combinations(embs, 2):
            sim = np.dot(e1, e2)
            scores.append(sim)
            same_person_scores.append(sim)
            
        print(f"[{name}] Average Self-Similarity: {np.mean(scores):.4f} (Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f})")
    
    if not same_person_scores:
        print("Not enough multi-embeddings per user to calculate same-person similarity distributions.")
    else:
        print(f"\n=> GLOBAL Same Person Average: {np.mean(same_person_scores):.4f}")
        print(f"=> GLOBAL Same Person MIN: {np.min(same_person_scores):.4f}")
        print("If the MIN is below 0.35, the person's face was distorted or differently lit during enrollment.\n")
        
    # 2. Inter-class Similarity (Different Persons)
    print("--- 2. DIFFERENT PERSON SIMILARITY ---")
    diff_person_scores = []
    
    names = list(identities.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            name1 = names[i]
            name2 = names[j]
            embs1 = identities[name1]
            embs2 = identities[name2]
            
            # Compare every embedding of Person A against every embedding of Person B
            pair_scores = []
            for e1 in embs1:
                for e2 in embs2:
                    sim = np.dot(e1, e2)
                    pair_scores.append(sim)
                    diff_person_scores.append(sim)
                    
            print(f"[{name1}] vs [{name2}] -> Max Similarity: {np.max(pair_scores):.4f} (Average: {np.mean(pair_scores):.4f})")
            
    if not diff_person_scores:
        print("\nNeed at least 2 distinct enrolled users to calculate different-person similarity.")
        return
        
    global_max_diff = np.max(diff_person_scores)
    global_avg_diff = np.mean(diff_person_scores)
    
    print(f"\n=> GLOBAL Different Person Average: {global_avg_diff:.4f}")
    print(f"=> GLOBAL Different Person MAX: {global_max_diff:.4f}")
    
    # 3. Final Recommendation
    print("\n=========================================")
    print("         THRESHOLD RECOMMENDATION       ")
    print("=========================================")
    
    recommended_threshold = float(global_max_diff) + 0.05
    # Strict floor matching what standard ONNX ArcFace implies
    if recommended_threshold < 0.25:
        recommended_threshold = 0.25
        
    print(f"To guarantee ZERO FALSE POSITIVES based on your dataset,")
    print(f"the strictest mathematical boundary to avoid crossover is:")
    print(f"\n>> RECOMMENDED THRESHOLD: {recommended_threshold:.2f} <<")
    
    print("\nIf you want to use Dual Threshold Safety (Gray Zone):")
    print(f"ACCEPTANCE THRESHOLD: {recommended_threshold + 0.02:.2f}")
    print(f"REJECTION THRESHOLD:  {global_max_diff:.2f}")
    
    print("\nRun this script frequently as you add more users to monitor dataset integrity.")

if __name__ == "__main__":
    calibrate_thresholds()
