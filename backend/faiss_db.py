import faiss
import numpy as np
import os

class EmployeeVectorDB:
    def __init__(self, embedding_dim=128):
        self.embedding_dim = embedding_dim
        
        # FAISS Index using Inner Product (Requires L2 Normalized vectors for Cosine Similarity)
        self.index = faiss.IndexFlatIP(embedding_dim) 
        self.employee_names = []
        self.employee_strengths = []
        
    def identify(self, query_embedding, threshold=0.45, gap=0.08):
        """
        Returns (Name, similarity_score, strength) or ('Unknown', similarity_score, 'NONE')
        """
        if hasattr(self, 'index') and self.index.ntotal == 0:
            print("[FAISS] GLOBAL REJECT (SAFE MODE): 0 Centroids in system.")
            return "Unknown", 0.0, "NONE"
            
        unique_identities = self.index.ntotal
        if unique_identities == 1:
            print("[FAISS] LIMITED MODE: Only 1 identity exists. Recognition restricted to extremely strong matches.")
            
        # Ensure input is 2D array [1, 512]
        query = np.array([query_embedding], dtype=np.float32)
        
        # Search all embedding centroids
        k = self.index.ntotal
        distances, indices = self.index.search(query, k)
        
        results = []
        for i in range(k):
            score = float(distances[0][i])
            idx = int(indices[0][i])
            if idx < len(self.employee_names):
                # Fallback to STRONG if list not perfectly aligned during transition
                strength = self.employee_strengths[idx] if idx < len(self.employee_strengths) else "STRONG"
                results.append((self.employee_names[idx], score, strength))
                
        results.sort(key=lambda x: x[1], reverse=True)
        if not results:
            return "Unknown", 0.0, "NONE"
            
        best_name, best_score, best_strength = results[0]
        
        if best_strength == "STRONG":
            # Dual-Threshold Approach
            ACCEPT_THRESHOLD = threshold         # default 0.55
            REJECT_THRESHOLD = threshold - 0.10  # default 0.45
            
            if best_score < REJECT_THRESHOLD:
                print(f"[FAISS] REJECTED: {best_name} is a WEAK Match (Score: {best_score:.4f} < {REJECT_THRESHOLD}).")
                return "Unknown", best_score, best_strength
                
            if REJECT_THRESHOLD <= best_score < ACCEPT_THRESHOLD:
                print(f"[FAISS] REJECTED: {best_name} is in UNCERTAIN/BORDERLINE Range (Score: {best_score:.4f}).")
                return "Unknown", best_score, best_strength
                
            # Check Gap Rule (False Positive Protection)
            if len(results) > 1:
                second_name, second_score, _ = results[1]
                score_diff = best_score - second_score
                
                # Partial match rejection
                if score_diff < gap:
                    print(f"[FAISS] REJECTED: Another identity partially matches (Gap {score_diff:.4f} < {gap}).")
                    return "Unknown", best_score, best_strength
                    
            print(f"[FAISS] STRONG MATCH: {best_name} (Score: {best_score:.4f})")
            return best_name, best_score, best_strength

        elif best_strength == "WEAK":
            WEAK_ABSOLUTE_THRESHOLD = 0.45
            WEAK_GAP_THRESHOLD = 0.08
            
            # 1. Use Absolute Threshold: Accept ONLY if similarity is extremely high
            if best_score < WEAK_ABSOLUTE_THRESHOLD:
                print(f"[FAISS] WEAK MODE REJECTED: {best_name} score {best_score:.4f} < {WEAK_ABSOLUTE_THRESHOLD}")
                return "Unknown", best_score, best_strength
                
            # 2. UNIQUENESS CHECK (CRITICAL)
            if len(results) > 1:
                second_name, second_score, _ = results[1]
                score_diff = best_score - second_score
                if score_diff < WEAK_GAP_THRESHOLD:
                    print(f"[FAISS] WEAK MODE REJECTED: Gap {score_diff:.4f} < {WEAK_GAP_THRESHOLD}")
                    return "Unknown", best_score, best_strength
                    
            print(f"[FAISS] WEAK MATCH CANDIDATE: {best_name} (Score: {best_score:.4f}). Validating temporally.")
            return best_name, best_score, best_strength
            
        return "Unknown", best_score, "NONE"
