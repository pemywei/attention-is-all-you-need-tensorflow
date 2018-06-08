import numpy as np

def getEmbedding(infile_path):
    emb_matrix = []
    
    with open(infile_path, "r") as infile:
        for row in infile:
            items = row.strip().split()
            emb_vec = [float(val) for val in items]
            emb_matrix.append(np.array(emb_vec))
    
    return np.array(emb_matrix, np.float32)
