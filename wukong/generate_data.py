import numpy as np

def generate_criteo_sample(filename, num_samples=100000): # 100k
    with open(filename, "w") as f:
        for _ in range(num_samples):
            # Criteo format: label, 13 dense, 26 sparse (tab separated)
            dense = np.random.rand(13)
            sparse = [hex(np.random.randint(0, 0xFFFF))[2:] for _ in range(26)]
            
            # Correlate with features but skew towards 75% negatives to match Criteo
            base_signal = 2.0 if (int(sparse[0], 16) % 4 == 0) else 0.0
            score = dense[0] + base_signal
            label = 1 if score > 1.5 else 0
            
            # Inject 10% Noise
            if np.random.rand() < 0.1:
                label = 1 - label
                
            dense_str = "\t".join([str(int(d*100)) for d in dense])
            sparse_str = "\t".join(sparse)
            f.write(f"{label}\t{dense_str}\t{sparse_str}\n")

if __name__ == "__main__":
    generate_criteo_sample("criteo_sample.txt")
    print("Generated 100k sample with 10% Label Noise in criteo_sample.txt")
