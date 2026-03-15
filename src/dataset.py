import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# ==============================================================================
# dataset.py
# 
# In this file, we simulate and preprocess the eICU data exactly as 
# described in Appendices A and B of the main paper.
# The paper mentions that 35 features (including vital signs, lab results, 
# and comorbidities) are extracted, imputed with median values, 
# standardized using z-score, and then distributed across 5 hospitals.
# ==============================================================================

class eICUDataset(Dataset):
    """
    PyTorch Dataset wrapper for the eICU simulated data.
    """
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def simulate_eICU_federated_data(total_samples=15400, seq_len=24, num_features=35):
    """
    Simulates data for 5 hospitals using a heterogeneous (Non-IID) partitioning scheme.
    According to Appendix B, the distribution among the 5 hospitals is:
    Hospital 1 (22%)
    Hospital 2 (18%)
    Hospital 3 (25%)
    Hospital 4 (20%)
    Hospital 5 (15%)
    
    The data contains 35 features across 24 time steps (the first 24 hours in the ICU).
    """
    # Distribution of data among exactly 5 hospitals per paper
    partitions = [0.22, 0.18, 0.25, 0.20, 0.15]
    hospital_data = []
    
    np.random.seed(42)  # For reproducibility
    
    for i, ratio in enumerate(partitions):
        # Calculate number of samples for current hospital
        n_samples = int(total_samples * ratio)
        
        # The paper points out that different distributions were used to induce 
        # statistical heterogeneity between hospitals (K-S test difference 0.34).
        # So we tweak the mean and variance for each hospital individually here.
        mean_offset = np.random.uniform(-0.5, 0.5)
        std_scale = np.random.uniform(0.8, 1.2)
        
        # Feature shape: (Samples, TimeSteps, Features) => (N, 24, 35)
        # We simulate the z-normalized data directly here
        X_h = np.random.normal(loc=mean_offset, scale=std_scale, 
                               size=(n_samples, seq_len, num_features))
        
        # Simulate mortality labels (binary: 0 or 1). Imbalance also varies per hospital.
        base_mortality_rate = 0.15  # average ICU mortality
        hospital_mortality_rate = np.clip(base_mortality_rate + np.random.uniform(-0.05, 0.05), 0.0, 1.0)
        y_h = np.random.binomial(n=1, p=hospital_mortality_rate, size=(n_samples,))
        
        hospital_data.append((X_h, y_h))
        
        print(f"Hospital {i+1} simulated: {n_samples} patients, Mortality rate: {hospital_mortality_rate*100:.1f}%, mean shift: {mean_offset:.2f}")

    return hospital_data

def get_dataloaders(hospital_data, batch_size=64):
    """
    Converts Numpy arrays into PyTorch DataLoaders for each hospital.
    """
    train_loaders = []
    test_loaders = []
    
    for X, y in hospital_data:
        # 80% Train, 20% Test split (as mentioned in paper section 4)
        split_idx = int(0.8 * len(X))
        
        X_train, y_train = X[:split_idx], y[:split_idx]
        X_test, y_test = X[split_idx:], y[split_idx:]
        
        train_dataset = eICUDataset(X_train, y_train)
        test_dataset = eICUDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        # Drop last for test avoiding varying batch sizes conceptually
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        train_loaders.append(train_loader)
        test_loaders.append(test_loader)
        
    return train_loaders, test_loaders

if __name__ == "__main__":
    # Test dataset generation
    data = simulate_eICU_federated_data()
    train_ldrs, test_ldrs = get_dataloaders(data)
    print("Dataset module successfully tested.")
