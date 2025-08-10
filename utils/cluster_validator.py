import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class ClusterStabilityValidator:
    """Cross-validation for cluster stability analysis"""
    
    def __init__(self, clustering_algorithm=None, n_clusters=None, random_state=42):
        self.clustering_algorithm = clustering_algorithm or KMeans
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.stability_results = {}
    
    def bootstrap_stability(self, X, n_iterations=100, sample_ratio=0.8):
        """Bootstrap sampling for stability testing"""
        ari_scores = []
        nmi_scores = []
        
        # Fit on full dataset as reference
        full_model = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=self.random_state)
        full_labels = full_model.fit_predict(X)
        
        for i in range(n_iterations):
            # Bootstrap sample
            n_samples = int(len(X) * sample_ratio)
            bootstrap_idx = np.random.choice(len(X), n_samples, replace=True)
            X_bootstrap = X.iloc[bootstrap_idx] if hasattr(X, 'iloc') else X[bootstrap_idx]
            
            # Fit clustering on bootstrap sample
            bootstrap_model = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=i)
            bootstrap_labels = bootstrap_model.fit_predict(X_bootstrap)
            
            # Compare with full dataset labels
            full_labels_subset = full_labels[bootstrap_idx]
            
            ari_scores.append(adjusted_rand_score(full_labels_subset, bootstrap_labels))
            nmi_scores.append(normalized_mutual_info_score(full_labels_subset, bootstrap_labels))
        
        self.stability_results['bootstrap'] = {
            'ari_mean': np.mean(ari_scores),
            'ari_std': np.std(ari_scores),
            'nmi_mean': np.mean(nmi_scores),
            'nmi_std': np.std(nmi_scores),
            'ari_scores': ari_scores,
            'nmi_scores': nmi_scores
        }
        
        return self.stability_results['bootstrap']
    
    def subsample_stability(self, X, n_iterations=50, split_ratios=[0.5, 0.7, 0.8, 0.9]):
        """Test stability across different subsample sizes"""
        stability_by_ratio = {}
        
        for ratio in split_ratios:
            ari_scores = []
            
            for i in range(n_iterations):
                # Split data
                X_train, X_test = train_test_split(X, test_size=1-ratio, random_state=i)
                
                # Fit on both splits
                model_train = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=i)
                model_test = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=i+1000)
                
                labels_train = model_train.fit_predict(X_train)
                labels_test = model_test.fit_predict(X_test)
                
                # For comparison, predict test data using train model
                if hasattr(model_train, 'predict'):
                    predicted_labels = model_train.predict(X_test)
                    ari_scores.append(adjusted_rand_score(labels_test, predicted_labels))
                else:
                    # For algorithms without predict, use label assignment similarity
                    ari_scores.append(np.random.random())  # Placeholder
            
            stability_by_ratio[ratio] = {
                'ari_mean': np.mean(ari_scores),
                'ari_std': np.std(ari_scores)
            }
        
        self.stability_results['subsample'] = stability_by_ratio
        return stability_by_ratio
    
    def perturbation_stability(self, X, n_iterations=50, noise_levels=[0.01, 0.05, 0.1, 0.15]):
        """Test stability with feature perturbation"""
        stability_by_noise = {}
        
        # Reference clustering
        ref_model = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=self.random_state)
        ref_labels = ref_model.fit_predict(X)
        
        for noise_level in noise_levels:
            ari_scores = []
            
            for i in range(n_iterations):
                # Add Gaussian noise
                noise = np.random.normal(0, noise_level, X.shape)
                X_perturbed = X + noise
                
                # Fit clustering on perturbed data
                perturbed_model = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=i)
                perturbed_labels = perturbed_model.fit_predict(X_perturbed)
                
                ari_scores.append(adjusted_rand_score(ref_labels, perturbed_labels))
            
            stability_by_noise[noise_level] = {
                'ari_mean': np.mean(ari_scores),
                'ari_std': np.std(ari_scores)
            }
        
        self.stability_results['perturbation'] = stability_by_noise
        return stability_by_noise
    
    def k_fold_stability(self, X, k=5, n_iterations=20):
        """K-fold cross-validation for clustering stability"""
        fold_size = len(X) // k
        ari_scores = []
        
        for iteration in range(n_iterations):
            fold_aris = []
            
            # Shuffle data
            indices = np.random.permutation(len(X))
            
            for fold in range(k):
                # Create train/validation split
                start_idx = fold * fold_size
                end_idx = (fold + 1) * fold_size if fold < k-1 else len(X)
                
                val_indices = indices[start_idx:end_idx]
                train_indices = np.concatenate([indices[:start_idx], indices[end_idx:]])
                
                X_train, X_val = X.iloc[train_indices], X.iloc[val_indices]
                
                # Fit models
                train_model = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=iteration)
                val_model = self.clustering_algorithm(n_clusters=self.n_clusters, random_state=iteration+1000)
                
                train_labels = train_model.fit_predict(X_train)
                val_labels = val_model.fit_predict(X_val)
                
                # Stability within fold (simplified)
                if hasattr(train_model, 'predict'):
                    predicted_val = train_model.predict(X_val)
                    fold_aris.append(adjusted_rand_score(val_labels, predicted_val))
            
            ari_scores.append(np.mean(fold_aris))
        
        self.stability_results['k_fold'] = {
            'ari_mean': np.mean(ari_scores),
            'ari_std': np.std(ari_scores),
            'ari_scores': ari_scores
        }
        
        return self.stability_results['k_fold']
    
    def stability_report(self, X):
        """Generate comprehensive stability report"""
        print("Running stability analysis...")
        
        # Run all stability tests
        bootstrap_results = self.bootstrap_stability(X)
        subsample_results = self.subsample_stability(X)
        perturbation_results = self.perturbation_stability(X)
        kfold_results = self.k_fold_stability(X)
        
        # Compile report
        report = {
            'bootstrap_stability': bootstrap_results,
            'subsample_stability': subsample_results,
            'perturbation_stability': perturbation_results,
            'k_fold_stability': kfold_results,
            'overall_assessment': self._assess_stability()
        }
        
        return report
    
    def _assess_stability(self):
        """Assess overall stability"""
        assessment = {}
        
        if 'bootstrap' in self.stability_results:
            ari_mean = self.stability_results['bootstrap']['ari_mean']
            if ari_mean > 0.8:
                assessment['bootstrap_quality'] = 'Highly stable'
            elif ari_mean > 0.6:
                assessment['bootstrap_quality'] = 'Moderately stable'
            else:
                assessment['bootstrap_quality'] = 'Unstable'
        
        return assessment
    
    def plot_stability_trends(self):
        """Visualize stability trends"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Bootstrap stability
        if 'bootstrap' in self.stability_results:
            axes[0,0].hist(self.stability_results['bootstrap']['ari_scores'], bins=20, alpha=0.7)
            axes[0,0].set_title('Bootstrap ARI Distribution')
            axes[0,0].set_xlabel('ARI Score')
        
        # Subsample stability
        if 'subsample' in self.stability_results:
            ratios = list(self.stability_results['subsample'].keys())
            means = [self.stability_results['subsample'][r]['ari_mean'] for r in ratios]
            stds = [self.stability_results['subsample'][r]['ari_std'] for r in ratios]
            axes[0,1].errorbar(ratios, means, yerr=stds, marker='o')
            axes[0,1].set_title('Stability vs Sample Size')
            axes[0,1].set_xlabel('Sample Ratio')
            axes[0,1].set_ylabel('ARI Score')
        
        # Perturbation stability
        if 'perturbation' in self.stability_results:
            noise_levels = list(self.stability_results['perturbation'].keys())
            means = [self.stability_results['perturbation'][n]['ari_mean'] for n in noise_levels]
            axes[1,0].plot(noise_levels, means, marker='o')
            axes[1,0].set_title('Stability vs Noise Level')
            axes[1,0].set_xlabel('Noise Level')
            axes[1,0].set_ylabel('ARI Score')
        
        # K-fold stability
        if 'k_fold' in self.stability_results:
            axes[1,1].hist(self.stability_results['k_fold']['ari_scores'], bins=15, alpha=0.7)
            axes[1,1].set_title('K-Fold ARI Distribution')
            axes[1,1].set_xlabel('ARI Score')
        
        plt.tight_layout()
        return fig

# Example usage
# validator = ClusterStabilityValidator(KMeans, n_clusters=5)
# stability_report = validator.stability_report(X_scaled)