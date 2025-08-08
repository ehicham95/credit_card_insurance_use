import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class ClusterVisualizer:
    discriminatory_features = None

    def __init__(self, data, cluster_labels, risk_features, optimal_k):
        self.data = data
        self.cluster_labels = cluster_labels
        self.risk_features = risk_features
        self.optimal_k = optimal_k
        self.df = data.copy()
        self.df['cluster'] = cluster_labels
        
    def plot_cluster_overview(self, figsize=(15, 12)):
        """Comprehensive overview of clusters"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Cluster sizes with log scale
        cluster_counts = self.df['cluster'].value_counts().sort_index()
        ax1 = axes[0, 0]
        bars = ax1.bar(cluster_counts.index, cluster_counts.values, 
                      color=sns.color_palette("husl", len(cluster_counts)))
        ax1.set_yscale('log')
        ax1.set_title('Cluster Sizes (Log Scale)', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Cluster ID')
        ax1.set_ylabel('Count (log scale)')
        
        # Add value labels on bars
        for bar, count in zip(bars, cluster_counts.values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + count*0.1,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Cluster percentages
        ax2 = axes[0, 1]
        percentages = (cluster_counts / len(self.df)) * 100
        wedges, texts, autotexts = ax2.pie(percentages.values, 
                                          labels=[f'Cluster {i}' for i in percentages.index],
                                          autopct='%1.1f%%', startangle=90,
                                          colors=sns.color_palette("husl", len(percentages)))
        ax2.set_title('Cluster Distribution', fontsize=14, fontweight='bold')
        
        # 3. Feature variance by cluster (helps identify discriminative features)
        feature_vars = []
        for feature in self.risk_features:
            cluster_means = self.df.groupby('cluster')[feature].mean()
            feature_vars.append(cluster_means.var())
        
        ax3 = axes[1, 0]
        feature_var_df = pd.DataFrame({'feature': self.risk_features, 'variance': feature_vars})
        feature_var_df = feature_var_df[feature_var_df['variance'] > 0.1]  # Filter out features with zero variance
        feature_var_df = feature_var_df.sort_values('variance', ascending=True)
        discriminatory_features = feature_var_df['feature'].tolist()
        
        bars = ax3.barh(feature_var_df['feature'], feature_var_df['variance'])
        ax3.set_title('Feature Discrimination Power\n(Variance of Cluster Means)', 
                     fontsize=14, fontweight='bold')
        ax3.set_xlabel('Variance')
        
        # 4. Silhouette-like analysis (distance from cluster centroids)
        ax4 = axes[1, 1]
        cluster_stats = self.df.groupby('cluster').agg({
            feat: ['mean', 'std'] for feat in self.risk_features[:5]  # Top 5 features
        }).round(2)
        
        # Create a simple heatmap of standardized means
        cluster_means = self.df.groupby('cluster')[discriminatory_features].mean()
        scaler = StandardScaler()
        standardized_means = pd.DataFrame(
            scaler.fit_transform(cluster_means),
            index=cluster_means.index,
            columns=cluster_means.columns
        )
        
        sns.heatmap(standardized_means, annot=True, cmap='RdYlBu_r', 
                   center=0, ax=ax4, cbar_kws={'label': 'Standardized Mean'})
        ax4.set_title('Cluster Profiles (Top 5 Features)', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Features')
        ax4.set_ylabel('Cluster ID')
        
        plt.tight_layout()
        return fig, discriminatory_features
    
    def plot_feature_distributions(self, features=None, figsize=(20, 15)):
        """Box plots for feature distributions with small cluster emphasis"""
        if features is None:
            features = self.risk_features
        
        n_features = len(features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_rows > 1 else [axes]
        
        colors = sns.color_palette("husl", len(self.df['cluster'].unique()))
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # Create box plot with custom properties for small clusters
            box_data = [self.df[self.df['cluster'] == cluster][feature].values 
                       for cluster in sorted(self.df['cluster'].unique())]
            
            bp = ax.boxplot(box_data, patch_artist=True, 
                           labels=[f'C{i}' for i in sorted(self.df['cluster'].unique())])
            
            # Color boxes and make small clusters more prominent
            for patch, color, cluster_id in zip(bp['boxes'], colors, sorted(self.df['cluster'].unique())):
                patch.set_facecolor(color)
                cluster_size = sum(self.df['cluster'] == cluster_id)
                # Make small clusters more visible with thicker borders
                if cluster_size < 500:
                    patch.set_linewidth(3)
                    patch.set_edgecolor('black')
                else:
                    patch.set_linewidth(1)
            
            ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3)
            
        # Remove empty subplots
        for i in range(n_features, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        return fig
    
    