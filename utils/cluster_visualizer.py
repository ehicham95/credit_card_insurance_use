import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import umap.umap_ as umap

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
        feature_var_df = feature_var_df[feature_var_df['variance'] >= 0.5]  # Filter out features with zero variance
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
        ax4.set_title('Cluster Profiles (Top Discriminative Features)', fontsize=14, fontweight='bold')
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

    def create_2d_visualizations(self, df_scaled, cluster_labels):
        # Convert cluster_labels to numeric if needed
        if not np.issubdtype(cluster_labels.dtype, np.number):
            cluster_labels_numeric = pd.factorize(cluster_labels)[0]
        else:
            cluster_labels_numeric = cluster_labels

        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df_scaled)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        tsne_result = tsne.fit_transform(df_scaled)

        # Create subplots
        fig = make_subplots(rows=1, cols=2, subplot_titles=(
            f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.3f})',
            't-SNE Visualization'
        ))

        # Add PCA trace
        fig.add_trace(
            go.Scatter(
                x=pca_result[:, 0],
                y=pca_result[:, 1],
                mode='markers',
                marker=dict(
                    color=cluster_labels_numeric,
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True
                ),
                name='PCA'
            ),
            row=1, col=1
        )

        # Add t-SNE trace
        fig.add_trace(
            go.Scatter(
                x=tsne_result[:, 0],
                y=tsne_result[:, 1],
                mode='markers',
                marker=dict(
                    color=cluster_labels_numeric,
                    colorscale='Viridis',
                    opacity=0.6,
                    showscale=True
                ),
                name='t-SNE'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=1200,
            title_text="Cluster Visualization with PCA and t-SNE",
            scene=dict(
                xaxis_title='PC1',
                yaxis_title='PC2'
            ),
            scene2=dict(
                xaxis_title='t-SNE1',
                yaxis_title='t-SNE2'
            )
        )

        # Update xaxis and yaxis for PCA plot
        fig.update_xaxes(title_text='PC1', row=1, col=1)
        fig.update_yaxes(title_text='PC2', row=1, col=1)

        # Update xaxis and yaxis for t-SNE plot
        fig.update_xaxes(title_text='t-SNE1', row=1, col=2)
        fig.update_yaxes(title_text='t-SNE2', row=1, col=2)

        # Show plot
        fig.show(renderer='vscode')

        return pca_result, tsne_result
    
    def create_umap_visualization(self, df_scaled, cluster_labels, n_neighbors=15, min_dist=0.1):
        # Convert cluster_labels to numeric if needed
        
        if isinstance(cluster_labels, pd.DataFrame):
            cluster_labels = cluster_labels.squeeze()  # or cluster_labels.iloc[:, 0]
        elif not np.issubdtype(cluster_labels.dtype, np.number):
            cluster_labels_numeric = pd.factorize(cluster_labels)[0]
        else:
            cluster_labels_numeric = cluster_labels

        # Initialize UMAP
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=42
        )

        # Fit and transform
        umap_result = umap_reducer.fit_transform(df_scaled)

        # Create Plotly figure
        fig = go.Figure()

        # Add UMAP trace
        fig.add_trace(go.Scatter(
            x=umap_result[:, 0],
            y=umap_result[:, 1],
            mode='markers',
            marker=dict(
                color=cluster_labels_numeric,
                colorscale='Viridis',
                opacity=0.6,
                size=5,
                showscale=True
            ),
            text=cluster_labels_numeric,  # This can be used for hover text
            hoverinfo='text'
        ))

        # Update layout
        fig.update_layout(
            title='UMAP Visualization of Payment Behavior Clusters',
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            height=600,
            width=800
        )

        # Show plot
        fig.show(renderer='vscode')
        
        return umap_result
    
    def generate_business_recommendations(self, df_engineered, cluster_column='cluster_labels'):
        cluster_profiles = df_engineered.groupby(cluster_column).agg({
            'PRC_FULL_PAYMENT': 'mean',
            'BALANCE_FREQUENCY': 'mean',
            'effective_utilization': 'mean',
            'cash_advance_dependency': 'mean',
            'payment_risk_score': 'mean',
            'PURCHASES': 'mean',
            'BALANCE': 'mean',
            'revolver_flag': 'mean'
        }).round(3)

        print("=== CLUSTER BUSINESS PROFILES & RECOMMENDATIONS ===\n")

        for cluster in sorted(df_engineered[cluster_column].unique()):
            profile = cluster_profiles.loc[cluster]
            cluster_size = (df_engineered[cluster_column] == cluster).sum()
            cluster_pct = cluster_size / len(df_engineered) * 100
            
            # Define distinct strategies based on actual data
            if cluster == 0:  # Cash advance dependent, high risk
                cluster_name = "Cash Advance Addicts"
                strategy = "INTERVENTION & RISK MITIGATION"
                actions = [
                    "• Immediate credit limit reviews and restrictions",
                    "• Offer debt counseling and financial literacy programs",
                    "• Convert to secured cards or close high-risk accounts",
                    "• Partner with financial wellness programs"
                ]
                
            elif cluster == 1:  # Low activity, low purchases
                cluster_name = "Dormant Low-Value"
                strategy = "ACTIVATION OR ATTRITION"
                actions = [
                    "• Send reactivation campaigns with spending bonuses",
                    "• Offer no-fee cards to reduce maintenance costs",
                    "• Consider account closure for cost reduction",
                    "• Target with basic reward categories"
                ]
                
            elif cluster == 2:  # High spenders, high CA, revolvers
                cluster_name = "High-Spend Revolvers"
                strategy = "PROFIT MAXIMIZATION"
                actions = [
                    "• Optimize interest rates for maximum revenue",
                    "• Offer premium cards with higher limits",
                    "• Monitor for overlimit and increase fees accordingly",
                    "• Upsell balance transfer and installment products"
                ]
                
            elif cluster == 3:  # High purchases, pay full, active
                cluster_name = "Premium Transactors"
                strategy = "RETENTION & PREMIUM SERVICES"
                actions = [
                    "• Offer premium rewards and cashback programs",
                    "• Provide concierge and travel benefits",
                    "• Cross-sell wealth management services",
                    "• Protect from competitor poaching with exclusive perks"
                ]
                
            else:  # cluster == 4: High utilization, high purchases, revolvers
                cluster_name = "High-Value Revolvers"
                strategy = "RELATIONSHIP DEEPENING"
                actions = [
                    "• Increase credit limits proactively",
                    "• Offer balance consolidation loans",
                    "• Provide financial planning services",
                    "• Target for multiple product relationships"
                ]
            
            print(f"CLUSTER {cluster}: {cluster_name}")
            print(f"Size: {cluster_size:,} customers ({cluster_pct:.1f}%)")
            print(f"Strategy: {strategy}")
            print(f"\nKey Metrics:")
            print(f"  Full Payment: {profile['PRC_FULL_PAYMENT']:.1%}")
            print(f"  Activity: {profile['BALANCE_FREQUENCY']:.2f}")
            print(f"  Risk Score: {profile['payment_risk_score']:.0f}")
            print(f"  Avg Purchases: ${profile['PURCHASES']:.0f}")
            print(f"  Cash Advance Dependency: {profile['cash_advance_dependency']:.1%}")
            
            print(f"\nActions:")
            for action in actions:
                print(action)
            print("\n" + "="*60 + "\n")


