"""
Comprehensive Spotify Data Analysis Pipeline
Performs complete data analysis including EDA, statistical analysis, and visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
import os
import glob
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import json

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class SpotifyDataAnalyzer:
    def __init__(self, data_file=None):
        """Initialize the analyzer with data file"""
        if data_file:
            self.df = pd.read_csv(data_file)
        else:
            # Find the most recent CSV file (try multiple patterns)
            csv_files = glob.glob("spotify_trending*.csv") + glob.glob("viral50*.csv") + glob.glob("*.csv")
            if csv_files:
                # Filter out analysis output files
                csv_files = [f for f in csv_files if 'analysis_output' not in f and 'processed_data' not in f]
                if csv_files:
                    self.df = pd.read_csv(max(csv_files, key=os.path.getctime))
                else:
                    raise FileNotFoundError("No Spotify data file found")
            else:
                raise FileNotFoundError("No Spotify data file found")
        
        self.output_dir = "analysis_output"
        os.makedirs(self.output_dir, exist_ok=True)
        self.insights = {}
        
    def load_and_clean_data(self):
        """Step 1: Load and clean the data"""
        print("="*80)
        print("STEP 1: DATA LOADING AND CLEANING")
        print("="*80)
        
        print(f"\nInitial dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Basic info
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nMissing values:\n{self.df.isnull().sum().sort_values(ascending=False)}")
        
        # Convert numeric columns
        numeric_cols = ['duration_ms', 'popularity', 'danceability', 'energy', 
                       'valence', 'acousticness', 'instrumentalness', 'liveness',
                       'speechiness', 'tempo', 'loudness', 'key', 'mode', 
                       'time_signature', 'chart_position']
        
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Convert date columns
        date_cols = ['release_date', 'added_at', 'collected_at']
        for col in date_cols:
            if col in self.df.columns:
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    pass
        
        # Handle explicit column
        if 'explicit' in self.df.columns:
            self.df['explicit'] = self.df['explicit'].astype(str).str.upper() == 'TRUE'
        
        # Create derived features
        if 'duration_ms' in self.df.columns:
            self.df['duration_minutes'] = self.df['duration_ms'] / 60000
            self.df['duration_seconds'] = self.df['duration_ms'] / 1000
        
        if 'release_date' in self.df.columns:
            self.df['release_year'] = pd.to_datetime(self.df['release_date'], errors='coerce').dt.year
            self.df['release_month'] = pd.to_datetime(self.df['release_date'], errors='coerce').dt.month
            self.df['age_days'] = (pd.Timestamp.now() - pd.to_datetime(self.df['release_date'], errors='coerce')).dt.days
        
        # Remove duplicates
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates(subset=['track_id'], keep='first')
        duplicates_removed = initial_count - len(self.df)
        print(f"\nRemoved {duplicates_removed} duplicate tracks")
        
        print(f"\nFinal dataset shape: {self.df.shape}")
        print(f"Total unique tracks: {len(self.df)}")
        
        self.insights['total_tracks'] = len(self.df)
        self.insights['duplicates_removed'] = duplicates_removed
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Step 2: Exploratory Data Analysis"""
        print("\n" + "="*80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS (EDA)")
        print("="*80)
        
        # Basic statistics
        print("\n--- Basic Statistics ---")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numeric_cols].describe())
        
        # Save statistics
        stats_df = self.df[numeric_cols].describe()
        stats_df.to_csv(f"{self.output_dir}/basic_statistics.csv")
        
        # Distribution of key features
        self._plot_distributions()
        
        # Correlation analysis
        self._plot_correlations()
        
        # Top artists analysis
        self._analyze_top_artists()
        
        # Market/Playlist analysis
        self._analyze_markets_playlists()
        
        # Temporal analysis
        self._temporal_analysis()
        
        # Genre/audio feature analysis
        self._audio_feature_analysis()
        
        return self.insights
    
    def _plot_distributions(self):
        """Plot distributions of key features"""
        print("\n--- Creating Distribution Plots ---")
        
        features = ['popularity', 'danceability', 'energy', 'valence', 
                   'acousticness', 'tempo', 'loudness', 'duration_minutes']
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(features):
            if feature in self.df.columns:
                ax = axes[idx]
                data = self.df[feature].dropna()
                ax.hist(data, bins=50, edgecolor='black', alpha=0.7)
                ax.set_title(f'Distribution of {feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_xlabel(feature.replace("_", " ").title())
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: feature_distributions.png")
    
    def _plot_correlations(self):
        """Plot correlation matrix"""
        print("\n--- Creating Correlation Analysis ---")
        
        audio_features = ['danceability', 'energy', 'valence', 'acousticness',
                         'instrumentalness', 'liveness', 'speechiness', 
                         'tempo', 'loudness', 'popularity']
        
        available_features = [f for f in audio_features if f in self.df.columns]
        corr_matrix = self.df[available_features].corr()
        
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8})
        plt.title('Correlation Matrix of Audio Features', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: correlation_matrix.png")
        
        # Store strong correlations
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.5:
                    corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        self.insights['strong_correlations'] = corr_pairs
    
    def _analyze_top_artists(self):
        """Analyze top artists"""
        print("\n--- Top Artists Analysis ---")
        
        if 'main_artist' in self.df.columns:
            top_artists = self.df['main_artist'].value_counts().head(20)
            print(f"\nTop 20 Artists:\n{top_artists}")
            
            plt.figure(figsize=(14, 8))
            top_artists.plot(kind='barh')
            plt.title('Top 20 Artists by Track Count', fontsize=16, fontweight='bold')
            plt.xlabel('Number of Tracks')
            plt.ylabel('Artist')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/top_artists.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: top_artists.png")
            
            self.insights['top_artists'] = top_artists.to_dict()
        
        # Collaboration analysis
        if 'is_collaboration' in self.df.columns:
            collab_stats = self.df['is_collaboration'].value_counts()
            print(f"\nCollaboration Statistics:\n{collab_stats}")
            
            plt.figure(figsize=(8, 6))
            collab_stats.plot(kind='pie', autopct='%1.1f%%', startangle=90)
            plt.title('Collaboration vs Solo Tracks', fontsize=14, fontweight='bold')
            plt.ylabel('')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/collaboration_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: collaboration_distribution.png")
    
    def _analyze_markets_playlists(self):
        """Analyze data by market and playlist"""
        print("\n--- Market and Playlist Analysis ---")
        
        if 'market' in self.df.columns:
            market_counts = self.df['market'].value_counts()
            print(f"\nTracks by Market:\n{market_counts}")
            
            plt.figure(figsize=(14, 8))
            market_counts.plot(kind='bar')
            plt.title('Number of Tracks by Market', fontsize=16, fontweight='bold')
            plt.xlabel('Market')
            plt.ylabel('Number of Tracks')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/market_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: market_distribution.png")
        
        if 'playlist' in self.df.columns:
            playlist_counts = self.df['playlist'].value_counts()
            print(f"\nTracks by Playlist:\n{playlist_counts}")
            
            plt.figure(figsize=(14, 8))
            playlist_counts.head(20).plot(kind='barh')
            plt.title('Top 20 Playlists by Track Count', fontsize=16, fontweight='bold')
            plt.xlabel('Number of Tracks')
            plt.ylabel('Playlist')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/playlist_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: playlist_distribution.png")
    
    def _temporal_analysis(self):
        """Analyze temporal patterns"""
        print("\n--- Temporal Analysis ---")
        
        if 'release_year' in self.df.columns:
            year_counts = self.df['release_year'].value_counts().sort_index()
            print(f"\nTracks by Release Year:\n{year_counts.tail(20)}")
            
            plt.figure(figsize=(14, 6))
            year_counts.plot(kind='line', marker='o')
            plt.title('Tracks Released by Year', fontsize=16, fontweight='bold')
            plt.xlabel('Year')
            plt.ylabel('Number of Tracks')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/release_year_trend.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: release_year_trend.png")
        
        if 'release_month' in self.df.columns:
            month_counts = self.df['release_month'].value_counts().sort_index()
            print(f"\nTracks by Release Month:\n{month_counts}")
            
            plt.figure(figsize=(12, 6))
            month_counts.plot(kind='bar')
            plt.title('Tracks Released by Month', fontsize=16, fontweight='bold')
            plt.xlabel('Month')
            plt.ylabel('Number of Tracks')
            plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=0)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/release_month_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: release_month_distribution.png")
    
    def _audio_feature_analysis(self):
        """Analyze audio features"""
        print("\n--- Audio Feature Analysis ---")
        
        features = ['danceability', 'energy', 'valence', 'acousticness', 
                   'instrumentalness', 'liveness', 'speechiness']
        
        available_features = [f for f in features if f in self.df.columns]
        
        if available_features:
            # Box plots
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            axes = axes.flatten()
            
            for idx, feature in enumerate(available_features[:8]):
                ax = axes[idx]
                data = self.df[feature].dropna()
                ax.boxplot(data, vert=True)
                ax.set_title(f'{feature.replace("_", " ").title()}', fontsize=12, fontweight='bold')
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
            
            # Hide unused subplots
            for idx in range(len(available_features), 8):
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/audio_features_boxplot.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: audio_features_boxplot.png")
            
            # Feature comparison
            if 'popularity' in self.df.columns:
                self._analyze_popularity_vs_features()
    
    def _analyze_popularity_vs_features(self):
        """Analyze relationship between popularity and audio features"""
        print("\n--- Popularity vs Audio Features ---")
        
        features = ['danceability', 'energy', 'valence', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']
        
        available_features = [f for f in features if f in self.df.columns]
        
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for idx, feature in enumerate(available_features[:9]):
            ax = axes[idx]
            data = self.df[[feature, 'popularity']].dropna()
            ax.scatter(data[feature], data['popularity'], alpha=0.5, s=20)
            ax.set_xlabel(feature.replace("_", " ").title())
            ax.set_ylabel('Popularity')
            ax.set_title(f'Popularity vs {feature.replace("_", " ").title()}', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(data) > 1:
                z = np.polyfit(data[feature], data['popularity'], 1)
                p = np.poly1d(z)
                ax.plot(data[feature], p(data[feature]), "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/popularity_vs_features.png", dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Saved: popularity_vs_features.png")
    
    def statistical_analysis(self):
        """Step 3: Statistical Analysis"""
        print("\n" + "="*80)
        print("STEP 3: STATISTICAL ANALYSIS")
        print("="*80)
        
        # Hypothesis testing
        self._hypothesis_testing()
        
        # Feature engineering
        self._feature_engineering()
        
        # Clustering analysis
        self._clustering_analysis()
        
        # PCA analysis
        self._pca_analysis()
        
        return self.insights
    
    def _hypothesis_testing(self):
        """Perform hypothesis testing"""
        print("\n--- Hypothesis Testing ---")
        
        # Test 1: Do collaborations have higher popularity?
        if 'is_collaboration' in self.df.columns and 'popularity' in self.df.columns:
            collab_pop = self.df[self.df['is_collaboration'] == True]['popularity'].dropna()
            solo_pop = self.df[self.df['is_collaboration'] == False]['popularity'].dropna()
            
            if len(collab_pop) > 0 and len(solo_pop) > 0:
                t_stat, p_value = stats.ttest_ind(collab_pop, solo_pop)
                print(f"\nHypothesis: Collaborations have different popularity than solo tracks")
                print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")
                print(f"Mean popularity - Collaborations: {collab_pop.mean():.2f}, Solo: {solo_pop.mean():.2f}")
                
                self.insights['collab_vs_solo'] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'collab_mean': float(collab_pop.mean()),
                    'solo_mean': float(solo_pop.mean()),
                    'significant': p_value < 0.05
                }
        
        # Test 2: Correlation between energy and popularity
        if 'energy' in self.df.columns and 'popularity' in self.df.columns:
            data = self.df[['energy', 'popularity']].dropna()
            if len(data) > 2:
                corr, p_value = stats.pearsonr(data['energy'], data['popularity'])
                print(f"\nCorrelation between Energy and Popularity:")
                print(f"Correlation: {corr:.4f}, P-value: {p_value:.4f}")
                
                self.insights['energy_popularity_correlation'] = {
                    'correlation': float(corr),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                }
    
    def _feature_engineering(self):
        """Create new features"""
        print("\n--- Feature Engineering ---")
        
        # Mood classification based on valence and energy
        if 'valence' in self.df.columns and 'energy' in self.df.columns:
            def classify_mood(row):
                v = row.get('valence', 0.5)
                e = row.get('energy', 0.5)
                if pd.isna(v) or pd.isna(e):
                    return 'Unknown'
                if v > 0.5 and e > 0.5:
                    return 'Happy/Energetic'
                elif v > 0.5 and e <= 0.5:
                    return 'Happy/Calm'
                elif v <= 0.5 and e > 0.5:
                    return 'Sad/Energetic'
                else:
                    return 'Sad/Calm'
            
            self.df['mood'] = self.df.apply(classify_mood, axis=1)
            mood_counts = self.df['mood'].value_counts()
            print(f"\nMood Classification:\n{mood_counts}")
            
            plt.figure(figsize=(10, 6))
            mood_counts.plot(kind='bar')
            plt.title('Track Mood Distribution', fontsize=14, fontweight='bold')
            plt.xlabel('Mood')
            plt.ylabel('Number of Tracks')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/mood_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: mood_distribution.png")
        
        # Popularity categories
        if 'popularity' in self.df.columns:
            self.df['popularity_category'] = pd.cut(
                self.df['popularity'],
                bins=[0, 30, 50, 70, 100],
                labels=['Low', 'Medium', 'High', 'Very High']
            )
            print(f"\nPopularity Categories:\n{self.df['popularity_category'].value_counts()}")
    
    def _clustering_analysis(self):
        """Perform K-means clustering on audio features"""
        print("\n--- Clustering Analysis ---")
        
        features = ['danceability', 'energy', 'valence', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness', 'tempo']
        
        available_features = [f for f in features if f in self.df.columns]
        data = self.df[available_features].dropna()
        
        if len(data) > 10:
            # Standardize features
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            # Find optimal number of clusters
            inertias = []
            K_range = range(2, 11)
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(scaled_data)
                inertias.append(kmeans.inertia_)
            
            # Elbow plot
            plt.figure(figsize=(10, 6))
            plt.plot(K_range, inertias, 'bo-')
            plt.xlabel('Number of Clusters (k)')
            plt.ylabel('Inertia')
            plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/elbow_plot.png", dpi=300, bbox_inches='tight')
            plt.close()
            print("✓ Saved: elbow_plot.png")
            
            # Perform clustering with k=5
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add clusters to dataframe
            cluster_indices = data.index
            self.df.loc[cluster_indices, 'audio_cluster'] = clusters
            
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            print(f"\nCluster Distribution:\n{cluster_counts}")
            
            self.insights['clusters'] = {
                'n_clusters': 5,
                'cluster_sizes': cluster_counts.to_dict()
            }
    
    def _pca_analysis(self):
        """Perform PCA on audio features"""
        print("\n--- PCA Analysis ---")
        
        features = ['danceability', 'energy', 'valence', 'acousticness',
                   'instrumentalness', 'liveness', 'speechiness', 'tempo', 'loudness']
        
        available_features = [f for f in features if f in self.df.columns]
        data = self.df[available_features].dropna()
        
        if len(data) > 10 and len(available_features) > 2:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(data)
            
            pca = PCA(n_components=min(3, len(available_features)))
            pca_result = pca.fit_transform(scaled_data)
            
            explained_var = pca.explained_variance_ratio_
            print(f"\nExplained Variance by Component:")
            for i, var in enumerate(explained_var):
                print(f"  PC{i+1}: {var:.4f} ({var*100:.2f}%)")
            print(f"Total Explained Variance: {sum(explained_var):.4f} ({sum(explained_var)*100:.2f}%)")
            
            # Plot PCA
            if pca_result.shape[1] >= 2:
                plt.figure(figsize=(12, 8))
                scatter = plt.scatter(pca_result[:, 0], pca_result[:, 1], 
                                     c=self.df.loc[data.index, 'popularity'] if 'popularity' in self.df.columns else None,
                                     cmap='viridis', alpha=0.6, s=30)
                plt.xlabel(f'PC1 ({explained_var[0]*100:.2f}% variance)')
                plt.ylabel(f'PC2 ({explained_var[1]*100:.2f}% variance)')
                plt.title('PCA: Audio Features (colored by popularity)', fontsize=14, fontweight='bold')
                plt.colorbar(scatter, label='Popularity')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{self.output_dir}/pca_plot.png", dpi=300, bbox_inches='tight')
                plt.close()
                print("✓ Saved: pca_plot.png")
            
            self.insights['pca'] = {
                'explained_variance': [float(v) for v in explained_var],
                'total_explained_variance': float(sum(explained_var))
            }
    
    def generate_insights(self):
        """Step 4: Generate comprehensive insights"""
        print("\n" + "="*80)
        print("STEP 4: GENERATING INSIGHTS")
        print("="*80)
        
        insights_text = []
        
        # Dataset overview
        insights_text.append(f"## Dataset Overview\n")
        insights_text.append(f"- Total unique tracks: {self.insights.get('total_tracks', len(self.df))}")
        insights_text.append(f"- Duplicates removed: {self.insights.get('duplicates_removed', 0)}")
        insights_text.append(f"- Total features: {len(self.df.columns)}")
        
        # Top insights
        if 'top_artists' in self.insights:
            top_artist = max(self.insights['top_artists'].items(), key=lambda x: x[1])
            insights_text.append(f"\n## Top Artists\n")
            insights_text.append(f"- Most featured artist: {top_artist[0]} ({top_artist[1]} tracks)")
        
        # Correlation insights
        if 'strong_correlations' in self.insights:
            insights_text.append(f"\n## Strong Correlations\n")
            for corr in self.insights['strong_correlations'][:5]:
                insights_text.append(f"- {corr['feature1']} ↔ {corr['feature2']}: {corr['correlation']:.3f}")
        
        # Statistical insights
        if 'collab_vs_solo' in self.insights:
            collab_insight = self.insights['collab_vs_solo']
            insights_text.append(f"\n## Collaboration Analysis\n")
            if collab_insight['significant']:
                insights_text.append(f"- Collaborations have {'higher' if collab_insight['collab_mean'] > collab_insight['solo_mean'] else 'lower'} popularity (statistically significant)")
            insights_text.append(f"- Mean popularity: Collaborations={collab_insight['collab_mean']:.2f}, Solo={collab_insight['solo_mean']:.2f}")
        
        # Save insights
        with open(f"{self.output_dir}/insights.txt", 'w') as f:
            f.write('\n'.join(insights_text))
        
        # Save insights as JSON (convert numpy types to native Python types)
        def convert_to_serializable(obj):
            if isinstance(obj, (np.integer, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (bool, np.bool_)):
                return bool(obj)
            return obj
        
        serializable_insights = convert_to_serializable(self.insights)
        with open(f"{self.output_dir}/insights.json", 'w') as f:
            json.dump(serializable_insights, f, indent=2)
        
        print("\n".join(insights_text))
        print(f"\n✓ Insights saved to {self.output_dir}/")
        
        return insights_text
    
    def save_processed_data(self):
        """Save processed dataset"""
        output_file = f"{self.output_dir}/processed_data.csv"
        self.df.to_csv(output_file, index=False)
        print(f"\n✓ Processed data saved to {output_file}")
        return output_file
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("\n" + "="*80)
        print("SPOTIFY DATA ANALYSIS PIPELINE")
        print("="*80)
        print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Step 1: Load and clean
        self.load_and_clean_data()
        
        # Step 2: EDA
        self.exploratory_data_analysis()
        
        # Step 3: Statistical analysis
        self.statistical_analysis()
        
        # Step 4: Generate insights
        self.generate_insights()
        
        # Step 5: Save processed data
        self.save_processed_data()
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETE!")
        print("="*80)
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"\nAll outputs saved to: {self.output_dir}/")
        print("="*80)


if __name__ == "__main__":
    # Run analysis
    analyzer = SpotifyDataAnalyzer()
    analyzer.run_complete_analysis()

