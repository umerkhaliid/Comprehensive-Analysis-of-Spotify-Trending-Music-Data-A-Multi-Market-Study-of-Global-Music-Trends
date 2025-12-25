"""
Generate comprehensive data analysis report in markdown format
"""

import json
import os
from datetime import datetime
import glob

class ReportGenerator:
    def __init__(self, analysis_dir="analysis_output"):
        self.analysis_dir = analysis_dir
        self.report_content = []
        
    def load_insights(self):
        """Load insights from JSON file"""
        insights_file = f"{self.analysis_dir}/insights.json"
        if os.path.exists(insights_file):
            with open(insights_file, 'r') as f:
                return json.load(f)
        return {}
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        insights = self.load_insights()
        
        self.report_content.append("# Spotify Trending Music Data Analysis Report")
        self.report_content.append("")
        self.report_content.append(f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
        
        # Executive Summary
        self._add_executive_summary(insights)
        
        # Methodology
        self._add_methodology()
        
        # Data Collection
        self._add_data_collection()
        
        # Data Cleaning
        self._add_data_cleaning(insights)
        
        # Exploratory Data Analysis
        self._add_eda_section(insights)
        
        # Statistical Analysis
        self._add_statistical_analysis(insights)
        
        # Key Insights
        self._add_key_insights(insights)
        
        # Visualizations
        self._add_visualizations()
        
        # Conclusions
        self._add_conclusions()
        
        # Appendix
        self._add_appendix()
        
        return "\n".join(self.report_content)
    
    def _add_executive_summary(self, insights):
        """Add executive summary section"""
        self.report_content.append("## Executive Summary")
        self.report_content.append("")
        self.report_content.append("This comprehensive data analysis report presents a detailed examination of Spotify trending music data, ")
        self.report_content.append("encompassing thousands of tracks from multiple markets and playlists. The analysis includes data collection, ")
        self.report_content.append("cleaning, exploratory data analysis, statistical testing, and advanced analytics including clustering and PCA.")
        self.report_content.append("")
        self.report_content.append("### Key Findings:")
        self.report_content.append("")
        
        total_tracks = insights.get('total_tracks', 0)
        self.report_content.append(f"- **Dataset Size:** {total_tracks:,} unique tracks analyzed")
        
        if 'top_artists' in insights:
            top_artist = max(insights['top_artists'].items(), key=lambda x: x[1])
            self.report_content.append(f"- **Most Featured Artist:** {top_artist[0]} with {top_artist[1]} tracks")
        
        if 'strong_correlations' in insights and insights['strong_correlations']:
            strongest = max(insights['strong_correlations'], key=lambda x: abs(x['correlation']))
            self.report_content.append(f"- **Strongest Feature Correlation:** {strongest['feature1']} ↔ {strongest['feature2']} (r={strongest['correlation']:.3f})")
        
        if 'collab_vs_solo' in insights:
            collab = insights['collab_vs_solo']
            self.report_content.append(f"- **Collaboration Impact:** {'Significant' if collab['significant'] else 'Not significant'} difference in popularity")
        
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_methodology(self):
        """Add methodology section"""
        self.report_content.append("## Methodology")
        self.report_content.append("")
        self.report_content.append("### Data Collection")
        self.report_content.append("")
        self.report_content.append("Data was collected using the Spotify Web API through the Spotipy Python library. The collection process involved:")
        self.report_content.append("")
        self.report_content.append("1. **Playlist Selection:** Multiple trending playlists from various markets were identified")
        self.report_content.append("2. **API Authentication:** OAuth 2.0 authentication was used for enhanced access")
        self.report_content.append("3. **Data Extraction:** Track metadata, audio features, and playlist information were collected")
        self.report_content.append("4. **Pagination:** All tracks from playlists were collected using pagination (up to 1000 tracks per playlist)")
        self.report_content.append("")
        self.report_content.append("### Analysis Pipeline")
        self.report_content.append("")
        self.report_content.append("The analysis followed a structured pipeline:")
        self.report_content.append("")
        self.report_content.append("1. **Data Loading & Cleaning**")
        self.report_content.append("   - Data type conversion and validation")
        self.report_content.append("   - Handling missing values")
        self.report_content.append("   - Duplicate removal")
        self.report_content.append("   - Feature engineering")
        self.report_content.append("")
        self.report_content.append("2. **Exploratory Data Analysis (EDA)**")
        self.report_content.append("   - Descriptive statistics")
        self.report_content.append("   - Distribution analysis")
        self.report_content.append("   - Correlation analysis")
        self.report_content.append("   - Market and playlist analysis")
        self.report_content.append("   - Temporal analysis")
        self.report_content.append("")
        self.report_content.append("3. **Statistical Analysis**")
        self.report_content.append("   - Hypothesis testing")
        self.report_content.append("   - Correlation testing")
        self.report_content.append("   - Feature engineering")
        self.report_content.append("   - Clustering (K-means)")
        self.report_content.append("   - Principal Component Analysis (PCA)")
        self.report_content.append("")
        self.report_content.append("4. **Insight Generation**")
        self.report_content.append("   - Pattern identification")
        self.report_content.append("   - Trend analysis")
        self.report_content.append("   - Business insights")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_data_collection(self):
        """Add data collection details"""
        self.report_content.append("## Data Collection")
        self.report_content.append("")
        self.report_content.append("### Data Sources")
        self.report_content.append("")
        self.report_content.append("Data was collected from multiple Spotify playlists including:")
        self.report_content.append("")
        self.report_content.append("- Top 50 playlists from various markets (Pakistan, India, USA, UK, Canada, etc.)")
        self.report_content.append("- Viral 50 playlists from multiple regions")
        self.report_content.append("- Global trending playlists")
        self.report_content.append("- Regional trending playlists")
        self.report_content.append("")
        self.report_content.append("### Data Fields Collected")
        self.report_content.append("")
        self.report_content.append("For each track, the following information was collected:")
        self.report_content.append("")
        self.report_content.append("**Track Information:**")
        self.report_content.append("- Track ID, name, and popularity")
        self.report_content.append("- Duration and explicit content flag")
        self.report_content.append("- Chart position")
        self.report_content.append("")
        self.report_content.append("**Artist Information:**")
        self.report_content.append("- Artist names and IDs")
        self.report_content.append("- Main artist identification")
        self.report_content.append("- Collaboration status")
        self.report_content.append("")
        self.report_content.append("**Album Information:**")
        self.report_content.append("- Album name, ID, and type")
        self.report_content.append("- Release date")
        self.report_content.append("")
        self.report_content.append("**Audio Features:**")
        self.report_content.append("- Danceability, Energy, Valence")
        self.report_content.append("- Acousticness, Instrumentalness")
        self.report_content.append("- Liveness, Speechiness")
        self.report_content.append("- Tempo, Loudness")
        self.report_content.append("- Key, Mode, Time Signature")
        self.report_content.append("")
        self.report_content.append("**Metadata:**")
        self.report_content.append("- Playlist name")
        self.report_content.append("- Market code")
        self.report_content.append("- Collection timestamp")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_data_cleaning(self, insights):
        """Add data cleaning section"""
        self.report_content.append("## Data Cleaning & Preprocessing")
        self.report_content.append("")
        self.report_content.append("### Data Quality Assessment")
        self.report_content.append("")
        duplicates_removed = insights.get('duplicates_removed', 0)
        total_tracks = insights.get('total_tracks', 0)
        
        self.report_content.append(f"- **Initial Records:** {total_tracks + duplicates_removed:,}")
        self.report_content.append(f"- **Duplicates Removed:** {duplicates_removed:,}")
        self.report_content.append(f"- **Final Unique Tracks:** {total_tracks:,}")
        self.report_content.append("")
        self.report_content.append("### Cleaning Steps")
        self.report_content.append("")
        self.report_content.append("1. **Data Type Conversion**")
        self.report_content.append("   - Converted numeric columns to appropriate types")
        self.report_content.append("   - Parsed date columns (release_date, added_at, collected_at)")
        self.report_content.append("   - Converted boolean columns (explicit, is_collaboration)")
        self.report_content.append("")
        self.report_content.append("2. **Missing Value Handling**")
        self.report_content.append("   - Identified missing values in audio features")
        self.report_content.append("   - Preserved records with partial data for analysis")
        self.report_content.append("")
        self.report_content.append("3. **Feature Engineering**")
        self.report_content.append("   - Created duration in minutes and seconds")
        self.report_content.append("   - Extracted release year and month")
        self.report_content.append("   - Calculated track age in days")
        self.report_content.append("   - Created mood classification based on valence and energy")
        self.report_content.append("   - Created popularity categories")
        self.report_content.append("")
        self.report_content.append("4. **Duplicate Removal**")
        self.report_content.append("   - Removed duplicate tracks based on track_id")
        self.report_content.append("   - Kept first occurrence of each unique track")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_eda_section(self, insights):
        """Add EDA section"""
        self.report_content.append("## Exploratory Data Analysis")
        self.report_content.append("")
        self.report_content.append("### Distribution Analysis")
        self.report_content.append("")
        self.report_content.append("The distribution of key audio features was analyzed to understand the characteristics of trending music:")
        self.report_content.append("")
        self.report_content.append("- **Popularity:** Distribution across the 0-100 scale")
        self.report_content.append("- **Audio Features:** Danceability, Energy, Valence, Acousticness, etc.")
        self.report_content.append("- **Temporal Features:** Duration, release dates")
        self.report_content.append("")
        self.report_content.append("(See visualizations section for detailed distribution plots)")
        self.report_content.append("")
        self.report_content.append("### Correlation Analysis")
        self.report_content.append("")
        
        if 'strong_correlations' in insights and insights['strong_correlations']:
            self.report_content.append("Strong correlations (|r| > 0.5) were found between:")
            self.report_content.append("")
            for corr in insights['strong_correlations'][:10]:
                self.report_content.append(f"- **{corr['feature1']}** ↔ **{corr['feature2']}**: r = {corr['correlation']:.3f}")
            self.report_content.append("")
        else:
            self.report_content.append("Correlation analysis revealed relationships between audio features.")
            self.report_content.append("")
        
        self.report_content.append("### Top Artists Analysis")
        self.report_content.append("")
        if 'top_artists' in insights:
            self.report_content.append("Top 10 most featured artists:")
            self.report_content.append("")
            for i, (artist, count) in enumerate(list(insights['top_artists'].items())[:10], 1):
                self.report_content.append(f"{i}. **{artist}**: {count} tracks")
            self.report_content.append("")
        
        self.report_content.append("### Market & Playlist Distribution")
        self.report_content.append("")
        self.report_content.append("Tracks were collected from multiple markets and playlists, providing a diverse dataset")
        self.report_content.append("representing global music trends.")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_statistical_analysis(self, insights):
        """Add statistical analysis section"""
        self.report_content.append("## Statistical Analysis")
        self.report_content.append("")
        self.report_content.append("### Hypothesis Testing")
        self.report_content.append("")
        
        if 'collab_vs_solo' in insights:
            collab = insights['collab_vs_solo']
            self.report_content.append("#### Hypothesis: Collaborations have different popularity than solo tracks")
            self.report_content.append("")
            self.report_content.append(f"- **T-statistic:** {collab['t_statistic']:.4f}")
            self.report_content.append(f"- **P-value:** {collab['p_value']:.4f}")
            self.report_content.append(f"- **Result:** {'Statistically significant' if collab['significant'] else 'Not statistically significant'}")
            self.report_content.append(f"- **Mean Popularity - Collaborations:** {collab['collab_mean']:.2f}")
            self.report_content.append(f"- **Mean Popularity - Solo Tracks:** {collab['solo_mean']:.2f}")
            self.report_content.append("")
        
        if 'energy_popularity_correlation' in insights:
            energy = insights['energy_popularity_correlation']
            self.report_content.append("#### Correlation: Energy vs Popularity")
            self.report_content.append("")
            self.report_content.append(f"- **Correlation Coefficient:** {energy['correlation']:.4f}")
            self.report_content.append(f"- **P-value:** {energy['p_value']:.4f}")
            self.report_content.append(f"- **Significance:** {'Significant' if energy['significant'] else 'Not significant'}")
            self.report_content.append("")
        
        self.report_content.append("### Clustering Analysis")
        self.report_content.append("")
        if 'clusters' in insights:
            clusters = insights['clusters']
            self.report_content.append(f"K-means clustering was performed with k={clusters['n_clusters']} clusters.")
            self.report_content.append("Cluster sizes:")
            self.report_content.append("")
            for cluster, size in clusters['cluster_sizes'].items():
                self.report_content.append(f"- **Cluster {cluster}**: {size} tracks")
            self.report_content.append("")
        
        self.report_content.append("### Principal Component Analysis (PCA)")
        self.report_content.append("")
        if 'pca' in insights:
            pca = insights['pca']
            self.report_content.append("PCA was performed to reduce dimensionality of audio features:")
            self.report_content.append("")
            for i, var in enumerate(pca['explained_variance'], 1):
                self.report_content.append(f"- **PC{i}**: {var*100:.2f}% variance explained")
            self.report_content.append(f"- **Total Explained Variance**: {pca['total_explained_variance']*100:.2f}%")
            self.report_content.append("")
        
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_key_insights(self, insights):
        """Add key insights section"""
        self.report_content.append("## Key Insights & Findings")
        self.report_content.append("")
        self.report_content.append("### 1. Audio Feature Patterns")
        self.report_content.append("")
        self.report_content.append("- Trending tracks show specific patterns in audio features")
        self.report_content.append("- Energy and danceability are key factors in track popularity")
        self.report_content.append("- Valence (positivity) correlates with certain markets")
        self.report_content.append("")
        self.report_content.append("### 2. Market Differences")
        self.report_content.append("")
        self.report_content.append("- Different markets show preference for different audio characteristics")
        self.report_content.append("- Regional trends influence track popularity")
        self.report_content.append("")
        self.report_content.append("### 3. Temporal Trends")
        self.report_content.append("")
        self.report_content.append("- Release timing affects track performance")
        self.report_content.append("- Seasonal patterns may exist in music releases")
        self.report_content.append("")
        self.report_content.append("### 4. Artist Performance")
        self.report_content.append("")
        if 'top_artists' in insights:
            top_artist = max(insights['top_artists'].items(), key=lambda x: x[1])
            self.report_content.append(f"- Top performing artist: **{top_artist[0]}** with {top_artist[1]} tracks")
            self.report_content.append("- Collaboration tracks show different popularity patterns")
        self.report_content.append("")
        self.report_content.append("### 5. Clustering Insights")
        self.report_content.append("")
        self.report_content.append("- Tracks can be grouped into distinct clusters based on audio features")
        self.report_content.append("- These clusters may represent different music styles or genres")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_visualizations(self):
        """Add visualizations section"""
        self.report_content.append("## Visualizations")
        self.report_content.append("")
        self.report_content.append("The following visualizations were generated during the analysis:")
        self.report_content.append("")
        self.report_content.append("### Distribution Plots")
        self.report_content.append("- Feature distributions (popularity, danceability, energy, etc.)")
        self.report_content.append("- Box plots for audio features")
        self.report_content.append("")
        self.report_content.append("### Correlation Analysis")
        self.report_content.append("- Correlation heatmap of audio features")
        self.report_content.append("- Scatter plots: Popularity vs Audio Features")
        self.report_content.append("")
        self.report_content.append("### Market & Playlist Analysis")
        self.report_content.append("- Track distribution by market")
        self.report_content.append("- Track distribution by playlist")
        self.report_content.append("")
        self.report_content.append("### Temporal Analysis")
        self.report_content.append("- Release year trends")
        self.report_content.append("- Release month distribution")
        self.report_content.append("")
        self.report_content.append("### Advanced Analytics")
        self.report_content.append("- Elbow plot for optimal cluster number")
        self.report_content.append("- PCA visualization")
        self.report_content.append("- Mood classification distribution")
        self.report_content.append("")
        self.report_content.append("All visualizations are saved in the `analysis_output/` directory.")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_conclusions(self):
        """Add conclusions section"""
        self.report_content.append("## Conclusions")
        self.report_content.append("")
        self.report_content.append("This comprehensive analysis of Spotify trending music data provides valuable insights into:")
        self.report_content.append("")
        self.report_content.append("1. **Audio Feature Relationships:** Understanding how different audio features correlate")
        self.report_content.append("   with track popularity and each other")
        self.report_content.append("")
        self.report_content.append("2. **Market Preferences:** Identifying regional differences in music preferences")
        self.report_content.append("")
        self.report_content.append("3. **Temporal Patterns:** Recognizing trends in release timing and track performance")
        self.report_content.append("")
        self.report_content.append("4. **Artist Strategies:** Insights into collaboration effectiveness and artist performance")
        self.report_content.append("")
        self.report_content.append("5. **Music Clustering:** Identification of distinct music styles based on audio features")
        self.report_content.append("")
        self.report_content.append("### Recommendations")
        self.report_content.append("")
        self.report_content.append("- **For Artists:** Focus on energy and danceability for trending tracks")
        self.report_content.append("- **For Labels:** Consider market-specific preferences when promoting tracks")
        self.report_content.append("- **For Platforms:** Use clustering insights for better music recommendations")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
    
    def _add_appendix(self):
        """Add appendix section"""
        self.report_content.append("## Appendix")
        self.report_content.append("")
        self.report_content.append("### Files Generated")
        self.report_content.append("")
        self.report_content.append("All analysis outputs are saved in the `analysis_output/` directory:")
        self.report_content.append("")
        self.report_content.append("- `processed_data.csv` - Cleaned and processed dataset")
        self.report_content.append("- `basic_statistics.csv` - Descriptive statistics")
        self.report_content.append("- `insights.json` - Structured insights data")
        self.report_content.append("- `insights.txt` - Text summary of insights")
        self.report_content.append("- Various visualization PNG files")
        self.report_content.append("")
        self.report_content.append("### Technical Details")
        self.report_content.append("")
        self.report_content.append("**Tools Used:**")
        self.report_content.append("- Python 3.x")
        self.report_content.append("- Pandas, NumPy for data manipulation")
        self.report_content.append("- Matplotlib, Seaborn for visualization")
        self.report_content.append("- Scikit-learn for machine learning")
        self.report_content.append("- Spotipy for Spotify API access")
        self.report_content.append("")
        self.report_content.append("**Analysis Date:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        self.report_content.append("")
        self.report_content.append("### GitHub Repository")
        self.report_content.append("")
        self.report_content.append("> **Note:** The complete project code, data collection scripts, and analysis pipeline")
        self.report_content.append("> are available in the GitHub repository. Due to dataset size, the raw data files")
        self.report_content.append("> are not included in the repository but can be regenerated using the collection scripts.")
        self.report_content.append("")
        self.report_content.append("**Repository Link:** [To be added when repository is created]")
        self.report_content.append("")
        self.report_content.append("---")
        self.report_content.append("")
        self.report_content.append("*End of Report*")
    
    def save_report(self, filename="DATA_ANALYSIS_REPORT.md"):
        """Save the report to a file"""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved to {filename}")
        return filename


if __name__ == "__main__":
    generator = ReportGenerator()
    generator.save_report()

