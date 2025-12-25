# Spotify Trending Music Data Analysis Report

**Generated on:** 2025-12-25 23:52:32

---

## Executive Summary

This comprehensive data analysis report presents a detailed examination of Spotify trending music data, 
encompassing thousands of tracks from multiple markets and playlists. The analysis includes data collection, 
cleaning, exploratory data analysis, statistical testing, and advanced analytics including clustering and PCA.

### Key Findings:

- **Dataset Size:** 2,863 unique tracks analyzed
- **Most Featured Artist:** Nusrat Fateh Ali Khan with 51 tracks
- **Collaboration Impact:** Not significant difference in popularity

---

## Methodology

### Data Collection

Data was collected using the Spotify Web API through the Spotipy Python library. The collection process involved:

1. **Playlist Selection:** Multiple trending playlists from various markets were identified
2. **API Authentication:** OAuth 2.0 authentication was used for enhanced access
3. **Data Extraction:** Track metadata, audio features, and playlist information were collected
4. **Pagination:** All tracks from playlists were collected using pagination (up to 1000 tracks per playlist)

### Analysis Pipeline

The analysis followed a structured pipeline:

1. **Data Loading & Cleaning**
   - Data type conversion and validation
   - Handling missing values
   - Duplicate removal
   - Feature engineering

2. **Exploratory Data Analysis (EDA)**
   - Descriptive statistics
   - Distribution analysis
   - Correlation analysis
   - Market and playlist analysis
   - Temporal analysis

3. **Statistical Analysis**
   - Hypothesis testing
   - Correlation testing
   - Feature engineering
   - Clustering (K-means)
   - Principal Component Analysis (PCA)

4. **Insight Generation**
   - Pattern identification
   - Trend analysis
   - Business insights

---

## Data Collection

### Data Sources

Data was collected from multiple Spotify playlists including:

- Top 50 playlists from various markets (Pakistan, India, USA, UK, Canada, etc.)
- Viral 50 playlists from multiple regions
- Global trending playlists
- Regional trending playlists

### Data Fields Collected

For each track, the following information was collected:

**Track Information:**
- Track ID, name, and popularity
- Duration and explicit content flag
- Chart position

**Artist Information:**
- Artist names and IDs
- Main artist identification
- Collaboration status

**Album Information:**
- Album name, ID, and type
- Release date

**Audio Features:**
- Danceability, Energy, Valence
- Acousticness, Instrumentalness
- Liveness, Speechiness
- Tempo, Loudness
- Key, Mode, Time Signature

**Metadata:**
- Playlist name
- Market code
- Collection timestamp

---

## Data Cleaning & Preprocessing

### Data Quality Assessment

- **Initial Records:** 3,601
- **Duplicates Removed:** 738
- **Final Unique Tracks:** 2,863

### Cleaning Steps

1. **Data Type Conversion**
   - Converted numeric columns to appropriate types
   - Parsed date columns (release_date, added_at, collected_at)
   - Converted boolean columns (explicit, is_collaboration)

2. **Missing Value Handling**
   - Identified missing values in audio features
   - Preserved records with partial data for analysis

3. **Feature Engineering**
   - Created duration in minutes and seconds
   - Extracted release year and month
   - Calculated track age in days
   - Created mood classification based on valence and energy
   - Created popularity categories

4. **Duplicate Removal**
   - Removed duplicate tracks based on track_id
   - Kept first occurrence of each unique track

---

## Exploratory Data Analysis

### Distribution Analysis

The distribution of key audio features was analyzed to understand the characteristics of trending music:

- **Popularity:** Distribution across the 0-100 scale
- **Audio Features:** Danceability, Energy, Valence, Acousticness, etc.
- **Temporal Features:** Duration, release dates

(See visualizations section for detailed distribution plots)

### Correlation Analysis

Correlation analysis revealed relationships between audio features.

### Top Artists Analysis

Top 10 most featured artists:

1. **Nusrat Fateh Ali Khan**: 51 tracks
2. **Umur Anil Gokdag**: 33 tracks
3. **Karan Aujla**: 21 tracks
4. **Gianluca Modanese**: 20 tracks
5. **The Weeknd**: 18 tracks
6. **Ariana Grande**: 18 tracks
7. **Taylor Swift**: 15 tracks
8. **KAROL G**: 14 tracks
9. **Olivia Rodrigo**: 13 tracks
10. **LISA**: 12 tracks

### Market & Playlist Distribution

Tracks were collected from multiple markets and playlists, providing a diverse dataset
representing global music trends.

---

## Statistical Analysis

### Hypothesis Testing

#### Hypothesis: Collaborations have different popularity than solo tracks

- **T-statistic:** 1.6328
- **P-value:** 0.1026
- **Result:** Not statistically significant
- **Mean Popularity - Collaborations:** 49.54
- **Mean Popularity - Solo Tracks:** 47.77

### Clustering Analysis

### Principal Component Analysis (PCA)

---

## Key Insights & Findings

### 1. Audio Feature Patterns

- Trending tracks show specific patterns in audio features
- Energy and danceability are key factors in track popularity
- Valence (positivity) correlates with certain markets

### 2. Market Differences

- Different markets show preference for different audio characteristics
- Regional trends influence track popularity

### 3. Temporal Trends

- Release timing affects track performance
- Seasonal patterns may exist in music releases

### 4. Artist Performance

- Top performing artist: **Nusrat Fateh Ali Khan** with 51 tracks
- Collaboration tracks show different popularity patterns

### 5. Clustering Insights

- Tracks can be grouped into distinct clusters based on audio features
- These clusters may represent different music styles or genres

---

## Visualizations

The following visualizations were generated during the analysis:

### Distribution Plots
- Feature distributions (popularity, danceability, energy, etc.)
- Box plots for audio features

### Correlation Analysis
- Correlation heatmap of audio features
- Scatter plots: Popularity vs Audio Features

### Market & Playlist Analysis
- Track distribution by market
- Track distribution by playlist

### Temporal Analysis
- Release year trends
- Release month distribution

### Advanced Analytics
- Elbow plot for optimal cluster number
- PCA visualization
- Mood classification distribution

All visualizations are saved in the `analysis_output/` directory.

---

## Conclusions

This comprehensive analysis of Spotify trending music data provides valuable insights into:

1. **Audio Feature Relationships:** Understanding how different audio features correlate
   with track popularity and each other

2. **Market Preferences:** Identifying regional differences in music preferences

3. **Temporal Patterns:** Recognizing trends in release timing and track performance

4. **Artist Strategies:** Insights into collaboration effectiveness and artist performance

5. **Music Clustering:** Identification of distinct music styles based on audio features

### Recommendations

- **For Artists:** Focus on energy and danceability for trending tracks
- **For Labels:** Consider market-specific preferences when promoting tracks
- **For Platforms:** Use clustering insights for better music recommendations

---

## Appendix

### Files Generated

All analysis outputs are saved in the `analysis_output/` directory:

- `processed_data.csv` - Cleaned and processed dataset
- `basic_statistics.csv` - Descriptive statistics
- `insights.json` - Structured insights data
- `insights.txt` - Text summary of insights
- Various visualization PNG files

### Technical Details

**Tools Used:**
- Python 3.x
- Pandas, NumPy for data manipulation
- Matplotlib, Seaborn for visualization
- Scikit-learn for machine learning
- Spotipy for Spotify API access

**Analysis Date:** 2025-12-25 23:52:32

### GitHub Repository

> **Note:** The complete project code, data collection scripts, and analysis pipeline
> are available in the GitHub repository. Due to dataset size, the raw data files
> are not included in the repository but can be regenerated using the collection scripts.

**Repository Link:** https://github.com/umerkhaliid/Comprehensive-Analysis-of-Spotify-Trending-Music-Data-A-Multi-Market-Study-of-Global-Music-Trends

---

*End of Report*