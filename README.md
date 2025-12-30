# Spotify Trending Music Data Analysis Project

## 📊 Project Overview

This project performs comprehensive data analysis on Spotify trending music data, collecting tracks from multiple playlists and markets worldwide. The analysis includes data collection, cleaning, exploratory data analysis (EDA), statistical analysis, and advanced analytics including clustering and PCA.

## 🎯 Key Features

- **Data Collection**: Automated collection from 50+ Spotify playlists across multiple markets
- **Data Cleaning**: Comprehensive preprocessing and duplicate removal
- **Exploratory Data Analysis**: Distribution analysis, correlation matrices, and visualizations
- **Statistical Analysis**: Hypothesis testing, feature engineering, clustering, and PCA
- **Comprehensive Reporting**: Professional markdown report with insights and visualizations

## 📈 Dataset Statistics

- **Total Tracks Collected**: 3,601 tracks
- **Unique Tracks**: 2,863 tracks (after duplicate removal)
- **Markets Covered**: 11+ countries (US, Japan, India, Brazil, UK, Germany, etc.)
- **Playlists Analyzed**: 50+ playlists including Top 50 and Viral 50 charts

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Data Collection

```bash
python collect_spotify_trending.py
```

This will collect data from multiple Spotify playlists. The script will:
- Search for playlists by name
- Collect track metadata and audio features
- Save data to timestamped CSV files

### Data Analysis

```bash
python data_analysis.py
```

This will run the complete analysis pipeline:
1. Data loading and cleaning
2. Exploratory Data Analysis (EDA)
3. Statistical analysis
4. Insight generation

### Generate Report

```bash
python generate_report.py
```

This generates a comprehensive markdown report (`DATA_ANALYSIS_REPORT.md`) with all findings and visualizations.

### IEEE LaTeX Report

The project includes an IEEE format LaTeX report (`IEEE_Report.tex`) ready for academic submission.

**Compile the LaTeX report:**
```bash
./compile_latex.sh
```

Or manually:
```bash
pdflatex IEEE_Report.tex
pdflatex IEEE_Report.tex  # Second pass for references
```

**Note:** Requires a LaTeX distribution (TeX Live, MacTeX, or MiKTeX).

## 📁 Project Structure

```
.
├── collect_spotify_trending.py    # Data collection script
├── data_analysis.py               # Main analysis pipeline
├── generate_report.py             # Report generation script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── DATA_ANALYSIS_REPORT.md        # Comprehensive markdown report
├── IEEE_Report.tex                # IEEE format LaTeX report
├── compile_latex.sh              # LaTeX compilation script
├── .gitignore                     # Git ignore file
├── analysis_output/               # Analysis results
│   ├── basic_statistics.csv      # Descriptive statistics
│   ├── insights.txt              # Text summary
│   └── *.png                     # Visualization files (10 figures)
└── [CSV files excluded - can be regenerated]
```

## 📊 Analysis Outputs

All analysis outputs are saved in the `analysis_output/` directory:

### Visualizations
- `feature_distributions.png` - Distribution of key audio features
- `correlation_matrix.png` - Correlation heatmap of audio features
- `top_artists.png` - Top 20 artists by track count
- `collaboration_distribution.png` - Collaboration vs solo tracks
- `market_distribution.png` - Track distribution by market
- `playlist_distribution.png` - Track distribution by playlist
- `release_year_trend.png` - Release year trends
- `release_month_distribution.png` - Release month patterns
- `popularity_vs_features.png` - Popularity vs audio features scatter plots
- `mood_distribution.png` - Mood classification
- `audio_features_boxplot.png` - Box plots of audio features

### Data Files
- `processed_data.csv` - Cleaned and processed dataset with engineered features
- `basic_statistics.csv` - Descriptive statistics for all numeric features
- `insights.json` - Structured insights in JSON format
- `insights.txt` - Human-readable insights summary

## 🔑 Key Findings

### Top Artists
- **Most Featured Artist**: Nusrat Fateh Ali Khan (51 tracks)
- Other top artists include Umur Anil Gokdag, Karan Aujla, The Weeknd, Ariana Grande, and Taylor Swift

### Market Distribution
- **Top Markets**: US (247 tracks), Japan (134), India (127), Brazil (107), UK (96)
- Data collected from 11+ countries worldwide

### Collaboration Analysis
- **Collaboration Rate**: 38.8% of tracks are collaborations
- Mean popularity: Collaborations (49.54) vs Solo (47.77)

### Temporal Trends
- **Peak Release Year**: 2025 (705 tracks), 2024 (500 tracks), 2023 (510 tracks)
- Most releases in February (306 tracks) and January (294 tracks)

## 🛠️ Technical Details

### Technologies Used
- **Python 3.x**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Scikit-learn** - Machine learning (clustering, PCA)
- **Spotipy** - Spotify Web API access
- **SciPy** - Statistical analysis

### API Configuration

The project uses Spotify Web API. Configure your credentials:

```python
SPOTIFY_CLIENT_ID = "your_client_id"
SPOTIFY_CLIENT_SECRET = "your_client_secret"
```

Or set environment variables:
```bash
export SPOTIFY_CLIENT_ID="your_client_id"
export SPOTIFY_CLIENT_SECRET="your_client_secret"
```

## 📝 Analysis Pipeline

1. **Data Collection**
   - Search for playlists by name
   - Collect track metadata (name, artist, album, popularity, etc.)
   - Fetch audio features (danceability, energy, valence, etc.)
   - Handle pagination for large playlists

2. **Data Cleaning**
   - Remove duplicates based on track_id
   - Convert data types
   - Handle missing values
   - Create derived features (duration in minutes, release year, etc.)

3. **Exploratory Data Analysis**
   - Descriptive statistics
   - Distribution analysis
   - Correlation analysis
   - Market and playlist analysis
   - Temporal analysis

4. **Statistical Analysis**
   - Hypothesis testing (collaborations vs solo tracks)
   - Feature engineering (mood classification, popularity categories)
   - K-means clustering
   - Principal Component Analysis (PCA)

5. **Insight Generation**
   - Pattern identification
   - Trend analysis
   - Business insights

## 📄 Reports

The project includes two comprehensive reports:

### 1. Markdown Report (`DATA_ANALYSIS_REPORT.md`)
- Executive Summary
- Methodology
- Data Collection Details
- Data Cleaning Process
- Exploratory Data Analysis
- Statistical Analysis
- Key Insights & Findings
- Visualizations
- Conclusions & Recommendations

### 2. IEEE LaTeX Report (`IEEE_Report.tex`)
- Academic format (IEEE conference paper style)
- Two-column layout
- 10 figures properly referenced
- 2 statistical tables
- Complete methodology and results sections
- Ready for academic submission
- Compile to PDF using `./compile_latex.sh`

## 🔗 GitHub Repository

> **Note**: The complete project code, data collection scripts, and analysis pipeline are available in the GitHub repository. Due to dataset size, the raw data files are not included in the repository but can be regenerated using the collection scripts.

**Repository Link**: https://github.com/umerkhaliid/Comprehensive-Analysis-of-Spotify-Trending-Music-Data-A-Multi-Market-Study-of-Global-Music-Trends

## 📊 Dataset Information

- **Source**: Spotify Web API
- **Collection Date**: December 2025
- **Total Records**: 3,601 tracks
- **Unique Tracks**: 2,863 tracks
- **Features**: 30+ features including track metadata, audio features, and derived features

## 🤝 Contributing

This is an academic/research project. For questions or contributions, please refer to the GitHub repository.

## 📜 License

This project is for educational and research purposes.

## 👤 Author

Umer Khalid

---

**Last Updated**: December 2025

