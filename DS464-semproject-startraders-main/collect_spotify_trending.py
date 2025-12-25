# collect_spotify_viral.py
import time
import json
import csv
import os
from datetime import datetime
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth
from spotipy.exceptions import SpotifyException

# -----------------------------
# CONFIG
# -----------------------------
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID", "f20c45b4407949b4a8df3e440a4eea4d")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET", "f35a11d15d8842b3812f404b64cc31dc")
REDIRECT_URI = "http://127.0.0.1:8888/callback"

USE_OAUTH = False  # Try client credentials first for automated collection
OUTFILE = "spotify_trending_data.csv"

# Enhanced playlist collection - multiple playlists and markets
VIRAL_PLAYLISTS = {
    "TRENDING PAKISTANI SONGS 2025": "TRENDING PAKISTANI SONGS 2025",
    "Top 50 - Pakistan": "Top 50 - Pakistan",
    "Viral 50 - Pakistan": "Viral 50 - Pakistan",
    "Today's Top Hits": "Today's Top Hits",
    "Global Top 50": "Global Top 50",
    "Top 50 - India": "Top 50 - India",
    "Viral 50 - India": "Viral 50 - India",
    "Top 50 - USA": "Top 50 - USA",
    "Viral 50 - USA": "Viral 50 - USA",
    "Top 50 - UK": "Top 50 - UK",
    "Viral 50 - UK": "Viral 50 - UK",
    "Top 50 - Canada": "Top 50 - Canada",
    "Viral 50 - Canada": "Viral 50 - Canada",
    "Top 50 - Australia": "Top 50 - Australia",
    "Viral 50 - Australia": "Viral 50 - Australia",
    "Top 50 - Germany": "Top 50 - Germany",
    "Viral 50 - Germany": "Viral 50 - Germany",
    "Top 50 - France": "Top 50 - France",
    "Viral 50 - France": "Viral 50 - France",
    "Top 50 - Brazil": "Top 50 - Brazil",
    "Viral 50 - Brazil": "Viral 50 - Brazil",
    "Top 50 - Mexico": "Top 50 - Mexico",
    "Viral 50 - Mexico": "Viral 50 - Mexico",
    "Top 50 - Japan": "Top 50 - Japan",
    "Viral 50 - Japan": "Viral 50 - Japan",
    "Top 50 - South Korea": "Top 50 - South Korea",
    "Viral 50 - South Korea": "Viral 50 - South Korea",
    "Top 50 - UAE": "Top 50 - UAE",
    "Viral 50 - UAE": "Viral 50 - UAE",
    "Top 50 - Saudi Arabia": "Top 50 - Saudi Arabia",
    "Viral 50 - Saudi Arabia": "Viral 50 - Saudi Arabia",
    "Top 50 - Turkey": "Top 50 - Turkey",
    "Viral 50 - Turkey": "Viral 50 - Turkey",
    "Top 50 - Russia": "Top 50 - Russia",
    "Viral 50 - Russia": "Viral 50 - Russia",
    "Top 50 - Spain": "Top 50 - Spain",
    "Viral 50 - Spain": "Viral 50 - Spain",
    "Top 50 - Italy": "Top 50 - Italy",
    "Viral 50 - Italy": "Viral 50 - Italy",
    "Top 50 - Netherlands": "Top 50 - Netherlands",
    "Viral 50 - Netherlands": "Viral 50 - Netherlands",
    "Top 50 - Sweden": "Top 50 - Sweden",
    "Viral 50 - Sweden": "Viral 50 - Sweden",
    "Top 50 - Norway": "Top 50 - Norway",
    "Viral 50 - Norway": "Viral 50 - Norway",
    "Top 50 - Poland": "Top 50 - Poland",
    "Viral 50 - Poland": "Viral 50 - Poland",
    "Top 50 - Argentina": "Top 50 - Argentina",
    "Viral 50 - Argentina": "Viral 50 - Argentina",
    "Top 50 - Chile": "Top 50 - Chile",
    "Viral 50 - Chile": "Viral 50 - Chile",
    "Top 50 - Colombia": "Top 50 - Colombia",
    "Viral 50 - Colombia": "Viral 50 - Colombia",
    "Top 50 - Indonesia": "Top 50 - Indonesia",
    "Viral 50 - Indonesia": "Viral 50 - Indonesia",
    "Top 50 - Philippines": "Top 50 - Philippines",
    "Viral 50 - Philippines": "Viral 50 - Philippines",
    "Top 50 - Thailand": "Top 50 - Thailand",
    "Viral 50 - Thailand": "Viral 50 - Thailand",
    "Top 50 - Malaysia": "Top 50 - Malaysia",
    "Viral 50 - Malaysia": "Viral 50 - Malaysia",
    "Top 50 - Singapore": "Top 50 - Singapore",
    "Viral 50 - Singapore": "Viral 50 - Singapore",
    "Top 50 - Egypt": "Top 50 - Egypt",
    "Viral 50 - Egypt": "Viral 50 - Egypt",
    "Top 50 - South Africa": "Top 50 - South Africa",
    "Viral 50 - South Africa": "Viral 50 - South Africa",
    "Top 50 - Nigeria": "Top 50 - Nigeria",
    "Viral 50 - Nigeria": "Viral 50 - Nigeria",
}

# Market codes for different regions
MARKETS = ["PK", "IN", "US", "GB", "CA", "AU", "DE", "FR", "BR", "MX", "JP", "KR", "AE", "SA", "TR", "RU", "ES", "IT", "NL", "SE", "NO", "PL", "AR", "CL", "CO", "ID", "PH", "TH", "MY", "SG", "EG", "ZA", "NG"]

# -----------------------------
# AUTHENTICATION
# -----------------------------
def init_spotify_client():
    if USE_OAUTH:
        print("→ Using OAuth (Recommended)")
        return spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET,
            redirect_uri=REDIRECT_URI,
            scope="playlist-read-private playlist-read-collaborative"
        ))
    else:
        print("→ Using Client Credentials (Limited)")
        return spotipy.Spotify(
            client_credentials_manager=SpotifyClientCredentials(
                client_id=SPOTIFY_CLIENT_ID,
                client_secret=SPOTIFY_CLIENT_SECRET
            )
        )


sp = init_spotify_client()

# ----------------------------------------------------------
# 1. Dynamic playlist lookup
# ----------------------------------------------------------
def get_playlist_id_by_name(search_name):
    try:
        print(f"→ Searching Spotify for playlist: {search_name}...")
        result = sp.search(q=search_name, type="playlist", limit=5)  # Get more results
        
        if result is None:
            print(f"✗ Search returned None for: {search_name}")
            return None
            
        playlists = result.get("playlists", {})
        if not playlists:
            print(f"✗ No playlists object in result for: {search_name}")
            return None
            
        items = playlists.get("items", [])
        if not items:
            print(f"✗ Playlist not found: {search_name}")
            return None
        
        # Try to find the best match
        playlist_id = None
        playlist_name = None
        for item in items:
            if item and item.get("id"):
                playlist_id = item["id"]
                playlist_name = item.get("name", "Unknown")
                # Prefer official Spotify playlists
                if "spotify" in item.get("owner", {}).get("id", "").lower():
                    break
        
        if playlist_id:
            print(f"✓ Found playlist: {playlist_name} → ID={playlist_id}")
            return playlist_id
        else:
            print(f"✗ No valid playlist ID found for: {search_name}")
            return None
        
    except Exception as e:
        print(f"✗ Playlist search failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# ----------------------------------------------------------
# 2. Safe audio feature fetch
# ----------------------------------------------------------
def fetch_audio_features_safe(track_ids):
    track_ids = [t for t in track_ids if t]
    if not track_ids:
        return []
    
    # Process in batches of 100 (API limit)
    all_features = []
    batch_size = 100
    
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i+batch_size]
        try:
            features = sp.audio_features(batch)
            if features:
                all_features.extend(features)
            else:
                all_features.extend([None] * len(batch))
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"✗ Failed audio features batch {i//batch_size + 1}: {e}")
            all_features.extend([None] * len(batch))
    
    return all_features


# ----------------------------------------------------------
# 3. Fetch playlist items with pagination
# ----------------------------------------------------------
def fetch_viral50(sp, playlist_search_name, market="PK", max_tracks=1000):
    playlist_id = get_playlist_id_by_name(playlist_search_name)
    if not playlist_id:
        return []

    print(f"\nFetching from playlist: {playlist_search_name} ({market})")

    all_raw_items = []
    offset = 0
    limit = 100  # Maximum allowed by API
    
    try:
        while offset < max_tracks:
            results = sp.playlist_items(
                playlist_id,
                market=market,
                limit=limit,
                offset=offset,
                fields="items(track(id,name,artists,album,duration_ms,popularity,explicit),added_at)"
            )
            
            items = results.get("items", [])
            if not items:
                break
                
            all_raw_items.extend(items)
            
            # Check if there are more items
            if len(items) < limit:
                break
                
            offset += limit
            time.sleep(0.1)  # Reduced rate limiting
            
    except SpotifyException as e:
        print(f"✗ Spotify API error ({e.http_status}): {e.msg}")
        return []
    except Exception as e:
        print(f"✗ General playlist fetch error: {e}")
        return []

    if not all_raw_items:
        print("✗ No tracks found in playlist")
        return []

    print(f"✓ Found {len(all_raw_items)} tracks")

    track_data, track_ids = [], []

    for idx, item in enumerate(all_raw_items, start=1):
        track = item.get("track")
        if not track:
            continue

        artists = [a.get("name") for a in track.get("artists", [])]
        artist_ids = [a.get("id") for a in track.get("artists", [])]

        album = track.get("album", {})

        data = {
            "track_id": track.get("id"),
            "track_name": track.get("name"),
            "artists": ", ".join(artists),
            "artist_ids": ", ".join(artist_ids),
            "main_artist": artists[0] if artists else None,
            "is_collaboration": len(artists) > 1,

            "album_name": album.get("name"),
            "album_id": album.get("id"),
            "release_date": album.get("release_date"),
            "album_type": album.get("album_type"),

            "duration_ms": track.get("duration_ms"),
            "popularity": track.get("popularity"),
            "explicit": track.get("explicit"),

            "chart_position": idx,
            "added_at": item.get("added_at"),

            "collected_at": datetime.utcnow().isoformat(),
            "playlist": playlist_search_name,
            "market": market
        }

        track_data.append(data)
        track_ids.append(track.get("id"))

        print(f"  #{idx} → {data['track_name']}")

    print("\n→ Fetching audio features...")
    try:
        audio_features = fetch_audio_features_safe(track_ids)
        
        for item, features in zip(track_data, audio_features):
            if features:
                item.update(features)
            else:
                for k in [
                    "danceability","energy","valence","acousticness",
                    "instrumentalness","liveness","speechiness",
                    "key","mode","tempo","time_signature","loudness"
                ]:
                    item[k] = None
    except Exception as e:
        print(f"⚠ Warning: Could not fetch audio features: {e}")
        print("  Continuing with track metadata only...")
        # Add empty audio features
        for item in track_data:
            for k in [
                "danceability","energy","valence","acousticness",
                "instrumentalness","liveness","speechiness",
                "key","mode","tempo","time_signature","loudness"
            ]:
                item[k] = None

    print("✓ Completed playlist fetch\n")
    return track_data


# ----------------------------------------------------------
# 4. Save CSV
# ----------------------------------------------------------
def save_csv(data):
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = f"spotify_trending{timestamp}.csv"

    # Collect all keys to form the CSV header
    fieldnames = sorted({key for row in data for key in row.keys()})

    with open(fname, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    print(f"✓ CSV saved → {fname}")


# ----------------------------------------------------------
# 5. Full collection runner
# ----------------------------------------------------------
def collect():
    all_tracks = []
    total_collected = 0
    failed_playlists = []

    for key, playlist_search in VIRAL_PLAYLISTS.items():
        print("\n" + "─" * 60)
        print(f"Processing: {key}")
        try:
            # Try different markets for each playlist
            tracks = []
            
            # Extract market code from playlist name if possible
            market_codes = {
                "Pakistan": "PK", "India": "IN", "USA": "US", "UK": "GB",
                "Canada": "CA", "Australia": "AU", "Germany": "DE", "France": "FR",
                "Brazil": "BR", "Mexico": "MX", "Japan": "JP", "South Korea": "KR"
            }
            
            market = None
            for country, code in market_codes.items():
                if country in playlist_search:
                    market = code
                    break
            
            # Try with market first
            if market:
                tracks = fetch_viral50(sp, playlist_search_name=playlist_search, market=market, max_tracks=1000)
            
            # If that fails, try without market
            if not tracks:
                tracks = fetch_viral50(sp, playlist_search_name=playlist_search, market=None, max_tracks=1000)
            
            # If still no tracks, try with PK market as fallback
            if not tracks:
                tracks = fetch_viral50(sp, playlist_search_name=playlist_search, market="PK", max_tracks=1000)
            
            if tracks:
                all_tracks.extend(tracks)
                total_collected += len(tracks)
                print(f"✓ Collected {len(tracks)} tracks (Total: {total_collected})")
            else:
                failed_playlists.append(key)
                print(f"✗ Failed to collect from: {key}")
        except Exception as e:
            print(f"✗ Error collecting from {key}: {e}")
            failed_playlists.append(key)
        
        time.sleep(0.5)  # Reduced rate limiting for faster collection
        
        # Save progress periodically
        if len(all_tracks) > 0 and len(all_tracks) % 1000 == 0:
            print(f"\n→ Progress save: {len(all_tracks)} tracks collected so far...")

    if all_tracks:
        print(f"\n{'='*60}")
        print(f"✓ Total tracks collected: {len(all_tracks)}")
        print(f"✗ Failed playlists: {len(failed_playlists)}")
        if failed_playlists:
            print(f"  {', '.join(failed_playlists[:10])}")
        print(f"{'='*60}\n")
        save_csv(all_tracks)
    else:
        print("✗ No data collected at all.")


# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------
if __name__ == "__main__":
    print("\n==============================")
    print(" SPOTIFY VIRAL 50 COLLECTOR")
    print("==============================\n")

    collect()
