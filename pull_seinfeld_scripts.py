"""
Script to pull Seinfeld scripts from IMSDb.
Returns a dict of dict of string with structure:
{
    "season 1": {
        "Episode Name": "full script content",
        ...
    },
    ...
}
"""

import requests
from bs4 import BeautifulSoup
from typing import Dict
import time
import json
import re


def mark_speaker_names(script_content: str) -> str:
    """
    Adds a pound sign before character names in screenplay format.
    Character names are typically in ALL CAPS at the beginning of a line.
    This helps distinguish when a character is speaking vs being mentioned in dialogue.
    
    Args:
        script_content: The raw script text
        
    Returns:
        Script with # added before speaker character names
    """
    lines = script_content.split('\n')
    processed_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Check if this line looks like a character name
        # Character names are in ALL CAPS, possibly with (V.O.), (O.S.), (CONT'D), etc.
        if (stripped and 
            stripped.isupper() and 
            not stripped.startswith('#') and 
            len(stripped) > 1 and
            not any(char.isdigit() for char in stripped.replace('.', '')) and
            any(char.isalpha() for char in stripped)):
            
            # Add pound sign before the speaker name
            indent = len(line) - len(line.lstrip())
            line = ' ' * indent + '#' + stripped
        
        processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def pull_seinfeld_scripts() -> Dict[str, Dict[str, str]]:
    """
    Pulls all Seinfeld scripts from IMSDb.
    
    Returns:
        Dict[str, Dict[str, str]]: Nested dictionary with structure
            {season: {episode_name: script_content}}
    """
    base_url = "https://imsdb.com/TV/Seinfeld.html"
    scripts = {}
    
    print("Fetching main page...")
    response = requests.get(base_url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all season headers
    season_headers = soup.find_all('h2')
    
    for idx, header in enumerate(season_headers):
        header_text = header.get_text().strip()
        
        # Check if this is a "Series X" header
        if header_text.startswith("Series "):
            season_number = int(header_text.split()[-1])
            current_season = f"season {season_number}"
            scripts[current_season] = {}
            print(f"Found {current_season}")
            
            # Find next season header to know when to stop
            next_header = season_headers[idx + 1] if idx + 1 < len(season_headers) else None
            
            # Get all paragraphs between this header and the next one
            current = header.find_next_sibling('p')
            while current:
                # Stop if we hit the next season header
                if next_header and current == next_header:
                    break
                
                # Get the link inside the paragraph
                link = current.find('a')
                if link:
                    href = link.get('href', '')
                    episode_title = link.get_text().strip()
                    
                    # Check if it's a Seinfeld script link
                    if 'Seinfeld' in href and 'Script' in href:
                        print(f"  Fetching: {episode_title}")
                        script = fetch_episode_script(href)
                        if script:
                            scripts[current_season][episode_title] = script
                            time.sleep(0.5)  # Be respectful to the server
                
                next_sibling = current.find_next_sibling()
                # Stop if we hit the next season header
                if next_sibling == next_header:
                    break
                current = next_sibling
    
    return scripts


def fetch_episode_script(script_url: str) -> str:
    """
    Fetches the script content from an episode URL.
    
    Args:
        script_url: The URL to the episode transcript summary page (can be relative)
        
    Returns:
        The script content as a string
    """
    try:
        # Ensure we have a full URL for the transcript summary page
        if script_url.startswith('/'):
            summary_url = 'https://imsdb.com' + script_url
        elif not script_url.startswith('http'):
            summary_url = 'https://imsdb.com/' + script_url
        else:
            summary_url = script_url
            
        # Step 1: Fetch the transcript summary page to find the actual script link
        response = requests.get(summary_url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for the "Read ... Script" link
        actual_script_url = None
        links = soup.find_all('a')
        for link in links:
            link_text = link.get_text()
            if 'Read' in link_text and 'Script' in link_text:
                actual_script_url = link.get('href')
                break
        
        if not actual_script_url:
            return ""
        
        # Convert to full URL if needed
        if actual_script_url.startswith('/'):
            actual_script_url = 'https://imsdb.com' + actual_script_url
        elif not actual_script_url.startswith('http'):
            actual_script_url = 'https://imsdb.com/' + actual_script_url
        
        # Step 2: Fetch the actual script page
        script_response = requests.get(actual_script_url)
        script_response.raise_for_status()
        script_soup = BeautifulSoup(script_response.content, 'html.parser')
        
        # Find the script content in the td with class "scrtext"
        script_td = script_soup.find('td', {'class': 'scrtext'})
        
        if script_td:
            script_content = script_td.get_text()
            # Add pound signs before character speaker names
            script_content = mark_speaker_names(script_content)
            return script_content.strip()
        else:
            return ""
    except Exception as e:
        print(f"    Error fetching {script_url}: {e}")
        return ""


if __name__ == "__main__":
    scripts = pull_seinfeld_scripts()
    
    # Print summary
    total_episodes = sum(len(eps) for eps in scripts.values())
    print(f"\n✓ Successfully fetched {total_episodes} episodes across {len(scripts)} seasons")
    
    # Display structure
    for season, episodes in scripts.items():
        print(f"  {season}: {len(episodes)} episodes")
    
    # Save to JSON file
    output_file = "seinfeld_scripts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(scripts, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Scripts saved to {output_file}")
