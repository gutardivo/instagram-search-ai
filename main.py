import instaloader
from instaloader.exceptions import TwoFactorAuthRequiredException, ProfileNotExistsException, PrivateProfileNotFollowedException
import os
import urllib.request
from image_classification import classify_by_desire
from PIL import Image
from io import BytesIO
import requests
from time import sleep

# Initialize Instaloader instance
L = instaloader.Instaloader()

# Login details (avoid hardcoding sensitive information)
username = os.getenv('INSTAGRAM_USERNAME')
password = os.getenv('INSTAGRAM_PASSWORD')

def login_instagram():
    """Login to Instagram using provided credentials and handle 2FA if required."""
    try:
        L.login(username, password)
    except TwoFactorAuthRequiredException:
        input_code = input("Enter 2FA Code: ")
        L.two_factor_login(input_code)

def load_visited_profiles(filename):
    """Load visited profiles from a file."""
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            return set(line.strip() for line in file)
    return set()

def save_visited_profiles(profiles, filename):
    """Save visited profiles to a file."""
    with open(filename, 'w') as file:
        for profile in profiles:
            file.write(f"{profile}\n")

def load_image_from_url(url):
    """Load an image from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status() 
        image = Image.open(BytesIO(response.content)) 
        return image
    except Exception as e:
        print(f"Error loading image from URL {url}: {e}")
        return None

def save_image(url, folder, filename):
    """Save image from URL to a specified folder."""
    filepath = os.path.join(folder, filename)
    urllib.request.urlretrieve(url, filepath)

def download_profile_and_last_post(profile):
    """Download profile picture and last post for a given profile."""
    profile_pic_url = profile.profile_pic_url
    save_image(profile_pic_url, base_dir, profile.username+'.jpg')

def image_classify(image_url, desired_gender, desired_hair_color=None):
    """Classify image based on desired characteristics."""
    image = load_image_from_url(image_url)
    if image is None:
        print(f"Failed to load image from {image_url}")
        return False
    return classify_by_desire(image, desired_gender, desired_hair_color)

def download_followers_images(profile):
    """Download images from profile's followers if they match desired characteristics."""
    visited_profiles.add(profile.username)
    save_visited_profiles(visited_profiles, visited_profiles_file)

    for follower in profile.get_followers():
        if follower.username not in visited_profiles:
            try:
                profile_pic_url = follower.profile_pic_url
                if image_classify(profile_pic_url, "female"):
                    follower_profile = instaloader.Profile.from_username(L.context, follower.username)
                    download_profile_and_last_post(follower_profile)
            except ProfileNotExistsException:
                print(f"Profile {follower.username} does not exist.")
            except PrivateProfileNotFollowedException:
                print(f"Profile {follower.username} is private.")

def download_followees_images(profile):
    """Download images from profile's followees if they match desired characteristics."""
    visited_profiles.add(profile.username)
    save_visited_profiles(visited_profiles, visited_profiles_file)

    for followee in profile.get_followees():
        if followee.username not in visited_profiles:
            try:
                profile_pic_url = followee.profile_pic_url
                if image_classify(profile_pic_url, "female", "blond hair"):
                    followee_profile = instaloader.Profile.from_username(L.context, followee.username)
                    download_profile_and_last_post(followee_profile)
            except ProfileNotExistsException:
                print(f"Profile {followee.username} does not exist.")
            except PrivateProfileNotFollowedException:
                print(f"Profile {followee.username} is private.")

if __name__ == "__main__":
    base_dir = 'instagram_profiles'
    visited_profiles_file = 'visited_profiles.txt'
    
    # Ensure base directory exists
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Load previously visited profiles
    visited_profiles = load_visited_profiles(visited_profiles_file)
    
    # Login to Instagram
    login_instagram()

    # Target Instagram profile to explore
    target_profile = instaloader.Profile.from_username(L.context, "target_username")

    # Download images of followees
    download_followees_images(target_profile)
    sleep(10)  # Sleep to prevent hitting rate limits
