# Instagram Profile Image Downloader with Image Classification

This project downloads Instagram profile pictures and their last post from followers and followees based on specified criteria such as gender and hair color. It uses the Instaloader library for Instagram scraping and a custom image classification function for filtering.

## Features
- **Login with 2FA support**: Supports logging in with Instagram's Two-Factor Authentication.
- **Profile image downloading**: Downloads profile pictures of followers and followees.
- **Image classification**: Uses a custom classifier to filter profiles based on gender and hair color.
- **Rate limiting**: Includes pauses between downloads to avoid hitting Instagram's rate limits.

## Setup

1. **Install dependencies:**
```
pip install instaloader requests pillow
```

2. **Set Instagram credentials:**
```
export INSTAGRAM_USERNAME='your_username'
export INSTAGRAM_PASSWORD='your_password'
```

3. **Run the script:**
```
python main.py
```