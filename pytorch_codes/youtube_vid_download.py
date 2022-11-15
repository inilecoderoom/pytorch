# https://www.youtube.com/watch?v=yLbAPK0LCfk  19:51 comedy
# https://www.youtube.com/watch?v=X_ZpdRlACC0  18:47 romantic
# https://www.youtube.com/watch?v=4gn2KiH0XKI  5:36  video
# https://www.youtube.com/watch?v=b4OH3vBANa4  3:11 sports
# https://www.youtube.com/watch?v=MUMCZZl9QCY  26:41 cartoon

from pytube import Playlist, YouTube

playlist_url = 'https://youtube.com/playlist?list=special_playlist_id'
p = Playlist(playlist_url)
for url in p.video_urls:
    try:
        yt = YouTube(url)
    except VideoUnavailable:
        print(f'Video {url} is unavaialable, skipping.')
    else:
        print(f'Downloading video: {url}')
        yt.streams.first().download()