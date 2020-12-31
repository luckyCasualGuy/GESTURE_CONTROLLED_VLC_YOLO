from vlc import Instance, MediaListPlayer
from pathlib import Path

class VLCController():

    def __init__(self, number_conf: dict):
        self.conf = number_conf
        self.media_player = MediaListPlayer()

    def setPlaylist(self, PATH_TO_PLAYLIST: str):
        playlist = Path(PATH_TO_PLAYLIST)
        if not playlist.exists(): raise ValueError('Playlist does not exist')

        player = Instance()
        media_list = player.media_list_new()

        for path in playlist.glob('*.mp3'):
            media = player.media_new(path)
            media_list.add_media(media)
            
        self.media_player.set_media_list(media_list)


    def configMapper(self, signal):
        if signal:
            if signal == self.conf['play']: self.media_player.play()
            if signal == self.conf['pause']: self.media_player.pause()
            if signal == self.conf['stop']: self.media_player.stop()
            if signal == self.conf['next']: self.media_player.next()
            if signal == self.conf['previous']: self.media_player.previous()