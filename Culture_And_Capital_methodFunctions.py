
import billboard

#get information from billboard

#feature 2: get rank, lastpos, peakpos

def get_data_from_billboard_01():
    print("running get_fata_from_billboard()")

    chart = billboard.ChartData('hot-100')
    songs = []
    for song in chart[:5]:
        songs.append({"Title": song.title, "Artist": song.artist})
    return songs

top_songs = get_data_from_billboard_01()


def get_data_from_billboard_02():
    chart = billboard.ChartData('hot-100')
    songs = []
    for song in chart[:5]:
        songs.append({
            "Rank": song.rank,
            "Title": song.title,
            "Artist": song.artist,
            "Last Week": song.lastPos,
            "Peak Position": song.peakPos
        })

    return songs
