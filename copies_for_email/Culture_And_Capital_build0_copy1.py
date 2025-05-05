

#Function Notes
#This function must collect 
#name + graph_top_5_trending_songs

#Step 1# connect to APIs: Source Billboard

class get_data_from_billboard:

    charts = billboard.ChartData('hot-100')
    for song in chart[:5]:
        print(song.title, song.artist)

    print("Done - gathered data from Billboard")



#Display Chart in Streamlit



#Step 2 # Display  


class SocialIndicatorTab:

    
    def __init__(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
        import billboard 
        pass

    def graph_top_5_trending_songs(self):
        
        class get_data_from_billboard:
            charts = billboard.ChartData('hot-100')
            for song in chart[:5]:
                print(song.title, song.artist)




            print("Done - gathered data from Billboard")

    
    def graph_music_entertainment_sector(self):

        



   