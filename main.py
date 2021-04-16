#author: Nikhil_Chauhan
from matplotlib.patches import Wedge
import matplotlib.pyplot as plt
from numpy.lib.function_base import _gradient_dispatcher
import pandas as pd
import matplotlib.dates as mdates
import datetime
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import csv
import pandas as pd
import matplotlib.patches as mpatches
import calendar


while True:
    print()
    print("===========GENDER AND AGE DETECTION WITH GRAPH ANALYSIS===========")
    print("1->Analyze Image")
    print("2->Plot Gender Data")
    print("3->Exit")
    print()

    choice = int(input("Enter your choice:"))
    if choice == 2:
        now = datetime.now()
        date = now.strftime("%d/%m/%Y")
        month = calendar.month_name[int(now.strftime("%m"))]
        time = now.strftime("%I:%M %p")
        my_labels = ['Woman', 'Man']
        my_colors = ['#ff9999', '#66b3ff']
        wp = {'linewidth': 1, 'edgecolor': "green"}
        my_explode = (0, 0.05)
        #csv_file = pd.read_csv('./csv_data/%s.csv'% month)
        my_file = Path('./csv_data/%s.csv' % month)
        if my_file.is_file():
            df_tips = pd.read_csv('./csv_data/%s.csv' % month)
            df = df_tips['Gender'].value_counts()
            plt.pie(df, labels=my_labels, autopct='%1.2f%%',
                    colors=my_colors, radius=0.65, startangle=90, wedgeprops=wp,)
            centre_circle = plt.Circle(
                (0, 0), 0.5, color='black', fc='white', linewidth=0)
            fig = plt.gcf()
            fig.gca().add_artist(centre_circle)
            plt.axis('equal')
            plt.tight_layout()
            pink_patch = mpatches.Patch(
                color='#ff9999', label='Woman: ' + str(df["Woman "]))
            blue_patch = mpatches.Patch(
                color='#66b3ff', label='Man: ' + str(df["Man"]))

            plt.legend(handles=[pink_patch, blue_patch],
                       title="Count", loc="best")
            plt.savefig('./graphs/%s.png' % month)
            # plt.show()
            print()
            print("Graph Saved in Graph dir")
            print()

            #   break
        else:
            print()
            print(
                "CSV Data Does not exist -> Please Run Analysis First and then try to plot.")
            print()
        #df_tips = pd.read_csv('./csv_data/%s.csv'% month)

    elif choice == 1:
        import test
        print()
        print("Analysis Done")
        print()
        # break
    elif choice == 3:
        print("Exiting")
        break
    else:
        print()
        print("Wrong Choice Choose Again")
        print()
