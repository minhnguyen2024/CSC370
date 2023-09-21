import sys
import pandas as pd
def rank_actors_by_awards():
    # filename = sys.argv[1]
    filename = "oscar_data.csv"
    data = []
    listOfWinnerActors = []
    listOfWinnerActresses = []
    with open(filename) as file:
        for line in file:

            line = line.strip()
            tuple = line.split(",")
            # dict = { "year": tuple[0], "category": tuple[1], "winner": tuple[2], "entity": tuple[3]}
            if tuple[1] == "ACTOR" and tuple[2] == "True":
                dict = {"entity": tuple[3]}
                listOfWinnerActors.append(tuple[3])
            elif tuple[1] == "ACTRESS" and tuple[2] == "True":
                dict = {"entity": tuple[3]}
                listOfWinnerActresses.append(tuple[3])

    listOfWinnerActressesSortedByName = sorted(listOfWinnerActresses)
    for ele in listOfWinnerActressesSortedByName:
        print(ele)

    for i in range(0, len(listOfWinnerActressesSortedByName)-1):
        ct = 0

    # print(listOfWinnerActresses)


def scratch():
    data = pd.read_csv("oscar_data.csv")


rank_actors_by_awards()
# scratch()