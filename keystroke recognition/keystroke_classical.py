import csv  
import os



def processing_csv_files(path_csv_file):
    #open CSV file
    data_structure = {}
    name = ('KD', 'DDKL', 'UUKL')
    with open(path_csv_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader,None)

        for row in enumerate(reader):
            data_structure[row[0]] = ({name[0]:()},{name[1]:()},{name[2]:()})
           
        print(data_structure)       


if __name__ == "__main__":
    path_csv_file = "./DSL-StrongPasswordData.csv"
    processing_csv_files(path_csv_file)