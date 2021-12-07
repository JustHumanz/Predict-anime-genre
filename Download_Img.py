import os,json,csv,requests

def checkRow(name):
    with open('poster.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            if name == row[0]:
                print("Already in csv",name)
                return False
    return True

def GetGenre():
    with open('poster.csv', 'r') as read_obj:
        csv_reader = csv.reader(read_obj)
        for row in csv_reader:
            return row

def append_dict_as_row(dict_of_elem, field_names):
    # Open file in append mode
    with open("poster.csv", 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        dict_writer = csv.DictWriter(write_obj, fieldnames=field_names)
        try:
            # Add dictionary as wor in the csv
            dict_writer.writerow(dict_of_elem)
        except:
            print("dict error")

path_to_json = 'json/'
Genre = GetGenre()
dictData = dict.fromkeys(Genre,0)

for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
  with open(path_to_json + file_name) as json_file:
    data = json.load(json_file)
    for i in data['data']['Page']['media']:
        newData = dictData.copy()
        AnimeName = i['title']['userPreferred']
        for z in ['/',' ','-']:
            AnimeName = AnimeName.replace(z, "_").strip()        
        
        for k in i['genres']:
            newData[k] = 1
        ImageName = "Poster/"+AnimeName+".jpg"

        print(ImageName)
        if checkRow(ImageName):
            r = requests.get(i['coverImage']['extraLarge'])
            if r.status_code != 200:
                print("Download Img Error",MangaName)
            ## Download IMG
            with open(ImageName, "wb") as f:
                f.write(r.content)

            newData["Image"] = ImageName
            append_dict_as_row(newData, Genre)
