import pickle


# reload a file to a variable
with open('lat.pkl', 'rb') as file:
    dict_lat =pickle.load(file)

print(dict_lat)

for i in range(len(dict_lat)): # for every pic
	print ("i", i)
	for x in range(len(dict_lat[str(i)][0])): # 20
		dict_lat[str(i)][0] = dict_lat[str(i)][0]*0.7
		#print dict_lat[str(i)][0], len(dict_lat[str(i)][0])

pickle.dump(dict_lat, open("lat_new.pkl", "wb"))



