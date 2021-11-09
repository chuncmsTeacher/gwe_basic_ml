from gwe_fishdata import FishData
import matplotlib.pyplot as plt

fdata = FishData()

bream = fdata.getSpecies('Bream')
smelt = fdata.getSpecies('Smelt')
br, bd = fdata.getFeatures(bream, 'Weight', 'Length2')
sr, sd = fdata.getFeatures(smelt, 'Weight', 'Length2')
plt.scatter(bd[:,1], bd[:,0])
plt.scatter(sd[:,1], sd[:,0])
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Length & Weight')
plt.legend(['bream', 'smelt'])
'''

for f in fdata.species[1:3]:
    temp = fdata.getSpecies(f)
    tr, td = fdata.getFeatures(temp, 'Weight', 'Length2')
    plt.scatter(td[:,1], td[:,0])
plt.xlabel('Length')
plt.ylabel('Weight')
plt.title('Length & Weight')
plt.legend(fdata.species[1:3])
'''
plt.show()
