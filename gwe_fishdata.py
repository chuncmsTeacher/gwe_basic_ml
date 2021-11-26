import numpy as np
class FishData:
    def __init__(self, file='Fish.csv'):
        with open(file) as f:
            lines = f.readlines()
        data = [line.strip().split(',') for line in lines]
        self.features = data.pop(0)
        #print(self.features)
        self.data = np.array(data)
        self.result = self.data[:,0]
        self.species = np.unique(self.result)
        #print(self.species)
        
    def getSpecies(self, *species):
        if len(species)==0:
            print(f'must have one Species in {self.species}')
            return []
        for s in species:
            if s not in self.species:
                print(f'{s} is not in {self.species}')
                return []
        rowindex = self.result == species[0]
        for s in species[1:]:
            rowindex |= self.result == s
            
        return self.data[rowindex]

    def getFeatures(self, fishlist, *features):
        if len(features) == 0:
            print(f'must have one Features in {self.features}')
            return [],[]
        colist = []
        for f in features:
            if f not in self.features:
                print(f'{f} is not in {self.features}')
                return [],[]
            colist.append(self.features.index(f))

        return fishlist[:,0], fishlist[:,colist].astype(float)
    
    def getFish(self, spec, fea):
        flist = self.getSpecies(*spec) if type(spec)==tuple else self.getSpecies(spec)
        return self.getFeatures(flist, *fea) if type(fea)==tuple else self.getFeatures(flist, fea)
    
if __name__=='__main__':
    mydata = FishData()
    bream = mydata.getSpecies('Bream')
    r, d = mydata.getFeatures(bream, 'Length1', 'Weight')
    print(r[:4])
    print(d[:4])
    br, bd = mydata.getFish(('Bream', 'Smelt'), ('Length1', 'Weight'))
    print(br[:4])
    print(bd[:4])
