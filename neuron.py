import numpy as np

def get_db():
    with open("./StudentPerformanceFactors.csv",'r') as r:
        a = r.readlines()
        a.pop(0)
        return a
    
a = get_db()

def clean_db():
    final=[]
    for f in a:
        aa=f.split(",")
        data=[]
        row=[]
        for b in range(2):
            # Pick 3 main datapoints
            data.append(int(aa[b]))
            
        row.append(data)
        row.append(int(aa[-1]))
            
        final.append(row)
        
        
    return final
 
    
    
cdb=clean_db()
print(np.array(cdb))

class Neuron:
    def __init__(self, num_inputs=3):
        self.weights = np.array()