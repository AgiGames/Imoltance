# Imoltance
A python library that computes features based on graphical strucures of molecules and compounds.

## Install
```!pip install imoltance```

## Usage
```
from imoltance import Imoltance
imlt = Imoltance('Bradley_Melting_Point_Dataset.csv', 'SMILES', ['Tm'], 231)
imlt.run()
```
