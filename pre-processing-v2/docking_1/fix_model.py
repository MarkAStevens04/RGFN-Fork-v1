from openmm.app import PDBFile
from pdbfixer import PDBFixer

fixer = PDBFixer(filename="../models/5HXB.cif")
fixer.findMissingResidues()  # decide which to keep — clear distal loops if you don't want them rebuilt
fixer.findMissingAtoms()
fixer.addMissingAtoms()  # rebuilds missing side-chain/backbone atoms

## Want to only keep chains X and Z
fixer.removeHeterogens(keepWater=False)  # strips waters, buffers, ligand
PDBFile.writeFile(fixer.topology, fixer.positions, open("../models/5HXB_clean.pdb", "w"))
