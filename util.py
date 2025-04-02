import unicodedata
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import Draw
import io

greek_letter_mapping = {
    "α": "alpha",
    "β": "beta",
    "γ": "gamma",
    "δ": "delta",
    "ε": "epsilon",
    "ζ": "zeta",
    "η": "eta",
    "θ": "theta",
    "ι": "iota",
    "κ": "kappa",
    "λ": "lambda",
    "μ": "mu",
    "ν": "nu",
    "ξ": "xi",
    "ο": "omicron",
    "π": "pi",
    "ρ": "rho",
    "σ": "sigma",
    "τ": "tau",
    "υ": "upsilon",
    "φ": "phi",
    "χ": "chi",
    "ψ": "psi",
    "ω": "omega"
}

def replace_greek_letters(molecule: str) -> str:
    for greek, eng in greek_letter_mapping.items():
        molecule = molecule.replace(greek, eng)
    return molecule

async def get_smiles(molecule: str) -> dict:
    molecule = unicodedata.normalize('NFC', molecule).replace('\u2013', '-').replace('\u2014', '-').replace('\u2010', '-')
    try:
        compounds = pcp.get_compounds(molecule, 'name')
        if compounds:
            smiles = compounds[0].canonical_smiles
            return {"molecule": molecule, "smiles": smiles}
        else:
            return {"molecule": molecule, "smiles": "Not found"}
    except Exception as e:
        return {"molecule": molecule, "smiles": f"Error: {str(e)}"}
    
def smiles_to_image(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes.seek(0)
        return img_bytes.getvalue()
    return None