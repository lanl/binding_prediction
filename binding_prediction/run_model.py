# Imports
from rdkit import Chem
from rdchiral.initialization import rdchiralReaction, rdchiralReactants
from rdchiral.main import rdchiralRun
from mordred import Calculator, descriptors
import pandas as pd
import pickle
from itertools import chain
import numpy as np
from multiprocessing import freeze_support
import sys


def main(smi_file):
    # Set file paths
    model = "model.pkl"  # pickled tuple containing trained model, X scaler and y scaler

    with open(smi_file, "r") as f:
        smiles = [s.removesuffix("\n") for s in f.readlines()]

    # Define templates to generate children
    reaction_templates = [
        "[!O;*:2]-[Nv3&H2:1]>>[!O;*:2]-[Nv3&H1:1]-C(=O)-O",
        "[Nv3&H1:1](-[!O;*:2])-[!O;*:3]>>O-C(=O)-[Nv3&H0:1](-[!O;*:2])-[*:3]",
        "[nv3&H1:1]>>O-C(=O)-[nv3&H0:1]",
    ]

    # Load model
    with open(model, "rb") as f:
        model, X_scaler, y_scaler = pickle.load(f)

    parents = [Chem.MolFromSmiles(smi) for smi in smiles]

    features = X_scaler.feature_names_in_

    mols = [Chem.MolFromSmiles(smi) for smi in smiles]

    # Form all of the children and a mapping to corresponding parent
    children = [None] * len(mols)
    pidx_map = []
    for i, mol in enumerate(mols):
        childs, _ = chiral_reaction_primary_secondary(
            Chem.MolToSmiles(mol), reaction_templates
        )
        children[i] = childs
        for child in childs:
            pidx_map.append(i)
    children = list(chain.from_iterable(children))

    calc_descrip = Calculator(descriptors, ignore_3D=True)
    c_descs = calc_descrip.pandas(children)
    p_descs = calc_descrip.pandas(mols)

    # Only keep molecules where all descriptors were successfully computed for parents and children
    p_descs_p = post_process_mordred(p_descs, features)
    c_descs_p = post_process_mordred(c_descs, features)
    overlap = c_descs_p.columns.intersection(p_descs_p.columns)
    c_desc_over = c_descs_p.loc[:, overlap]
    p_desc_over = p_descs_p.loc[:, overlap]

    # Use mapping to compute delta descriptors and create dataframe, again skipping molecules with errors
    mydict_list = []
    for i, row in c_desc_over.iterrows():
        try:
            delta = row[features] - p_desc_over.loc[[pidx_map[i]]][features]
            mydict_list.append(delta.to_dict(orient="records")[0])
        except:
            pass
    delta = pd.DataFrame(mydict_list)
    X = X_scaler.transform(delta)

    # Run model and get predictions
    binds = y_scaler.inverse_transform(model.predict(X).reshape(-1, 1))

    # Print out the values
    for i, pidx in enumerate(pidx_map):
        mystr = f"{Chem.MolToSmiles(mols[pidx])},{Chem.MolToSmiles(children[i])},{binds[i][0]:.6f}\n"
        print(mystr)


def chiral_reaction_primary_secondary(smiles: str, reaction_templates: list):
    """
    Add CO2 to SMILES using reaction smarts.

    Keyword Arguments:
    smiles: str               -- smiles string
    reaction_templates: list  -- list of SMARTS reactions

    Returns:
    mol_products: list    -- list of 2D rdkit mol objects
    smiles_products: list -- list of smiles with CO2 added
    """
    rxns = [rdchiralReaction(template) for template in reaction_templates]
    reactant = rdchiralReactants(smiles)

    products = [rdchiralRun(rxn, reactant) for rxn in rxns]
    smiles_products = list(set(chain.from_iterable(products)))
    mol_products = [Chem.MolFromSmiles(smiles) for smiles in smiles_products]

    return mol_products, smiles_products


def post_process_mordred(df, features):
    """Post-process mordred."""
    df = df.apply(lambda x: pd.to_numeric(x, errors="coerce"))
    df["Lipinski"] = df["Lipinski"].apply(lambda x: int(x))
    df["GhoseFilter"] = df["GhoseFilter"].apply(lambda x: int(x))
    df = df[features]
    return df


if __name__ == "__main__":
    freeze_support()
    main(sys.argv[1])
