import argparse
import csv
import datetime
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL.Image import Image
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFMCS
from rdkit.Geometry import Point3D
from tqdm import tqdm


INSTRUCTION_LIST = [
    "What precursors can be used to synthesize the molecule in <|image|>? Please output the precursor image.",
    "Which precursors can be used to synthesize the molecule shown in <|image|>? Please provide the precursor image.",
    "What are the precursors for synthesizing the molecule in <|image|>? Please provide the corresponding precursor image.",
    "Can you identify the precursors for the molecule in <|image|>? Please include the precursor structure image.",
    "What precursor molecules can be used to synthesize the compound in <|image|>? Please output the precursor image.",
    "What are the possible precursors for the synthesis of the molecule in <|image|>? Please provide the precursor structure image.",
    "From what precursors can the molecule in <|image|> be synthesized? Please include the precursor image.",
    "What are the starting precursors for the synthesis of the molecule in <|image|>? Please output the precursor image.",
    "Which precursor molecules lead to the synthesis of the compound in <|image|>? Please provide the precursor structure image.",
    "What are the precursors involved in the synthesis of the molecule shown in <|image|>? Please include the precursor image."
]


def get_center_x(mol):
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    return sum(xs) / len(xs) if xs else 0


def get_x_range(mol):
    conf = mol.GetConformer()
    xs = [conf.GetAtomPosition(i).x for i in range(mol.GetNumAtoms())]
    return min(xs), max(xs)


def shift_mol_horizontally(mol, offset_x):
    conf = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        conf.SetAtomPosition(i, Point3D(pos.x + offset_x, pos.y, pos.z))


def align_sort_and_separate(product, reactant_list, padding=2.0):
    # 1. initial alignment
    temp_mols = []
    for mol in reactant_list:
        mcs = rdFMCS.FindMCS([product, mol], timeout=60)
        if mcs.numAtoms > 0:
            mcs_template = Chem.MolFromSmarts(mcs.smartsString)
            try:
                AllChem.GenerateDepictionMatching2DStructure(mol, product, refPatt=mcs_template)
            except:
                AllChem.Compute2DCoords(mol)
        else:
            AllChem.Compute2DCoords(mol)
        temp_mols.append(mol)

    # 2. sort by center x
    temp_mols.sort(key=lambda m: get_center_x(m))

    # 3. separate to avoid overlap
    aligned_mols = []
    last_max_x = None

    for i, mol in enumerate(temp_mols):
        if i == 0:
            _, max_x = get_x_range(mol)
            last_max_x = max_x
        else:
            curr_min_x, curr_max_x = get_x_range(mol)
            # calculate shift distance
            shift_dist = (last_max_x + padding) - curr_min_x
            shift_mol_horizontally(mol, shift_dist)

            # update last_max_x
            _, new_max_x = get_x_range(mol)
            last_max_x = new_max_x

        aligned_mols.append(mol)

    return aligned_mols


def draw_aligned_molecules(reactant_smiles_raw: str, product_smiles: str) -> tuple[Image, Image] | tuple[None, None]:
    if "." in product_smiles:
        return None, None  # Skip if product is not a single molecule

    product = Chem.MolFromSmiles(product_smiles)
    AllChem.Compute2DCoords(product)

    reactant_mols = [Chem.MolFromSmiles(s) for s in reactant_smiles_raw.split('.')]

    aligned_reactants = align_sort_and_separate(product, reactant_mols, padding=3.0)

    combined_reactant = aligned_reactants[0]
    for i in range(1, len(aligned_reactants)):
        combined_reactant = Chem.CombineMols(combined_reactant, aligned_reactants[i])

    img = Draw.MolsToGridImage(
        [combined_reactant, product],
        molsPerRow=2,
        subImgSize=(args.size, args.size),
    )

    reactant_img = img.crop((0, 0, args.size, args.size))
    product_img = img.crop((args.size, 0, args.size * 2, args.size))
    return [reactant_img, product_img]


def gen_one(item: dict, out_filename: str) -> dict | None:
    split = "unknown"
    if "train" in out_filename:
        split = "train"
    elif "test" in out_filename:
        split = "test"
    elif "val" in out_filename:
        split = "val"

    product_img_path = []
    reactant_img_path = []

    reactant_smiles_raw = item["output"]
    product_smiles = item["input"]
    try:
        reactant_img, product_img = draw_aligned_molecules(reactant_smiles_raw, product_smiles)
        if reactant_img is None or product_img is None:
            return None
    except:
        return None

    product_img_path = os.path.join(args.data_output, "images", split, f"{item['idx']}_product.png")
    reactant_img_path = os.path.join(args.data_output, "images", split, f"{item['idx']}_reactant.png")

    product_img.save(product_img_path)
    reactant_img.save(reactant_img_path)

    return {
        "conversations": [
            {
                "from": "human",
                "value": item["instruction"]
            },
            {
                "from": "gpt",
                "value": "<|image|>"
            }
        ],
        "image": [
            product_img_path,
            reactant_img_path
        ],
        "metadata": {
            "idx": item["idx"]
        }
    }


def gen_data(data_split: list, out_filename: str):
    results = []

    if not os.path.exists(os.path.join(args.data_output, "images", "train")):
        os.makedirs(os.path.join(args.data_output, "images", "train"))
    if not os.path.exists(os.path.join(args.data_output, "images", "test")):
        os.makedirs(os.path.join(args.data_output, "images", "test"))
    if not os.path.exists(os.path.join(args.data_output, "images", "val")):
        os.makedirs(os.path.join(args.data_output, "images", "val"))

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(gen_one, item, out_filename): item for item in data_split}
        for future in tqdm(as_completed(futures), total=len(futures), dynamic_ncols=True):
            result = future.result()
            if result is None:
                continue
            results.append(result)

    with open(out_filename, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def clean_atom_mapping(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def prepare_raw_data(file_path: str) -> list:
    with open(file_path, "r") as f:
        reader = csv.DictReader(f)
        data_list = [row for row in reader]

    results = []
    for idx, item in enumerate(tqdm(data_list)):
        origin = item["id"]
        reactants, reagents, production = item["reactants>reagents>production"].split(">")
        reactants = clean_atom_mapping(reactants)
        production = clean_atom_mapping(production)
        instruction = random.choice(INSTRUCTION_LIST)

        results.append({
            "idx": idx,
            "origin": origin,
            "instruction": instruction,
            "input": production,
            "output": reactants
        })

    return results


def main():
    random.seed(args.seed)

    train_data = prepare_raw_data(os.path.join(args.data_input, "uspto50k_train.csv"))
    test_data = prepare_raw_data(os.path.join(args.data_input, "uspto50k_test.csv"))
    val_data = prepare_raw_data(os.path.join(args.data_input, "uspto50k_val.csv"))

    gen_data(train_data, os.path.join(args.data_output, "uspto50k_train.json"))
    gen_data(test_data, os.path.join(args.data_output, "uspto50k_test.json"))
    gen_data(val_data, os.path.join(args.data_output, "uspto50k_val.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--data-input", type=str, default="./dataset/raw")
    parser.add_argument("--data-output", type=str, default=f"./dataset/uspto50k_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    main()
