from flask import Flask, jsonify, request, render_template
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions, AllChem, rdDepictor
from PIL import Image, ImageDraw
import cairo
import numpy as np
from collections import namedtuple, defaultdict
import os
import json
import hashlib
import uuid
import random
from scipy.signal import correlate2d
app = Flask(__name__)
# File paths
ALL_TREES_FILE = "mixed_ground_truth_molecules_trees.json"
MASTER_FILE = "master_paths.json"
USER_SESSIONS_DIR = "user_sessions"
FEEDBACK_SUMMARY_FILE = "feedback_summary.json"
IMAGE_DIR = "static/molecule_images"
# Ensure necessary directories exist
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(USER_SESSIONS_DIR, exist_ok=True)
@app.route("/")
def intro():
    return render_template("intro.html")
@app.route("/trees")
def trees():
    return render_template("index.html")  # Your existing template for displaying trees
def generate_hash(input_string):
    """Generate a hash from a SMILES or SMARTS string."""
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()

def merge_highlighted_images(image_path1, image_path2, merged_filename):
    """Merges two molecule images, correcting for pixel shifts using cross-correlation."""
    img1 = Image.open(image_path1.lstrip('/'))
    img2 = Image.open(image_path2.lstrip('/'))

    # Convert images to grayscale for correlation
    img1_gray = img1.convert('L')
    img2_gray = img2.convert('L')

    # Convert images to NumPy arrays
    arr1 = np.array(img1_gray)
    arr2 = np.array(img2_gray)

    # Find the shift using cross-correlation
    correlation = correlate2d(arr1, arr2, mode='valid')
    y, x = np.unravel_index(np.argmax(correlation), correlation.shape)

    # Calculate shift
    shift_x = x
    shift_y = y


    #Adjust shift to center
    shift_x = shift_x - arr1.shape[1]//2
    shift_y = shift_y - arr1.shape[0]//2


    #Shift image 2
    arr2 = np.pad(arr2, ((max(0,-shift_y),max(0,shift_y)),(max(0,-shift_x),max(0,shift_x))), mode='constant',constant_values=255)
    if shift_x < 0:
        arr2 = arr2[:, abs(shift_x):]
    elif shift_x > 0:
        arr2 = arr2[:, :arr2.shape[1]-shift_x]

    if shift_y < 0:
        arr2 = arr2[abs(shift_y):, :]
    elif shift_y > 0:
        arr2 = arr2[:arr2.shape[0]-shift_y, :]


    # Convert images back to RGB (if they were RGB originally)
    if img1.mode == 'RGB':
        arr1 = np.array(img1)
        arr2 = np.array(img2)
    # Merge using masks for non-white pixels (same logic as before)
    mask1 = np.any(arr1 != 255, axis=-1)
    mask2 = np.any(arr2 != 255, axis=-1)
    merged_arr = np.where(mask1[:, :, np.newaxis], arr1,
                          np.where(mask2[:, :, np.newaxis], arr2, 255))
    merged_img = Image.fromarray(merged_arr.astype(np.uint8))
    merged_filepath = os.path.join(IMAGE_DIR, merged_filename)
    merged_img.save(merged_filepath)
    return f"/{merged_filepath.replace(os.sep, '/')}"

def map_reacting_atoms_to_products(rxn,reactingAtoms):
    ''' figures out which atoms in the products each mapped atom in the reactants maps to '''
    res = []
    for ridx,reacting in enumerate(reactingAtoms):
        reactant = rxn.GetReactantTemplate(ridx)
        for raidx in reacting:
            mapnum = reactant.GetAtomWithIdx(raidx).GetAtomMapNum()
            foundit=False
            for pidx,product in enumerate(rxn.GetProducts()):
                for paidx,patom in enumerate(product.GetAtoms()):
                    if patom.GetAtomMapNum()==mapnum:
                        res.append(AtomInfo(mapnum,ridx,raidx,pidx,paidx))
                        foundit = True
                        break
                    if foundit:
                        break
    return res
def get_mapped_neighbors(atom):
    ''' test all mapped neighbors of a mapped atom'''
    res = {}
    amap = atom.GetAtomMapNum()
    if not amap:
        return res
    for nbr in atom.GetNeighbors():
        nmap = nbr.GetAtomMapNum()
        if nmap:
            if amap>nmap:
                res[(nmap,amap)] = (atom.GetIdx(),nbr.GetIdx())
            else:
                res[(amap,nmap)] = (nbr.GetIdx(),atom.GetIdx())
    return res

AtomInfo = namedtuple('AtomInfo',('mapnum','reactant','reactantAtom','product','productAtom'))
BondInfo_p = namedtuple('BondInfo',('product','productAtoms','productBond','status'))
def find_modifications_in_products(rxn):

    ''' returns a 2-tuple with the modified atoms and bonds from the reaction '''
    reactingAtoms = rxn.GetReactingAtoms()
    amap = map_reacting_atoms_to_products(rxn,reactingAtoms)
    res = []
    seen = set()
    # this is all driven from the list of reacting atoms:
    for _,ridx,raidx,pidx,paidx in amap:
        reactant = rxn.GetReactantTemplate(ridx)
        ratom = reactant.GetAtomWithIdx(raidx)
        product = rxn.GetProductTemplate(pidx)
        patom = product.GetAtomWithIdx(paidx)

        rnbrs = get_mapped_neighbors(ratom)
        pnbrs = get_mapped_neighbors(patom)
        for tpl in pnbrs:
            pbond = product.GetBondBetweenAtoms(*pnbrs[tpl])
            if (pidx,pbond.GetIdx()) in seen:
                continue
            seen.add((pidx,pbond.GetIdx()))
            if not tpl in rnbrs:
                # new bond in product
                res.append(BondInfo_p(pidx,pnbrs[tpl],pbond.GetIdx(),'New'))
            else:
                # present in both reactants and products, check to see if it changed
                rbond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
                if rbond.GetBondType()!=pbond.GetBondType():
                    res.append(BondInfo_p(pidx,pnbrs[tpl],pbond.GetIdx(),'Changed'))
    return amap,res

def draw_product_with_modified_bonds(rxn,atms,bnds, filepath, size, in_stock=False):
    productIdx=None
    showAtomMaps=False
    if productIdx is None:
        pcnts = [x.GetNumAtoms() for x in rxn.GetProducts()]
        largestProduct = list(sorted(zip(pcnts,range(len(pcnts))),reverse=True))[0][1]
        productIdx = largestProduct
    d2d = Draw.MolDraw2DCairo(size,size)
    pmol = Chem.Mol(rxn.GetProductTemplate(productIdx))
    Chem.SanitizeMol(pmol)
    if not showAtomMaps:
        for atom in pmol.GetAtoms():
            atom.SetAtomMapNum(0)
    bonds_to_highlight=[]
    highlight_bond_colors={}
    atoms_seen = set()
    for binfo in bnds:
        if binfo.product==productIdx and binfo.status=='New':
            bonds_to_highlight.append(binfo.productBond)
            atoms_seen.update(binfo.productAtoms)
            highlight_bond_colors[binfo.productBond] = (1,.4,.4)
        if binfo.product==productIdx and binfo.status=='Changed':
            bonds_to_highlight.append(binfo.productBond)
            atoms_seen.update(binfo.productAtoms)
            highlight_bond_colors[binfo.productBond] = (.4,.4,1)
    atoms_to_highlight=set()
    for ainfo in atms:
        if ainfo.product != productIdx or ainfo.productAtom in atoms_seen:
            continue
        atoms_to_highlight.add(ainfo.productAtom)
    

#    d2d.drawOptions().useBWAtomPalette()
    d2d.drawOptions().continuousHighlight=True
    d2d.drawOptions().padding = .1
    d2d.drawOptions().baseFontSize = 0.8
    d2d.drawOptions().highlightBondWidthMultiplier = 10
    d2d.drawOptions().highlightRadius = 0.05
    d2d.drawOptions().fixedScale = .5
#    d2d.drawOptions().fixedBondLength = 40
    d2d.drawOptions().centreMoleculesBeforeDrawing = True
    d2d.drawOptions().setHighlightColour((0,.9,.9,.8))
    d2d.drawOptions().fillHighlights=False
    atoms_to_highlight.update(atoms_seen)

    rdDepictor.SetPreferCoordGen(True)
    rdDepictor.Compute2DCoords(pmol)
    rdDepictor.StraightenDepiction(pmol)
    d2d.DrawMolecule(pmol,highlightAtoms=atoms_to_highlight,highlightBonds=bonds_to_highlight,
                     highlightBondColors=highlight_bond_colors)
    d2d.FinishDrawing()
    d2d.WriteDrawingText(filepath)
    frame_width = 2
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    # Draw the colored frame
    frame_color = (0, 1, 0) if in_stock else (1, 0.65, 0)  # Green if in stock, orange if not
    ctx.set_source_rgb(*frame_color)
    ctx.rectangle(0, 0, size, size)
    ctx.set_line_width(frame_width * 2)
    ctx.stroke()
    
    # Draw the molecule image inside the frame
    mol_surface = cairo.ImageSurface.create_from_png(filepath)
    ctx.set_source_surface(mol_surface, frame_width, frame_width)
    ctx.paint()

    surface.write_to_png(filepath)
    return f"/{filepath.replace(os.sep, '/')}"

BondInfo_r = namedtuple('BondInfo', ('reactant', 'reactantAtoms', 'reactantBond', 'status'))
def find_modifications_in_reactants(rxn):

    reactingAtoms = rxn.GetReactingAtoms()
    amap = map_reacting_atoms_to_products(rxn, reactingAtoms)
    res = []
    seen = set()

    for _, ridx, raidx, pidx, paidx in amap:
        reactant = rxn.GetReactantTemplate(ridx)
        ratom = reactant.GetAtomWithIdx(raidx)
        product = rxn.GetProductTemplate(pidx)
        patom = product.GetAtomWithIdx(paidx)

        rnbrs = get_mapped_neighbors(ratom)
        pnbrs = get_mapped_neighbors(patom)

        for tpl in rnbrs:
            r_bond = reactant.GetBondBetweenAtoms(*rnbrs[tpl])
            if (ridx, r_bond.GetIdx()) in seen:
                continue
            seen.add((ridx, r_bond.GetIdx()))

            if tpl not in pnbrs:  # Bond broken in reactant
                res.append(BondInfo_r(ridx, rnbrs[tpl], r_bond.GetIdx(), 'Broken'))
            else:
                # Bond exists in both; check for changes (this part is redundant if only broken bonds are needed).
                p_bond = product.GetBondBetweenAtoms(*pnbrs[tpl])
                if r_bond.GetBondType() != p_bond.GetBondType():
                    res.append(BondInfo_r(ridx, rnbrs[tpl], r_bond.GetIdx(), 'Changed')) #Possibly redundant

    return amap, res


def draw_reactants_with_modified_bonds(smiles, rxn, atms, bnds, filepath, size, in_stock=False ):
    """Generates images for all reactants, highlighting modified bonds."""
    showAtomMaps=False
    num_reactants = rxn.GetNumReactantTemplates()
    reactant_images = []

    for reactantIdx in range(num_reactants):
        d2d = Draw.MolDraw2DCairo(size,size)  #Adjust size as needed
        rmol = Chem.Mol(rxn.GetReactantTemplate(reactantIdx))
        Chem.SanitizeMol(rmol)
        rmol1 = rmol   
        for atom in rmol1.GetAtoms():
            atom.SetAtomMapNum(0)
        smi_mol = Chem.MolFromSmiles(smiles)
        if Chem.MolToSmiles(smi_mol) == Chem.MolToSmiles(rmol1):
            if not showAtomMaps:
                for atom in rmol.GetAtoms():
                    atom.SetAtomMapNum(0)

            bonds_to_highlight = []
            highlight_bond_colors = {}
            atoms_seen = set()

            for binfo in bnds:
                if binfo.reactant == reactantIdx and binfo.status == 'Broken':
                    bonds_to_highlight.append(binfo.reactantBond)
                    atoms_seen.update(binfo.reactantAtoms)
                    highlight_bond_colors[binfo.reactantBond] = (0.9, 0.9, 0)  # Blue for broken bonds
                #elif binfo.reactant == reactantIdx and binfo.status == 'Changed': #Uncomment if needed
                #    bonds_to_highlight.append(binfo.reactantBond)
                #    atoms_seen.update(binfo.reactantAtoms)
                #    highlight_bond_colors[binfo.reactantBond] = (0, 0, 1)  # Blue for changed bonds


            atoms_to_highlight = set()
            for ainfo in atms:
                if ainfo.reactant != reactantIdx or ainfo.reactantAtom in atoms_seen:
                    continue
                atoms_to_highlight.add(ainfo.reactantAtom)


    #        d2d.drawOptions().useBWAtomPalette()
            d2d.drawOptions().baseFontSize = 0.8
            d2d.drawOptions().continuousHighlight = True
            d2d.drawOptions().highlightBondWidthMultiplier = 10
            d2d.drawOptions().highlightRadius = 0.05
            d2d.drawOptions().fixedScale = .5
#            d2d.drawOptions().fixedBondLength = 40
            d2d.drawOptions().padding = .1
            
            d2d.drawOptions().centreMoleculesBeforeDrawing = True
            d2d.drawOptions().setHighlightColour((0.9, 0.9, 0))
            d2d.drawOptions().fillHighlights = False
            atoms_to_highlight.update(atoms_seen)

            rdDepictor.SetPreferCoordGen(True)
            rdDepictor.Compute2DCoords(rmol)
            rdDepictor.StraightenDepiction(rmol)
            d2d.DrawMolecule(rmol, highlightAtoms=atoms_to_highlight, highlightBonds=bonds_to_highlight,
                            highlightBondColors=highlight_bond_colors)
            d2d.FinishDrawing()
            d2d.WriteDrawingText(filepath)
            frame_width = 2
            surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
            ctx = cairo.Context(surface)
            # Draw the colored frame
            frame_color = (0, 1, 0) if in_stock else (1, 0.65, 0)  # Green if in stock, orange if not
            ctx.set_source_rgb(*frame_color)
            ctx.rectangle(0, 0, size, size)
            ctx.set_line_width(frame_width * 2)
            ctx.stroke()
            
            # Draw the molecule image inside the frame
            mol_surface = cairo.ImageSurface.create_from_png(filepath)
            ctx.set_source_surface(mol_surface, frame_width, frame_width)
            ctx.paint()

            surface.write_to_png(filepath)
        else:
            continue
    return f"/{filepath.replace(os.sep, '/')}"

def descendant_reaction_wrap(smiles, descendant_reaction, filepath, size, in_stock):
    r,p = descendant_reaction.split(">>")   
    descendant_reaction = ">>".join([p,r])    
    rxn = AllChem.ReactionFromSmarts(descendant_reaction, useSmiles=True)                                       
    rxn.Initialize()
    atms,bnds = find_modifications_in_products(rxn)
    draw_product_with_modified_bonds(rxn, atms, bnds, filepath, size, in_stock)

def preceeding_reaction_wrap(smiles, preceding_reaction, filepath, size, in_stock):
    r,p = preceding_reaction.split(">>")   
    preceding_reaction = ">>".join([p,r])    
    rxn = AllChem.ReactionFromSmarts(preceding_reaction, useSmiles=True)                                       
    rxn.Initialize()
    atms,bnds = find_modifications_in_reactants(rxn)
    draw_reactants_with_modified_bonds(smiles, rxn, atms, bnds, filepath, size, in_stock)


def generate_highlighted_molecule_images(smiles, filename_base, preceding_reaction=None, descendant_reaction=None,  in_stock=False):
    images = {"neutral": {}, "highlighted": {}, "merged": {}}
    filenames = []
    sizes=[100, 200, 300, 400]
    for size in sizes:
    # Generate neutral image
        neutral_filename = f"{filename_base}_neutral_{size}.png"
        neutral_path = generate_molecule_image(smiles, neutral_filename, size, in_stock)
        if neutral_path:
            images["neutral"][size] = neutral_path
            filenames.append(neutral_filename)
        highlighted_images = []
        # Generate image for descendant reaction (if exists)
        if descendant_reaction:
            descendant_filename = f"{filename_base}_descendant_{size}.png"
            descendant_filepath = os.path.join(IMAGE_DIR, descendant_filename)
            descendant_reaction_wrap(smiles, descendant_reaction, descendant_filepath, size, in_stock)
            descendant_path = f"/{descendant_filepath.replace(os.sep, '/')}"
            images["highlighted"][size] = descendant_path
            #images.append(descendant_path)
            filenames.append(descendant_filename)
            highlighted_images.append(descendant_path)
        # Generate image for preceding reaction (if exists)
        if preceding_reaction:
            preceding_filename = f"{filename_base}_preceding_{size}.png"
            preceding_filepath = os.path.join(IMAGE_DIR, preceding_filename)
            preceeding_reaction_wrap(smiles, preceding_reaction, preceding_filepath, size, in_stock)
            preceding_path = f"/{preceding_filepath.replace(os.sep, '/')}"
            #images.append(preceding_path)
            images["highlighted"][size] = preceding_path
            filenames.append(preceding_filename)
            highlighted_images.append(preceding_path)
        # Generate merged image if both highlighted images exist
        if preceding_reaction and descendant_reaction:
            merged_filename = f"{filename_base}_merged_{size}.png"
            merged_filepath = os.path.join(IMAGE_DIR, merged_filename)
            merged_path = merge_highlighted_images(highlighted_images[0], highlighted_images[1], merged_filename)
            #images.append(merged_path)
            filenames.append(merged_filename)
            images["merged"][size] = f"/{merged_filepath.replace(os.sep, '/')}"
    #print(images)
    return images#, filenames

def generate_reaction_image(rxn, filename):
    rxn = rdChemReactions.ReactionFromSmarts(rxn, useSmiles=True)
    if rxn:
        filepath = os.path.join(IMAGE_DIR, filename)
        d2d = Draw.MolDraw2DCairo(800,300)
        dopts = d2d.drawOptions()
        dopts.baseFontSize = 0.8
        dopts.fixedScale = .5
#        dopts.fixedBondLength = 40
        dopts.centreMoleculesBeforeDrawing = True
        d2d.DrawReaction(rxn)
        d2d.FinishDrawing()
        d2d.WriteDrawingText(filepath)
        return f"/{filepath.replace(os.sep, '/')}"
    return None

def generate_molecule_image(smiles, filename, size, in_stock):
    mol = Chem.MolFromSmiles(smiles)
    frame_width = 2
    filepath = os.path.join(IMAGE_DIR, filename)
    #size = (size,size)
    if mol:
        filepath = os.path.join(IMAGE_DIR, filename)
        d2d = Draw.MolDraw2DCairo(size,size)
        dopts = d2d.drawOptions()
        dopts.baseFontSize = 0.8
        dopts.fixedScale = .5
#        dopts.fixedBondLength = 40
        dopts.padding = .1
        dopts.centreMoleculesBeforeDrawing = True
        rdDepictor.SetPreferCoordGen(True)
        rdDepictor.Compute2DCoords(mol)
        rdDepictor.StraightenDepiction(mol)
        d2d.DrawMolecule(mol)
        d2d.FinishDrawing()
        d2d.WriteDrawingText(filepath)
# Create a new surface with frame
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
        ctx = cairo.Context(surface)
        # Draw the colored frame
        frame_color = (0, 1, 0) if in_stock else (1, 0.65, 0)  # Green if in stock, orange if not
        ctx.set_source_rgb(*frame_color)
        ctx.rectangle(0, 0, size, size)
        ctx.set_line_width(frame_width * 2)
        ctx.stroke()
        
        # Draw the molecule image inside the frame
        mol_surface = cairo.ImageSurface.create_from_png(filepath)
        ctx.set_source_surface(mol_surface, frame_width, frame_width)
        ctx.paint()

        surface.write_to_png(filepath)
        # Save the framed image

#        d2d.FinishDrawing()
#        d2d.WriteDrawingText(filepath)
        return f"/{filepath.replace(os.sep, '/')}"
    return None


def load_all_trees():
    if os.path.exists(ALL_TREES_FILE):
        with open(ALL_TREES_FILE, "r") as file:
            return json.load(file)
    return []
def load_master_paths():
    if os.path.exists(MASTER_FILE):
        with open(MASTER_FILE, "r") as file:
            return json.load(file)
    return []
def save_master_paths(data):
    with open(MASTER_FILE, "w") as file:
        json.dump(data, file, indent=4)
"""
def add_image_paths_to_nodes(tree, prefix, parent_hash="", parent_mol=None):
    if "smiles" in tree:
        # Generate a unique hash for this node
        tree["hash"] = generate_hash(f"{parent_hash}.{tree['smiles']}" if parent_hash else tree['smiles'])
        image_name_base = f"{prefix}_{tree['hash']}"
        if tree["type"] == "mol":
            preceding_reaction = None
            in_stock = tree.get("in_stock", False)
            if parent_mol and "children" in parent_mol:
                for child in parent_mol["children"]:
                    if child["type"] == "reaction":
                        preceding_reaction = child.get("metadata", {}).get("mapped_reaction_smiles")
                        break
            descendant_reaction = None
            if tree.get("children"):
                for child in tree["children"]:
                    if child["type"] == "reaction":
                        descendant_reaction = child.get("metadata", {}).get("mapped_reaction_smiles")
                        break
            # Generate neutral and highlighted images
            images, _ = generate_highlighted_molecule_images(tree["smiles"], image_name_base, preceding_reaction, descendant_reaction, in_stock)
            if images:
                tree["image_path"] = images[0][300]  # Neutral image
                tree["has_multiple_images"] = len(images) > 1
                if tree["has_multiple_images"]:
                    tree["highlighted_image_path"] = images[-1]  # Use the last image (merged if available, otherwise the single highlighted image)
        elif tree["type"] == "reaction":
            tree["image_path"] = generate_reaction_image(tree["smiles"], f"{image_name_base}.png")
    # Recursively add hashes and images to children
    for child in tree.get("children", []):
        add_image_paths_to_nodes(child, prefix, tree["hash"], tree if tree["type"] == "mol" else parent_mol)
    return tree
"""
def add_image_paths_to_nodes(tree, prefix, parent_hash="", parent_mol=None):
    if "smiles" in tree:
        tree["hash"] = generate_hash(f"{parent_hash}.{tree['smiles']}" if parent_hash else tree['smiles'])
        image_name_base = f"{prefix}_{tree['hash']}"
        if tree["type"] == "mol":
            preceding_reaction = None
            in_stock = tree.get("in_stock", False)
            if parent_mol and "children" in parent_mol:
                for child in parent_mol["children"]:
                    if child["type"] == "reaction":
                        preceding_reaction = child.get("metadata", {}).get("mapped_reaction_smiles")
                        break
            descendant_reaction = None
            if tree.get("children"):
                for child in tree["children"]:
                    if child["type"] == "reaction":
                        descendant_reaction = child.get("metadata", {}).get("mapped_reaction_smiles")
                        break
            # Generate neutral and highlighted images
            images = generate_highlighted_molecule_images(tree["smiles"], image_name_base, preceding_reaction, descendant_reaction, in_stock)
            if images:
                tree["image_paths"] = images["neutral"]
                tree["image_path"] = tree["image_paths"][300]  # Default to largest size for backwards compatibility
                tree["has_multiple_images"] = "highlighted" in images or "merged" in images
                if images["highlighted"]!={}:
                    tree["highlighted_image_paths"] = images["highlighted"]
                    tree["highlighted_image_path"] = tree["highlighted_image_paths"][300]  # Default to largest size for backwards compatibility
                if images["merged"]!={}:
                    tree["merged_image_paths"] = images["merged"]
                    tree["merged_image_path"] = tree["merged_image_paths"][300]  # Default to largest size for backwards compatibility
        elif tree["type"] == "reaction":
            tree["image_path"] = generate_reaction_image(tree["smiles"], f"{image_name_base}.png")
    # Recursively add hashes and images to children
    for child in tree.get("children", []):
        add_image_paths_to_nodes(child, prefix, tree["hash"], tree if tree["type"] == "mol" else parent_mol)
    return tree
"""
def create_user_session():
    user_id = str(uuid.uuid4())
    master_paths = load_master_paths()
    selected_paths = random.sample(range(len(master_paths)), min(20, len(master_paths)))
    user_session = {
        'user_id': user_id,
        'selected_paths': selected_paths,
        'feedback': {}
    }
    # Save user session
    with open(os.path.join(USER_SESSIONS_DIR, f'{user_id}.json'), 'w') as f:
        json.dump(user_session, f)
    return user_id
"""

@app.route("/submit_user_info", methods=["POST"])
def submit_user_info():
    data = request.json
    user_id = str(uuid.uuid4())
    master_paths = load_master_paths()
    selected_paths = random.sample(range(len(master_paths)), min(20, len(master_paths)))
    user_data = {
        "user_id": user_id,
        "education": data.get("education"),
        "years_experience": data.get("experience"),
        "selected_paths": selected_paths,
        "feedback": {}
    }
    with open(os.path.join(USER_SESSIONS_DIR, f"{user_id}.json"), "w") as f:
        json.dump(user_data, f)
    return jsonify({"success": True, "user_id": user_id})
@app.route("/get_user_trees", methods=["GET"])
def get_user_trees():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({"error": "User ID not provided"}), 400
    user_session_file = os.path.join(USER_SESSIONS_DIR, f'{user_id}.json')
    if not os.path.exists(user_session_file):
        return jsonify({"error": "User session not found"}), 404
    with open(user_session_file, 'r') as f:
        user_session = json.load(f)
    master_paths = load_master_paths()
    user_trees = [master_paths[i] for i in user_session['selected_paths']]
    return jsonify({"user_id": user_id, "trees": user_trees})
@app.route("/submit_feedback", methods=["POST"])
def submit_feedback():
    feedback_data = request.json
    if not feedback_data or "node" not in feedback_data or "user_id" not in feedback_data:
        return jsonify({"error": "Invalid feedback data"}), 400
    user_id = feedback_data["user_id"]
    node_data = feedback_data["node"]
    user_session_file = os.path.join(USER_SESSIONS_DIR, f'{user_id}.json')
    if not os.path.exists(user_session_file):
        return jsonify({"error": "User session not found"}), 404
    with open(user_session_file, 'r+') as f:
        user_session = json.load(f)
        user_session['feedback'][node_data['hash']] = {
            "feedback": node_data.get("feedback", []),  # Now an array
            "feedback_text": node_data.get("feedback_text", "None")
        }
        f.seek(0)
        json.dump(user_session, f)
        f.truncate()
    update_feedback_summary(node_data['hash'], user_id, node_data.get("feedback", []), node_data.get("feedback_text", "None"))
    return jsonify({"message": "Feedback submitted successfully"}), 200



@app.route("/submit_general_feedback", methods=["POST"])
def submit_general_feedback():
    feedback_data = request.json
    if not feedback_data or "node" not in feedback_data or "user_id" not in feedback_data:
        return jsonify({"error": "Invalid feedback data"}), 400
    user_id = feedback_data["user_id"]
    node_data = feedback_data["node"]
    root_hash = node_data.get("root_hash")  # Get the root molecule hash
    if not root_hash:
        return jsonify({"error": "Root hash not provided"}), 400
    user_session_file = os.path.join(USER_SESSIONS_DIR, f'{user_id}.json')
    if not os.path.exists(user_session_file):
        return jsonify({"error": "User session not found"}), 404
    # Update user session
    with open(user_session_file, 'r+') as f:
        user_session = json.load(f)
        if 'general_feedback' not in user_session:
            user_session['general_feedback'] = {}
        user_session['general_feedback'][root_hash] = {
            "feedback": node_data.get("general_feedback", "None"),
            "feedback_text": node_data.get("general_feedback_text", "None")
        }
        f.seek(0)
        json.dump(user_session, f)
        f.truncate()
    # Update feedback summary
    update_feedback_summary(f'general_feedback_{root_hash}', user_id, node_data.get("general_feedback", "None"), node_data.get("general_feedback_text", "None"))
    return jsonify({"message": "General feedback submitted successfully"}), 200

def update_feedback_summary(key, user_id, feedback, feedback_text):
    with open(FEEDBACK_SUMMARY_FILE, 'r+') as f:
        try:
            summary = json.load(f)
        except json.JSONDecodeError:
            summary = {}
        if key not in summary:
            summary[key] = []
        summary[key].append({
            'user_id': user_id,
            'feedback': feedback,  # Now an array
            'feedback_text': feedback_text
        })
        f.seek(0)
        json.dump(summary, f, indent=4)
        f.truncate()
if __name__ == "__main__":
    # Process the original file and create the master file if it doesn't exist
    if not os.path.exists(MASTER_FILE):
        all_trees = load_all_trees()
        master_paths = []
        for molecule_index, molecule_trees in enumerate(all_trees):
            top_tree = all_trees[str(molecule_index)][0]
            add_image_paths_to_nodes(top_tree, f"molecule_{molecule_index}")
            master_paths.append(top_tree)
        save_master_paths(master_paths)
    # Ensure feedback summary file exists
    if not os.path.exists(FEEDBACK_SUMMARY_FILE):
        with open(FEEDBACK_SUMMARY_FILE, 'w') as f:
            json.dump({}, f)
    app.run(debug=True)