import sys
import argparse 
from hgraph import *
from rdkit import Chem
from rdkit.Chem import rdchem
from multiprocessing import Pool
from tqdm import tqdm
from itertools import permutations
import os
from rdkit.Chem import Draw

def check_order(hmol, mol, cls):
    for cluster in permutations(cls):
        bad = False
        for i in range(len(cluster)):
            if hmol.mol.GetAtomWithIdx(cluster[i]).GetSymbol() != mol.GetAtomWithIdx(i).GetSymbol():
                bad = True
                continue
            for j in range(len(cluster)):
                bond = hmol.mol.GetBondBetweenAtoms(cluster[i], cluster[j])
                bond_ = mol.GetBondBetweenAtoms(i, j)
                if (bond == None) ^ (bond_ == None): 
                    bad = True
                    break
                if bond:
                    if bond.GetBondType() != bond_.GetBondType():
                        bad = True
                        break  
            if bad:
                break
        if not bad:
            return cluster


def induce_mol(hmol, cluster):
    mol = Chem.MolFromSmiles('')
    ed_mol = Chem.EditableMol(mol)
    for i, c in enumerate(cluster):
        ed_mol.AddAtom(rdchem.Atom(hmol.mol.GetAtomWithIdx(c).GetSymbol()))
    mol = ed_mol.GetMol()
    for i, c1 in enumerate(cluster):
        for j, c2 in enumerate(cluster):
            if i > j: continue
            b = hmol.mol.GetBondBetweenAtoms(c1, c2)
            if b:
                ed_mol.AddBond(i, j, b.GetBondType())
    
    mol = ed_mol.GetMol()     
    check_order(hmol, mol, cluster)
    return mol


def annotate_extra_mol(m, labels, red_grps):
    for i, a in enumerate(m.GetAtoms()): 
        a.SetProp("molAtomMapNumber", str(i+1))    
    for i in range(m.GetNumAtoms()):
        m.GetAtomWithIdx(i).SetBoolProp('r', False)
        m.GetAtomWithIdx(i).SetProp('r_grp', '')
    for i in range(m.GetNumBonds()):
        m.GetBondWithIdx(i).SetBoolProp('r', False)
        m.GetBondWithIdx(i).SetBoolProp('picked', False)
    for l, red_grp in zip(labels, red_grps):
        if l <= m.GetNumAtoms():
            a = m.GetAtomWithIdx(l-1)
            a.SetBoolProp('r', True)
            a.SetProp('r_grp', red_grp)
            a.SetProp('atomLabel', f"R_{a.GetSymbol()}{l}")
            for b in a.GetBonds():
                b.SetBoolProp('r', True)    


def process(data):
    vocab_mols = {}
    l = 0
    for line in data:
        s = line.strip("\r\n ")
        hmol = MolGraph(s)
        node_data = hmol.mol_tree.nodes(data=True)
        for edge in hmol.mol_tree.edges(data=True):
            a = edge[0]
            b = edge[1]
         
            mol_a = induce_mol(hmol, hmol.clusters[a])
            cluster_a = hmol.clusters[a]       
            # cluster_a = check_order(hmol, vocab_mols[smi_a], hmol.clusters[a])
            mol_b = induce_mol(hmol, hmol.clusters[b])
            cluster_b = hmol.clusters[b]                
            ab = list(set(cluster_a) | set(cluster_b)) # red
            if len(ab) == max(len(cluster_a), len(cluster_b)):
                continue
            
            ab_mol = induce_mol(hmol, ab)
            if len(cluster_b) < len(cluster_a):
                cluster = set(cluster_b)-set(cluster_a) 
            else:
                cluster = set(cluster_a)-set(cluster_b)
            for i, at in enumerate(ab):
                ab_mol.GetAtomWithIdx(i).SetBoolProp('r', at in cluster)
            
            labels = [i+1 for i in range(ab_mol.GetNumAtoms()) if ab[i] in cluster]
            red_grps = ['1' for _ in labels]     
            canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(ab_mol))                            
            if canon_smi in vocab_mols:
                continue

            annotate_extra_mol(ab_mol, labels, red_grps)   
            vocab_mols[canon_smi] = (l, ab_mol)
            l += 1    

    return vocab_mols


def preprocess(smi):
    mol = Chem.MolFromSmiles(smi)
    for i in range(mol.GetNumAtoms()-1,-1,-1):
        a = mol.GetAtomWithIdx(i)
        if a.GetSymbol() == '*':
            ed_mol = Chem.EditableMol(mol)
            ed_mol.RemoveAtom(i)
            mol = ed_mol.GetMol()
    return Chem.MolToSmiles(mol)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--group_dir')
    args = parser.parse_args()

    if args.file:
        data = []
        lines = open(args.file).readlines()
        data = [preprocess(l.split(',')[0]) for l in lines]
    else:
        data = [mol for line in sys.stdin for mol in line.split()[:2]]


    vocab = process(data)
    os.makedirs(os.path.join(args.group_dir, f'with_label/'), exist_ok=True)
    f = open(os.path.join(args.group_dir, "group_smiles.txt"), 'w+')        
    for smi, (i, mol) in sorted(vocab.items(), key=lambda x: x[0]):
        f.write(f"{smi}\n")
        Draw.MolToFile(mol, os.path.join(args.group_dir, f'with_label/{i+1}.png'), size=(2000, 2000))
    f.close()
