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
import matplotlib.pyplot as plt
import networkx as nx

def check_order(orig_mol, mol, cls):
    """
    Check whether subgraph induced by cls ~ mol
    """    
    for cluster in permutations(cls):
        bad = False
        for i in range(len(cluster)):
            if orig_mol.GetAtomWithIdx(cluster[i]).GetSymbol() != mol.GetAtomWithIdx(i).GetSymbol():
                bad = True
                continue
            for j in range(len(cluster)):
                bond = orig_mol.GetBondBetweenAtoms(cluster[i], cluster[j])
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
    check_order(hmol.mol, mol, cluster)
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


def draw_hmol(path, hmol, node_data):
    fig = plt.Figure(figsize=(100, 100))
    ax = fig.add_subplot()
    G = hmol.mol_tree
    options = {"node_size": 16000}
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, ax=ax, **options)    
    nx.draw_networkx_edges(G, pos, ax=ax)    
    nx.draw_networkx_labels(G, pos, node_data, font_size=100, ax=ax)
    fig.savefig(path)


def dfs(i, hmol, edges, vis):
    vis[i] = True
    for j in hmol.mol_tree[i]:
        if vis[j]: 
            continue
        breakpoint()
        if len(hmol.clusters[j]) == 1:
            for k in hmol.mol_tree[j]:    
                if vis[k]: continue
                edges.append((i, k))
            vis[j] = 1
            for k in hmol.mol_tree[j]:
                if vis[k]: continue
                dfs(k, hmol, edges, vis)
                
        else:
            edges.append((i, j))
            dfs(j, hmol, edges, vis)


def dfs_explore(i, hmol, vis, explored):
    vis[i] = True
    hmol.clusters[i] = tuple(a for a in hmol.clusters[i] if a not in explored)
    explored |= set(hmol.clusters[i])
    for j in hmol.mol_tree[i]:
        if vis[j]: 
            continue
    
        dfs_explore(j, hmol, vis, explored)

def extract_groups(hmol, edges, node_label, vocab_mols, l):
    for edge in edges:
        a = edge[0]
        b = edge[1]
        
        mol_a = induce_mol(hmol, hmol.clusters[a])
        cluster_a = hmol.clusters[a]       
        # cluster_a = check_order(hmol, vocab_mols[smi_a], hmol.clusters[a])
        mol_b = induce_mol(hmol, hmol.clusters[b])
        cluster_b = hmol.clusters[b]                
        ab = list(set(cluster_a) | set(cluster_b)) # red
        if len(ab) == max(len(cluster_a), len(cluster_b)):
            assert min(len(cluster_a), len(cluster_b)) == 1
            breakpoint()
            continue
        
        ab_mol = induce_mol(hmol, ab)
        cluster = set(cluster_b)-set(cluster_a) # a is red              
        canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(ab_mol))                            
        if canon_smi in vocab_mols:
            orig_mol = vocab_mols[canon_smi][2].mol
            prev_cluster_a = vocab_mols[canon_smi][3]
            cur_mol_a = induce_mol(hmol, cluster_a)
            if check_order(orig_mol, cur_mol_a, prev_cluster_a):
                if a in node_label:
                    node_label[a] += f"-{vocab_mols[canon_smi][0]}"
                else:
                    node_label[a] = str(vocab_mols[canon_smi][0])
            else:
                breakpoint()
        else:                              
            if a in node_label: 
                node_label[a] += f"-{str(l)}"
            else:
                node_label[a] = str(l)

        # set node with group
        for j, at in enumerate(ab):
            ab_mol.GetAtomWithIdx(j).SetBoolProp('r', at in cluster)
        
        # indices of red atoms
        labels = [j+1 for j in range(ab_mol.GetNumAtoms()) if ab[j] in cluster]
        red_grps = ['1' for _ in labels]     

        annotate_extra_mol(ab_mol, labels, red_grps)   
        if canon_smi not in vocab_mols:
            vocab_mols[canon_smi] = (l, ab_mol, hmol, cluster_a, cluster_b)
            l += 1
    return l


def extract_clusters(hmol, edges, node_label, vocab_mols, l):
    dom = {}
    for node in hmol.mol_tree.nodes():
        # can we be dom? check for any dom neighbors
        # dom_nei = False
        # for nei in hmol.mol_tree[node]:
        #     dom_nei |= dom.get(nei, False)
        
        # dom[node] = not dom_nei
        dom_nei = False
        cluster = set(hmol.clusters[node])
        nei_cluster = set()
        for nei in hmol.mol_tree[node]:
            cluster |= set(hmol.clusters[nei])
            nei_cluster |= set(hmol.clusters[nei])
        cluster = list(cluster)
        mol = induce_mol(hmol, cluster)
        labels = []
        for a in mol.GetAtoms():   
            red = False
            if not dom_nei and (cluster[a.GetIdx()] not in hmol.clusters[node]):
                red = True
            if dom_nei and (cluster[a.GetIdx()] in nei_cluster):
                red = True
            if red:
                a.SetBoolProp('r', True)
                labels.append(a.GetIdx()+1)
            else:
                a.SetBoolProp('r', False)

        red_grps = ['1' for _ in labels]     
    
        canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        if canon_smi in vocab_mols:
            node_label[node] = vocab_mols[canon_smi][0]
        else:
            node_label[node] = str(l)

        annotate_extra_mol(mol, labels, red_grps)   
        if canon_smi not in vocab_mols:            
            vocab_mols[canon_smi] = (l, mol)
            l += 1
    return l



def process(args, data):
    vocab_mols = {}
    walks = []
    l = 1
    hmol_fig_dir = os.path.join(args.group_dir, 'hmol_figs')
    fig_dir = os.path.join(args.group_dir, 'figs')
    os.makedirs(hmol_fig_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    for i, line in enumerate(data):
        s = line.strip("\r\n ")        
        hmol = MolGraph(s)           
        for j, a in enumerate(hmol.mol.GetAtoms()):
            a.SetProp('atomLabel', f"{a.GetSymbol()}{j+1}")
        Chem.Draw.MolToFile(hmol.mol, os.path.join(fig_dir, f'{i+1}.png'))             
        node_label = {}
        edges = []
        vis = [False for _ in hmol.mol_tree.nodes()]
        # dfs(0, hmol, edges, vis)
        explored = set()
        dfs_explore(0, hmol, vis, explored)
        # extract_groups(hmol, edges, node_label, vocab_mols)
        l = extract_clusters(hmol, edges, node_label, vocab_mols, l)
        draw_hmol(os.path.join(hmol_fig_dir, f'{i+1}.png'), hmol, node_label)

    return vocab_mols, walks


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


    vocab, walks = process(args, data)
    os.makedirs(os.path.join(args.group_dir, f'with_label/'), exist_ok=True)
    f = open(os.path.join(args.group_dir, "group_smiles.txt"), 'w+')        
    for smi, (i, mol, *_) in sorted(vocab.items(), key=lambda x: x[0]):
        f.write(f"{smi}\n")
        Draw.MolToFile(mol, os.path.join(args.group_dir, f'with_label/{i}.png'), size=(2000, 2000))
    f.close()
