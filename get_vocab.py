import sys
import argparse 
from hgraph import *
from rdkit import Chem
from rdkit.Chem import rdchem
from multiprocessing import Pool
from tqdm import tqdm
from itertools import permutations
from functools import reduce
import os
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import sys
import numpy as np
from PIL import Image
sys.path.append('/home/msun415/my_data_efficient_grammar/')

# from fuseprop import find_fragments


def mol_to_graph(mol, inds, r=False):
    graph = nx.Graph()
    for i, ind in enumerate(inds):
        a = mol.GetAtomWithIdx(ind)
        kwargs = {}
        if r: kwargs['r'] = a.GetBoolProp('r')
        graph.add_node(str(i), symbol=a.GetSymbol(), **kwargs)
    for i, u in enumerate(inds):
        for j, v in enumerate(inds):
            bond = mol.GetBondBetweenAtoms(u, v)
            if bond:
                graph.add_edge(str(i), str(j), bond_type=bond.GetBondType())
    return graph
        

def check_order(orig_mol, mol, cls, r=False):
    """
    Check whether subgraph induced by cls ~ mol
    r further checks if red atoms are the same
    """    
    
    graph_1 = mol_to_graph(orig_mol, cls, r=r)
    graph_2 = mol_to_graph(mol, list(range(mol.GetNumAtoms())), r=r)
    def node_match(a, b):
        return (a['symbol'] == b['symbol']) and ((not r) or (a['r'] == b['r']))    
    def edge_match(ab, cd):
        return ab['bond_type'] == cd['bond_type']
    if len(graph_1) != len(graph_2):
        return False
    if len(graph_1) == 1:
        res = list(dict(graph_1.nodes(data=True)).values())[0] == list(dict(graph_2.nodes(data=True)).values())[0]
    else:
        res = nx.is_isomorphic(graph_1, graph_2, node_match, edge_match)
    
    return res

    # below is much slower, but just to check
    ans = False
    for cluster in permutations(cls):
        bad = False
        for i in range(len(cluster)):
            if orig_mol.GetAtomWithIdx(cluster[i]).GetSymbol() != mol.GetAtomWithIdx(i).GetSymbol():
                bad = True
                continue
            if r and orig_mol.GetAtomWithIdx(cluster[i]).GetBoolProp('r') != mol.GetAtomWithIdx(i).GetBoolProp('r'):
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
            ans = True
            break
    if res != ans:
        breakpoint()
    return ans


def induce_mol(old_mol, cluster):
    mol = Chem.MolFromSmiles('')
    ed_mol = Chem.EditableMol(mol)
    for i, c in enumerate(cluster):
        ed_mol.AddAtom(rdchem.Atom(old_mol.GetAtomWithIdx(c).GetSymbol()))
    mol = ed_mol.GetMol()
    for i, c1 in enumerate(cluster):
        for j, c2 in enumerate(cluster):
            if i > j: continue
            b = old_mol.GetBondBetweenAtoms(c1, c2)
            if b:
                ed_mol.AddBond(i, j, b.GetBondType())
    
    mol = ed_mol.GetMol()    
    if not check_order(old_mol, mol, cluster):
        breakpoint()
    return mol


def annotate_extra_mol(m, labels, red_grps):
    """
    labels: list of atom id's with red grps
    red_grps: list of hyphen-separated red group id's
    """
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
            a.SetProp('atomLabel', f"R{red_grp}_{a.GetSymbol()}{l}")
            for b in a.GetBonds():
                b.SetBoolProp('r', True)    


def draw_hmol(path, G):
    fig = plt.Figure(figsize=(100, 100))
    ax = fig.add_subplot()
    options = {"node_size": 64000}
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=G.nodes, ax=ax, **options)    
    nx.draw_networkx_edges(G, pos, ax=ax)    
    nx.draw_networkx_labels(G, pos, font_size=100, ax=ax)
    fig.savefig(path)


# def dfs(i, G, walk):
#     walk.append(i)
#     for j in G[i]:
#         if j in walk:
#             continue
#         dfs(j, G, walk)

# dfs(0, G, walk)   

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
            vis[j] = True
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
        
        mol_a = induce_mol(hmol.mol, hmol.clusters[a])
        cluster_a = hmol.clusters[a]       
        # cluster_a = check_order(hmol, vocab_mols[smi_a], hmol.clusters[a])
        mol_b = induce_mol(hmol.mol, hmol.clusters[b])
        cluster_b = hmol.clusters[b]                
        ab = list(set(cluster_a) | set(cluster_b)) # red
        if len(ab) == max(len(cluster_a), len(cluster_b)):
            assert min(len(cluster_a), len(cluster_b)) == 1
            breakpoint()
            continue
        
        ab_mol = induce_mol(hmol.mol, ab)
        cluster = set(cluster_b)-set(cluster_a) # a is red              
        canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(ab_mol))                            
        if canon_smi in vocab_mols:
            orig_mol = vocab_mols[canon_smi][2].mol
            prev_cluster_a = vocab_mols[canon_smi][3]
            cur_mol_a = induce_mol(hmol.mol, cluster_a)
            breakpoint()
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


def extract_clusters(hmol, edges, vocab_mols, l):
    node_label = {}
    for node in hmol.mol_tree.nodes():
        # can we be dom? check for any dom neighbors
        # dom_nei = False
        # for nei in hmol.mol_tree[node]:
        #     dom_nei |= dom.get(nei, False)        
        # dom[node] = not dom_nei
        dic = {}
        cluster = set(hmol.clusters[node])
        nei_cluster = set()
        for nei in hmol.mol_tree[node]:
            cluster |= set(hmol.clusters[nei])
            nei_cluster |= set(hmol.clusters[nei])
        cluster = list(cluster)
        mol = induce_mol(hmol.mol, cluster)
        labels = [] # indices of atoms which are red
        red_grps = [] # for each atom in labels, which group(s) they're in
        for a in mol.GetAtoms():   
            red = 0
            if cluster[a.GetIdx()] not in hmol.clusters[node]:
                for nei in hmol.mol_tree[node]:
                    if cluster[a.GetIdx()] in hmol.clusters[nei]:
                        if red:
                            breakpoint()
                        red = nei+1
            if red:
                a.SetBoolProp('r', True)
                labels.append(a.GetIdx()+1)
                if red not in dic:                    
                    dic[red] = str(len(dic)+1)
                red_grp = dic[red]
                red_grps.append(red_grp)
            else:
                a.SetBoolProp('r', False)
       
        annotate_extra_mol(mol, labels, red_grps)
        try:
            canon_smi = Chem.CanonSmiles(Chem.MolToSmiles(mol))
        except:
            canon_smi = Chem.MolToSmiles(mol)

        vocab_i = -1
        for i, (k, vocab_mol, _) in enumerate(vocab_mols):
            # check isomorphism
            if mol.GetNumAtoms() != vocab_mol.GetNumAtoms():
                continue
                
            if check_order(mol, vocab_mol, list(range(mol.GetNumAtoms())), r=True):
                vocab_i = i
                break
                
        if vocab_i > -1:
            node_label[node] = str(vocab_mols[vocab_i][-1])
        else:                 
            vocab_mols.append((canon_smi, mol, l))
            node_label[node] = str(l)
            l += 1

    counts = {} 
    for k,v in node_label.items():
        counts[v] = counts.get(v, 0)+1
        node_label[k] = f"G{v}"
        if counts[v]>1:
            node_label[k] += f":{counts[v]-1}"
    hmol.mol_tree = nx.relabel_nodes(hmol.mol_tree, node_label)
    return l


def extract_fragments(mol, vocab_mols, l):
    fragments = find_fragments(mol)
    node_label = {}
    edges = []
    global_to_local_idxes = {}
    for i, (frag_smi, cluster) in enumerate(fragments):
        extras = []
        edges_i = []
        for j, (other_smi, other_cluster) in enumerate(fragments):
            if i == j:
                continue
            overlap = [c for c in other_cluster if c in cluster]
            if len(overlap) > 1:
                breakpoint()
            # if overlap is bond
            if len(overlap) == 1:
                
                ring_bonds = [r for ring in mol.GetRingInfo().AtomRings() for r in ring if overlap[0] in ring]
                is_ring = overlap[0] in ring_bonds  
         
                if is_ring and set(ring_bonds).intersection(cluster) == set(ring_bonds):
                    # for self, neighbors of overlap[0] is red
                    # for other, ring_bonds is red (see below)
                    neis = mol.GetAtomWithIdx(overlap[0]).GetNeighbors()
                    neis_inds = [a.GetIdx() for a in neis if a.GetIdx() in other_cluster]
                    assert len(neis_inds) == 1
                    extras.append(neis_inds)
                    # self_mol = induce_mol(mol, cluster + overlap[:1])
                    # self_mol.GetAtomWithIdx(len(cluster))
                elif is_ring and set(ring_bonds).intersection(other_cluster) == set(ring_bonds):
                    extras.append(list(set(ring_bonds)))
                else:
                    neis = mol.GetAtomWithIdx(overlap[0]).GetNeighbors()
                    neis_inds = [a.GetIdx() for a in neis if a.GetIdx() in other_cluster]
                    assert len(neis_inds) == 1
                    # neis_ = mol.GetAtomWithIdx(overlap[0]).GetNeighbors()
                    # neis_inds_ = [a.GetIdx() for a in neis if a.GetIdx() in other_cluster]                    
                    # assert (len(neis_inds) == 1) or (len(neis_inds_) == 1)
                    if len(neis_inds) == 1:
                        extras.append(neis_inds)
                        
                edge_data = [extras[-1]]
                # if (i, j) in red_match_lookup:
                #     index = red_match_lookup[(i,j)]
                #     edge_data['r_grp_1']=edges[index][-1]['r_grp_2']
                #     edges[index][-1]['r_grp_1'] = extras[-1]
                # else:
                #     red_match_lookup[(j,i)] = len(edges)
                edges_i.append([i,j,edge_data])

            elif len(overlap) > 1:
                breakpoint()
        # check extras don't collide        
        extra_atoms = [e for extra in extras for e in extra]
        extra_set = set(extra_atoms)
        if len(extra_set) != len(extra_atoms):
            breakpoint()
        cluster = [c for c in cluster if c not in extra_set]
        group = induce_mol(mol, cluster + extra_atoms)

        """
        Map atom idx in orig mol to group atom idx        
        """
        global_to_local_idxes[i] = dict(zip(cluster + extra_atoms, range(group.GetNumAtoms())))
        local_idx_map = dict(zip(cluster+extra_atoms, range(mol.GetNumAtoms())))
        for _, _, edge_data in edges_i:
            mapped_idxes = []
            for global_idx in edge_data[0]:
                mapped_idxes.append(local_idx_map[global_idx])
            edge_data.append(mapped_idxes)

        for k in range(len(cluster)):
            group.GetAtomWithIdx(k).SetBoolProp('r', False)        
        for k in range(len(cluster), len(cluster)+len(extra_atoms)):
            group.GetAtomWithIdx(k).SetBoolProp('r', True)
        labels = list(range(len(cluster)+1, len(cluster)+len(extra_atoms)+1))
        red_grps = []
        for ind, extra in enumerate(extras):
            for e in extra:
                red_grps.append(str(ind+1))
        annotate_extra_mol(group, labels, red_grps)
        vocab_i = -1
        for k, (_, vocab_mol, _) in enumerate(vocab_mols):
            # check isomorphism
            if group.GetNumAtoms() != vocab_mol.GetNumAtoms():
                continue                
            if check_order(group, vocab_mol, list(range(group.GetNumAtoms())), r=True):
                vocab_i = k
                break                
        if vocab_i > -1:
            node_label[i] = str(vocab_mols[vocab_i][-1])
        else:              
            vocab_mols.append((frag_smi, group, l))
            node_label[i] = str(l)
            l += 1      

        edges += edges_i

    counts = {} 
    for k,v in node_label.items():
        counts[v] = counts.get(v, 0)+1
        node_label[k] = f"G{v}"
        if counts[v]>1:
            node_label[k] += f":{counts[v]-1}"

    graph = nx.DiGraph()
    for i in range(len(fragments)):
        graph.add_node(node_label[i])
    
    
    """
    We have [(i, j, r_grp_1_global, r_grp_1_local)] and vice-versa
    Create i->j with r_grp_1, r_grp_2, b_2, b_1
    """    
    edge_data_lookup = {}
    for edge in edges:
        edge_data_lookup[tuple(edge[:2])] = edge[2]
    for src, dest, (global_idx_1, r_grp_1) in edges:
        breakpoint()
        try:
            if graph.has_edge(node_label[dest], node_label[src]): 
                continue
        except:
            breakpoint()
        b2 = [global_to_local_idxes[dest][idx] for idx in global_idx_1]
        global_idx_2, r_grp_2 = edge_data_lookup[(dest, src)]
        b1 = [global_to_local_idxes[src][idx] for idx in global_idx_2]
        graph.add_edge(node_label[src], node_label[dest], r_grp_1=r_grp_1, b2=b2, r_grp_2=r_grp_2, b1=b1)
        graph.add_edge(node_label[dest], node_label[src], r_grp_1=r_grp_2, b2=b1, r_grp_2=r_grp_1, b1=b2)

    return l, graph


            
            
def process(args, data):
    vocab_mols = []
    l = 1
    hmol_fig_dir = os.path.join(args.group_dir, 'hmol_figs')
    fig_dir = os.path.join(args.group_dir, 'figs')
    walk_dir = os.path.join(args.group_dir, 'walks')
    os.makedirs(hmol_fig_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(walk_dir, exist_ok=True)
    clearer_smiles = []
    clearer_indices = [40, 41, 42, 43, 44, 45, 46, 55, 76, 77, 78, 79, 80, 81, 82, 97, 98, 99, 100, 105, 106, 107, 108, 109, 171]
    pics = []
    for i, line in tqdm(enumerate(data)):
        # if i+1 not in [170,204,206,253,255,286,288,293,295,299,301,303,309]:
        if i+1 not in clearer_indices:
            continue
        s = line.strip("\r\n ") 
        hmol = MolGraph(s)           
        for j, a in enumerate(hmol.mol.GetAtoms()):
            a.SetProp('atomLabel', f"{a.GetSymbol()}{j+1}")
        Chem.Draw.MolToFile(hmol.mol, os.path.join(fig_dir, f'{i+1}.png'), size=(1000,1000))
        pics.append(os.path.join(fig_dir, f'{i+1}.png'))
        clearer_smiles.append(f"{i+1} {s}")

        # edges = []
        # vis = [False for _ in hmol.mol_tree.nodes()]
        # dfs(0, hmol, edges, vis)
        # explored = set()
        # dfs_explore(0, hmol, vis, explored)
        # extract_groups(hmol, edges, node_label, vocab_mols)
        # l = extract_clusters(hmol, edges, vocab_mols, l)        
        # G = hmol.mol_tree
        # draw_hmol(os.path.join(hmol_fig_dir, f'{i+1}.png'), G)        
        # cur_len = len(vocab_mols)
        # l, G = extract_fragments(hmol.mol, vocab_mols, l)
        # draw_hmol(os.path.join(hmol_fig_dir, f'{i+1}.png'), G)        
        # for _, mol, ind in vocab_mols[cur_len:]:
            # Chem.rdmolfiles.MolToMolFile(mol, os.path.join(args.group_dir, f"all_groups/{ind}.mol"))
            # Draw.MolToFile(mol, os.path.join(args.group_dir, f'all_groups/with_label/{ind}.png'), size=(2000, 2000))            
        
        # if not G.edges():
        #     assert len(G.nodes()) == 1
        #     root = list(G.nodes())[0]
        #     G.add_edge(root, root)
        # nx.write_edgelist(G, os.path.join(walk_dir, f"walk_{i}.edgelist"))
    with open('./seg_debug_smiles.txt', 'w+') as f: 
        f.write('\n'.join(clearer_smiles))
    grid_dim = 1+int(np.sqrt(len(pics)-1))
    f, axes = plt.subplots(grid_dim, (len(pics)-1)//grid_dim+1, figsize=(10,10))
    grid_dim = (len(pics)-1)//grid_dim+1
    for i in range(len(pics)):
        axes[i//grid_dim][i%grid_dim].imshow(Image.open(pics[i]))
        axes[i//grid_dim][i%grid_dim].set_title(f"{clearer_indices[i]}")
        axes[i//grid_dim][i%grid_dim].set_xticks([])
        axes[i//grid_dim][i%grid_dim].set_yticks([])
    f.savefig('./grid.png', dpi=1000.)
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



def main(args):
    if args.data_file:
        data = []
        lines = open(args.data_file).readlines()
        # data = [preprocess(l.split(',')[0]) for l in lines]
        data = [l.split(',')[0] for l in lines]
    else:
        data = [mol for line in sys.stdin for mol in line.split()[:2]]


    vocab = process(args, data)
    os.makedirs(os.path.join(args.group_dir, f'all_groups/with_label/'), exist_ok=True)
    # f = open(os.path.join(args.group_dir, "group_smiles.txt"), 'w+')        
    extra = ""
    for smi, mol, i in vocab:
        # f.write(f"{smi}\n")
        for a in mol.GetAtoms():
            r_grp = a.GetProp('r_grp')
            if r_grp:
                extra += f"{a.GetIdx()+1}:{r_grp}\n"
            else: 
                extra += f"{a.GetIdx()+1}\n"
        extra += "\n"
        # Chem.rdmolfiles.MolToMolFile(mol, os.path.join(args.group_dir, f"all_groups/{i}.mol"))
        # Draw.MolToFile(mol, os.path.join(args.group_dir, f'all_groups/with_label/{i}.png'), size=(2000, 2000))
    # f.close()

    # with open(os.path.join(args.group_dir, "all_groups/all_extra.txt"), 'w+') as f:
        # f.write(extra)    



def seg_mol(mol, mol_segs, vocab_mols, l):
    node_label = {}
    edges = []
    global_to_local_idxes = {}    
    clusters = []
    for i, g1 in enumerate(mol_segs):            
        extras = []
        edges_i = []
        if ':' in g1:
            if len(g1.split(':')) == 2:
                cluster, red_bond_info = g1.split(':')
            else:
                print(f"{g1} bad syntax")            
                return l, None
        else:
            cluster = g1
            if len(mol_segs) != 1:
                print(f"if {g1} has no red atoms, there should be only one segment")
                return l, None               
            else:
                red_bond_info = ''
        
        # syntax-check cluster and red_bond_info
        for c in cluster.split(','):
            if not c.isdigit():
                print(f"{cluster} bad syntax")            
                return l, None
        if red_bond_info:
            for red_bond_group in red_bond_info.split(';'):
                for r in red_bond_group.split(','):
                    if not r.isdigit():
                        print(f"{red_bond_info} bad syntax")
                        return l, None
                
                       
        if not cluster:
            print(f"{g1} bad syntax")            
            return l, None            
       
        if red_bond_info:
            cluster = set(map(int, cluster.split(',')))
            given_extras = red_bond_info.split(';')
            for e in given_extras:
                extra = list(map(int, e.split(',')))
                e_atoms = set(map(int, e.split(',')))
                extras.append(extra)
                for j, g2 in enumerate(mol_segs):
                    try:
                        extra_cluster, _ = g2.split(':')
                    except:
                        print(f"{g2} bad syntax")                    
                        return l, None
                    extra_cluster = set(map(int, extra_cluster.split(',')))    
                    if extra_cluster & e_atoms:
                        if extra_cluster & e_atoms != e_atoms:
                            print(f"{extra} is not entirely contained in {g2.split(':')[0]}")                        
                            return l, None                    
                        edges_i.append([i, j, [extras[-1]]])
                        break
                    if j == len(mol_segs)-1:
                        print(f"seg {i} extra {extra} is not among black atom sets")
                        return l, None
            for exist_cluster in clusters:
                if cluster & exist_cluster:
                    if cluster == exist_cluster:
                        breakpoint()
                    print(f"{cluster} should not intersect existing {exist_cluster}")
                    return l, None
            clusters.append(cluster)                
            cluster, red_bond_info = g1.split(':')
            cluster = list(map(int, cluster.split(',')))
        else:
            cluster = list(map(int, cluster.split(',')))
            clusters.append(set(cluster))
        extra_atoms = [e for extra in extras for e in extra]
        extra_set = set(extra_atoms)            
        if len(extra_set) != len(extra_atoms):
            from collections import Counter
            bad_extras = ','.join([str(c) for c, v in Counter(extra_atoms).items() if v>1])
            print(f"{bad_extras} should appear at most once in {red_bond_info}")
            return l, None
        if set(cluster) & set(extra_atoms):
            print(f"red {extra_atoms} should not intersect black atom set {cluster}")
            return l, None
        
        for idx in cluster+extra_atoms:
            if idx > mol.GetNumAtoms():
                print(f"{idx} should not exceed mol's number of atoms ({mol.GetNumAtoms()})")
                return l, None
        group_idxs = [idx-1 for idx in cluster + extra_atoms]
        
        group = induce_mol(mol, group_idxs)
    
        frag_smi = Chem.MolToSmiles(group)
        global_to_local_idxes[i] = dict(zip(cluster + extra_atoms, range(group.GetNumAtoms())))
        local_idx_map = dict(zip(cluster+extra_atoms, range(mol.GetNumAtoms())))
        for _, _, edge_data in edges_i:
            mapped_idxes = []
            for global_idx in edge_data[0]:
                mapped_idxes.append(local_idx_map[global_idx])
            edge_data.append(mapped_idxes)  

        for k in range(len(cluster)):
            group.GetAtomWithIdx(k).SetBoolProp('r', False)        
        for k in range(len(cluster), len(cluster)+len(extra_atoms)):
            group.GetAtomWithIdx(k).SetBoolProp('r', True)            
        labels = list(range(len(cluster)+1, len(cluster)+len(extra_atoms)+1))
        red_grps = []
        for ind, extra in enumerate(extras):
            for e in extra:
                red_grps.append(str(ind+1))            
        annotate_extra_mol(group, labels, red_grps)
        vocab_i = -1
        for k, (_, vocab_mol, _) in enumerate(vocab_mols):
            # check isomorphism
            if group.GetNumAtoms() != vocab_mol.GetNumAtoms():
                continue                
            if check_order(group, vocab_mol, list(range(group.GetNumAtoms())), r=True):
                vocab_i = k
                break
        if vocab_i > -1:
            node_label[i] = str(vocab_mols[vocab_i][-1])
        else:              
            vocab_mols.append((frag_smi, group, l))
            node_label[i] = str(l)
            l += 1      

        edges += edges_i            

    counts = {} 
    for k,v in node_label.items():
        counts[v] = counts.get(v, 0)+1
        node_label[k] = f"G{v}"
        if counts[v]>1:
            node_label[k] += f":{counts[v]-1}"

    graph = nx.DiGraph()
    for i in range(len(mol_segs)):
        graph.add_node(node_label[i])    
    
    """
    We have [(i, j, r_grp_1_global, r_grp_1_local)] and vice-versa
    Create i->j with r_grp_1, r_grp_2, b_2, b_1
    """    
    edge_data_lookup = {}
    for edge in edges:
        edge_data_lookup[tuple(edge[:2])] = edge[2]
    for src, dest, (global_idx_1, r_grp_1) in edges:
        """
        Process the inter-motif edge from src->dest
        global_idx_1: idxes in original mol of group src's extra atoms
        r_grp_1: idxes in group src of its extra atoms
        """
        try:
            if graph.has_edge(node_label[dest], node_label[src]): 
                continue
        except:
            breakpoint()
        if [global_to_local_idxes[src][idx] for idx in global_idx_1] != r_grp_1:
            breakpoint()
        b2 = [global_to_local_idxes[dest][idx] for idx in global_idx_1]
        if (dest, src) not in edge_data_lookup:
            print(f"segment {mol_segs[dest]} needs to connect back to {mol_segs[src]}")
            return l, None
        global_idx_2, r_grp_2 = edge_data_lookup[(dest, src)]
        b1 = [global_to_local_idxes[src][idx] for idx in global_idx_2]
        graph.add_edge(node_label[src], node_label[dest], r_grp_1=r_grp_1, b2=b2, r_grp_2=r_grp_2, b1=b1)
        graph.add_edge(node_label[dest], node_label[src], r_grp_1=r_grp_2, b2=b1, r_grp_2=r_grp_1, b1=b2)                            

    if reduce(lambda x,y: x|y, clusters) != set(range(1, mol.GetNumAtoms()+1)):
        print(f"black atoms don't add up to {list(range(1, mol.GetNumAtoms()+1))}")
        return l, None        
   
    # print(" ".join([mol_seg.split(":")[0] for mol_seg in mol_segs]))
    # return l, True
    return l, graph



def seg_groups(args):
    vocab_mols = []
    l = 1    
    hmol_fig_dir = os.path.join(args.seg_dir, 'hmol_figs')
    fig_dir = os.path.join(args.seg_dir, 'figs')
    walk_dir = os.path.join(args.seg_dir, 'walks')
    group_dir = os.path.join(args.group_dir, f'all_groups')
    label_dir = os.path.join(args.group_dir, f'all_groups/with_label')
    os.makedirs(hmol_fig_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(walk_dir, exist_ok=True)    
    os.makedirs(group_dir, exist_ok=True)   
    os.makedirs(label_dir, exist_ok=True)  

    data = []
    lines = open(args.data_file).readlines()
    data = [l.split(',')[0] for l in lines]    
    
    segs = defaultdict(list)
    with open(args.seg_file) as f:
        while True:
            line = f.readline().rstrip('\n')
            if not line:
                break
            mol_i, *_, seg_info = line.split()
            segs[int(mol_i)].append(seg_info)

    ignore_is = set()    
    if args.ignore_file:
        ignore_is = set(map(int, open(args.ignore_file).readlines()))

    for i, line in enumerate(data):
        if i+1 in ignore_is:
            continue
        s = line.strip("\r\n ") 
        mol = Chem.MolFromSmiles(s)
        Chem.Kekulize(mol, clearAromaticFlags=True)
        if i+1 not in segs:
            continue
        else:
            print(f"segmenting mol {i+1}")
        cur_len = len(vocab_mols)
        l, G = seg_mol(mol, segs[i+1], vocab_mols, l)                
        if G is None:
            open(args.out_file, 'a+').write(f"{i+1}\n")
            continue
        # draw_hmol(os.path.join(hmol_fig_dir, f'{i+1}.png'), G)
        # for _, mol, ind in vocab_mols[cur_len:]:
        #     Chem.rdmolfiles.MolToMolFile(mol, os.path.join(group_dir, f"{ind}.mol"))
        #     Draw.MolToFile(mol, os.path.join(label_dir, f'{ind}.png'), size=(200, 200))              
        # if not G.edges():
        #     breakpoint()
        #     assert len(G.nodes()) == 1
        #     root = list(G.nodes())[0]
        #     G.add_edge(root, root)
        # nx.write_edgelist(G, os.path.join(walk_dir, f"walk_{i}.edgelist"))        
    return vocab_mols
        


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--file')
    parser.add_argument('--group_dir')
    # seg info
    parser.add_argument('--seg_dir')
    parser.add_argument('--do_seg', action='store_true')
    parser.add_argument('--seg_file')
    parser.add_argument('--data_file')    
    parser.add_argument('--ignore_file')
    parser.add_argument('--out_file')
    args = parser.parse_args()

    if args.do_seg:    
        seg_groups(args)
    else:
        main(args)