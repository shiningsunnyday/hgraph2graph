import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_file')
    parser.add_argument('--out_file')
    args = parser.parse_args()
    lines = open(args.in_file).readlines()
    f = open(args.out_file, 'w+')
    for l in lines:
        l = l.rstrip('\n')

        try:

            if '#' not in l.split()[0]:
                mol_no, bond, *_ = l.split()
                f.write(f"{mol_no} {bond} ; ;\n")
            else:
                mol_no, *bonds, r_grp1, r_grp2 = l.split()
                assert '#' in mol_no
                l = l.replace('#', '')
                f.write(l+'\n')
        except:
            breakpoint()
    f.close()
    