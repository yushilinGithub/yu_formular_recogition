import argparse
import os
import pandas as pd
from dautils import blue_score
def main(args):
<<<<<<< HEAD
    assert os.path.exists(args.src_path),"{} not exists".format(args.src_path)

=======
<<<<<<< HEAD
    assert os.path.exists(args.src_path),"{} not exists".format(args.src_path)
=======
    assert os.path.exists(args.src_path),"{} not exists".format(args.path)
>>>>>>> fd3e137e3bb2be304ce03de39d315d99f2122c3d
>>>>>>> e76a5cdab7bcf49c11128972565dac8dc95c5716
    file = open(args.src_path,"r")
    data = []
    l_j = {}
    for i,f in enumerate(file):
        
        line = f.split('\t')
        if line[0].startswith('T'):
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> e76a5cdab7bcf49c11128972565dac8dc95c5716
            l_j['T'] = line[1].strip()
            l_j['line'] = line[0].split("-")[1]
        elif line[0].startswith('H'):
            l_j['H'] = line[2].strip()
        elif line[0].startswith('D'):
            l_j['D'] = line[2].strip()
<<<<<<< HEAD

=======
=======
            l_j['T'] = line[1]
            l_j['line'] = line[0].split("-")[1]
        elif line[0].startswith('H'):
            l_j['H'] = line[2]
        elif line[0].startswith('D'):
            l_j['D'] = line[2]
>>>>>>> fd3e137e3bb2be304ce03de39d315d99f2122c3d
>>>>>>> e76a5cdab7bcf49c11128972565dac8dc95c5716
        
        if len(l_j)==4:
            s=blue_score.compute_bleu([[l_j['T']]],[l_j['D']])
            l_j['s'] = s[0]
            data.append(l_j)
            l_j={}
    df = pd.DataFrame(data)
    df = df.sort_values(by = 's')
<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> e76a5cdab7bcf49c11128972565dac8dc95c5716
    df.to_csv(args.output)
    file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path",type=str,default="/home/public/yushilin/formular/results/mopai_chinese_support/generate-valid.csv")
    parser.add_argument("--output",type=str,default="/home/public/yushilin/formular/results/mopai_chinese_support/data_observation.lst")
<<<<<<< HEAD
=======
=======
    df.to_csv("obs.csv")
    file.close()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path",type=str,default="/home/public/yushilin/formular/results/test_swin_200/generate-test.txt")
>>>>>>> fd3e137e3bb2be304ce03de39d315d99f2122c3d
    args = parser.parse_args()
    main(args)
>>>>>>> e76a5cdab7bcf49c11128972565dac8dc95c5716
