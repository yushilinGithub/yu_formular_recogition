import sys, logging, argparse, os
from pathlib import Path
def process_args(args):
    parser = argparse.ArgumentParser(description='Generate vocabulary file.')

<<<<<<< HEAD
    parser.add_argument('--data-path', dest='data_path',required=True,
                        type=str,
                        help=('Input file containing a tokenized formula per line.'
                        ))
    parser.add_argument('--output-file', dest='output_file',required=True,
                        type=str,
=======
    parser.add_argument('--data-path', dest='data_path',
                        type=str, default="/home/public/yushilin/handwrite/HandWriteFormula/",
                        help=('Input file containing a tokenized formula per line.'
                        ))
    parser.add_argument('--output-file', dest='output_file',
                        type=str, default="/home/yushilin/workspace/ocr/okayOCR/dictionary/handWriteFormulaComp.txt",
>>>>>>> e76a5cdab7bcf49c11128972565dac8dc95c5716
                        help=('Output file for putting vocabulary.'
                        ))
    parser.add_argument('--unk-threshold', dest='unk_threshold',
                        type=int, default=1,
                        help=('If the number of occurences of a token is less than (including) the threshold, then it will be excluded from the generated vocabulary.'
                        ))
    parser.add_argument('--log-path', dest="log_path",
                        type=str, default='log.txt',
                        help=('Log file path, default=log.txt' 
                        ))
    parameters = parser.parse_args(args)
    return parameters

def main(args):
    parameters = process_args(args)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s',
        filename=parameters.log_path)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    logging.info('Script being executed: %s'%__file__)


    data_path = Path(parameters.data_path)
    assert data_path.exists(), data_path

    txt_files = [data_path / "train/chemistry/train.txt",
                data_path / "train/math/train.txt",
                data_path / "train/physics/train.txt",
                data_path / "val/Task1/valid.txt"]
    
    latexs = []
    for label_file in txt_files:
        lines = label_file.read_text().strip().split("\n")
        for line in lines:
            latex = line.split("\t")[1]
            if latex.strip():
                latexs.append(latex)
    vocab = {}
    max_len = 0
    for latex in latexs:
        tokens = latex.split()
        tokens_out = []
        for token in tokens:
            tokens_out.append(token)
            if token not in vocab:
                vocab[token] = 0
            vocab[token] += 1

    vocab_sort = sorted(list(vocab.keys()))
    vocab_out = []
    num_unknown = 0
    for word in vocab_sort:
        if vocab[word] > parameters.unk_threshold:
            vocab_out.append(word)
        else:
            num_unknown += 1
    #vocab = ["'"+word.replace('\\','\\\\').replace('\'', '\\\'')+"'" for word in vocab_out]
    #vocab = [word for word in vocab_out]

    with open(parameters.output_file, 'w') as fout:
        vocab = dict(sorted(vocab.items(), key=lambda item: item[1],reverse=True))
        for key,value in vocab.items():
            fout.write("{} {}\n".format(key,value))
    logging.info('#UNK\'s: %d'%num_unknown)

if __name__ == '__main__':
    main(sys.argv[1:])
    logging.info('Jobs finished')
