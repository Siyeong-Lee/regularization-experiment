
import sys
import subprocess
import multiprocessing as mp
# from Queue import Queue
from multiprocessing import Queue
from threading import Thread

def run(e, a, n, m, d, l, r, p, x, o, w, q, z):
    subprocess.call('THEANO_FLAGS=device=gpu,floatX=float32 python go.py -e {0} -a {1} -n {2} -m {3} -d {4} -l {5} -r {6} -p {7} -x {8} -o {9} -w {10} -q {11} -z {12}'.format(e, a, n, m, d, l, r, p, x, o, w, q, z), shell=True)

epoch = [50]
aug = [False]
noise = [ False]
maxout = [False]
dropout = [False]
l1 = [False]
l2 = [True]
maxpooling = [False]
deep = [False]
noise_rate = [0.01]
weight_constraint = [True, False]
l1_weight = [0.001, 0.005, 0.01, 0.05, 0.1]
l2_weight = [0.01, 0.05, 0.1, 0.5]
for e in epoch:
    for a in aug:
        for n in noise:
            for m in maxout:
                for d in dropout:
                    for one in l1:
                        for two in l2:
                            for p in maxpooling:
                                for x in deep:
                                    for o in noise_rate:
                                        for w in weight_constraint:
                                            for q in l1_weight:
                                                for z in l2_weight:
                                                    run(e, a, n, m, d, one, two, p, x, o, w, q, z)
