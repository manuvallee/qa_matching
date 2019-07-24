from eval import *
import sys
filename = sys.argv[1] 
ignore_noanswer = sys.argv[2]
map, mrr, acc, ap = eval_short(filename+'.scores', ignore_noanswer = False)

print(map,mrr,acc)
