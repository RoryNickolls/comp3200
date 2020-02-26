import argparse
import log_parser

parser = argparse.ArgumentParser()
parser.add_argument("--param", type=str)
parser.add_argument("--model1", type=str)
parser.add_argument("--model2", type=str)
args = vars(parser.parse_args())

prefix = "log/sync/"
param_file = prefix + args["param"]
model1_file = prefix + args["model1"]
model2_file = prefix + args["model2"]

param_pattern = "^(rx MDL\ntx network.NetworkConfig\n){2}((((tx|rx) (?!CON).*\n))*(tx CON\n(((tx|rx) (?!CON).*\n))*tx CON\n*))*$"
model_pattern = "^tx MDL\nrx \*network.NetworkConfig\n(tx REQ\nrx \*\[\]mat.Dense\nrx \*\[\]mat.VecDense\ntx UPD\ntx \[\]mat.Dense\ntx \[\]mat.VecDense\nrx CON\n*)*$"

print("Testing parameter server %s" % param_file)
log_parser.check_log(param_file, param_pattern)

print("Testing model replica 1 %s" % model1_file)
log_parser.check_log(model1_file, model_pattern)

print("Testing model replica 2 %s" % model2_file)
log_parser.check_log(model2_file, model_pattern)