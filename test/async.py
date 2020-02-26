import argparse
import log_parser

parser = argparse.ArgumentParser()
parser.add_argument("--param", type=str)
parser.add_argument("--model", type=str)
args = vars(parser.parse_args())

prefix = "log/async/"
param_file = prefix + args["param"]
model_file = prefix + args["model"]

param_pattern = "^rx MDL\ntx network.NetworkConfig\n(rx REQ\ntx \[\]mat.Dense\ntx \[\]mat.VecDense\nrx UPD\nrx \*\[\]mat.Dense\nrx \*\[\]mat.VecDense\n*)*$"
model_pattern = "^tx MDL\nrx \*network.NetworkConfig\n(tx REQ\nrx \*\[\]mat.Dense\nrx \*\[\]mat.VecDense\ntx UPD\ntx \[\]mat.Dense\ntx \[\]mat.VecDense\n*)*$"

print("Testing parameter server %s" % param_file)
log_parser.check_log(param_file, param_pattern)

print("Testing model replica %s" % model_file)
log_parser.check_log(model_file, model_pattern)