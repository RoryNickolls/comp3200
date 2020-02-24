import argparse
import log_parser

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--model', type=str)
parser.add_argument('--param', type=str)
args = vars(parser.parse_args())

prefix = "log/downpour/"
data_file = prefix + args["data"]
model_file = prefix + args["model"]
param_file = prefix + args["param"]

data_pattern = '^(rx \*network.Data)\n(rx REQ \d+\ntx \[\]\[\]network.Record\n*)*$'
model_pattern = '^(tx MDL\nrx \*network.NetworkConfig)\n((tx REQ \d+\nrx \*\[\]\[\]network.Record)\n(tx REQ\nrx \*\[\]mat.Dense\nrx \*\[\]mat.VecDense\n*(tx UPD\ntx \[\]mat.Dense\ntx \[\]mat.VecDense)*\n*)*)*$'
param_pattern = '^(rx MDL\ntx network.NetworkConfig)\n((rx REQ\ntx \[\]mat.Dense\ntx \[\]mat.VecDense\n)+(rx UPD\nrx \*\[\]mat.Dense\nrx \*\[\]mat.VecDense\n*)+)*$'
print("Testing data server log %s" % data_file)

if log_parser.check_log(data_file, data_pattern):
    print("Data server test passed.")
else:
    print("Data server test failed.")

print("Testing model replica log %s" % model_file)

if log_parser.check_log(model_file, model_pattern):
    print("Model replica test passed.")
else:
    print("Model replica test failed.")

print("Testing parameter server log %s" % param_file)

if log_parser.check_log(param_file, param_pattern):
    print("Parameter server test passed.")
else:
    print("Parameter server test failed.")

