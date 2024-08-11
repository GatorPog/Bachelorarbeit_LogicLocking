# Import the necessary libraries
import math
import os
from random import *
import argparse
import logging
import sys
"""
The code used here is an altered version of the code from https://github.com/gatelabdavis/SMTAttack/tree/master
"""

def rnd_enc(args):

    org_bench_address = args.original  # sys.argv[1] > original bench name
    if float(args.rnd_percent) < 0.1:
        percentage = "0" + str(int(float(args.rnd_percent) * 100))
    else:
        percentage = str(int(float(args.rnd_percent) * 100))

    rnd_obf_bench_folder = args.obfuscated  # sys.argv[2] > obfuscated bench file

    # Check if rnd_obf_bench_folder is a directory
    if os.path.isdir(rnd_obf_bench_folder):
        # Generate a new file name based on the original file name
        original_name = os.path.basename(org_bench_address)
        file_name, ext = os.path.splitext(original_name)
        rnd_obf_bench_folder = os.path.join(rnd_obf_bench_folder, f"{file_name}_obfuscated{ext}")

    key_values = []  # To store the key

    # Initialize variables
    inserted = 0
    bench_gates = 0
    new_bench = ""

    # Read the original .bench file
    with open(org_bench_address, 'r') as bench_file:
        for line in bench_file:
            new_bench += line
            if " = " in line:
                bench_gates += 1

    randins_number = math.floor(float(args.rnd_percent) * bench_gates)

    # Write the initial copy of the original circuit to the obfuscated file
    with open(rnd_obf_bench_folder, "w") as bench_file:
        bench_file.write(new_bench)

    while inserted < randins_number:
        new_gates = ""
        p_ins = ""
        p_outs = ""
        key_ins = ""

        with open(rnd_obf_bench_folder, 'r') as bench_file:
            for line in bench_file:
                if "INPUT(G" in line:
                    p_ins += line
                elif "INPUT(key" in line:
                    key_ins += line
                elif "OUTPUT" in line:
                    p_outs += line
                elif " = " in line and "_old" not in line:
                    if random() < randins_number/bench_gates and inserted < randins_number:
                        line1 = line.replace(" = ", "_enc = ")
                        gate_out = line[0: line.find(" =")]
                        if random() > 0.5:
                            line2 = gate_out + " = XNOR(keyinput" + str(inserted) + ", " + gate_out + "_enc)"
                            key_ins += "INPUT(keyinput" + str(inserted) + ")\n"
                            key_values.append(1)  # XNOR gates require a key value of 1 to produce correct output
                        else:
                            line2 = gate_out + " = XOR(keyinput" + str(inserted) + ", " + gate_out + "_enc)"
                            key_ins += "INPUT(keyinput" + str(inserted) + ")\n"
                            key_values.append(0)  # XOR gates require a key value of 0 to produce correct output
                        new_gates += line1 + line2 + "\n"

                        inserted += 1
                    else:
                        new_gates += line
                elif " = " in line and "_old" in line:
                    new_gates += line

        with open(rnd_obf_bench_folder, "w") as bench_file:
            bench_file.write(p_ins + "\n" + key_ins + "\n" + p_outs + "\n" + new_gates)

    # Generate the key file path
    key_file_path = os.path.splitext(rnd_obf_bench_folder)[0] + "_key.txt"

    # Write the key to a new file
    with open(key_file_path, "w") as key_file:
        key_file.write("".join(map(str, key_values)))
        key_values = list(key_values)
        key_file.write("\n")
        key_file.write(str(key_values))
    logging.error(f"Obfuscated .bench file written to: {rnd_obf_bench_folder}")
    logging.error(f"Key file written to: {key_file_path}")

# Example of how to call the rnd_enc function (in practice, this would be done via command-line arguments)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Randomly obfuscate a .bench file with XOR/XNOR gates.")
    parser.add_argument("--original", type=str, help="Path to the original .bench file")
    parser.add_argument("--rnd_percent", type=float, help="Percentage of gates to obfuscate")
    parser.add_argument("--obfuscated", type=str, help="Path to the output obfuscated .bench file or directory")
    args = parser.parse_args()

    rnd_enc(args)