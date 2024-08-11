import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


# Define Gate and Netlist classes
class Gate:
    def __init__(self, gate_type, inputs=None):
        self.gate_type = gate_type
        self.inputs = inputs if inputs else []
        self.outputs = []

    def add_output(self, output_gate):
        self.outputs.append(output_gate)

class Netlist:
    def __init__(self, gates=None, key_inputs=None, outputs=None):
        self.gates = gates if gates is not None else []
        self.key_inputs = key_inputs if key_inputs is not None else []
        self.outputs = outputs if outputs is not None else []

# Define BFS Extraction
def bfs_extraction(bfs_type, root_gate, encoding_table, max_depth, fan):
    locality_vector = []
    queue = [root_gate]
    current_depth = 1
    visited = set()  # Use a set to track visited gates

    visited.add(root_gate)

    while queue and current_depth <= max_depth:
        gate = queue.pop(0)

        neighbors = gate.inputs if bfs_type == 'Backward' else gate.outputs

        # Ensure the gate has a fixed number of fan-ins or fan-outs
        for i in range(fan - len(neighbors)):
            neighbors.append(Gate('Empty'))

        # Visit neighbors
        for neighbor in neighbors:
            if neighbor not in visited:
                queue.append(neighbor)
                visited.add(neighbor)
                encoded_value = encoding_table.get(neighbor.gate_type, 0)
                if bfs_type == 'Backward':
                    locality_vector.insert(0, encoded_value)
                else:
                    locality_vector.append(encoded_value)

        current_depth += 1

    return locality_vector

# Define Locality Vector Extraction
def extract_locality_vectors(netlist, backward_depth, forward_depth, fan_in, fan_out, encoding_table, activation_key=None):
    locality_vectors = []
    labels = []  # Store the labels corresponding to the locality vectors

    for key_input in netlist.key_inputs:
        # Find the XOR gate (or another gate) that the key input is connected to
        connected_gates = [gate for gate in netlist.gates if key_input in gate.inputs]

        if not connected_gates:
            print(f"Warning: Key input {key_input} is not connected to any gate. Skipping.")
            continue

        for connected_gate in connected_gates:
            # Extract backward BFS values starting from the gate connected to the key input
            lb = bfs_extraction('Backward', connected_gate, encoding_table, backward_depth, fan_in)

            # Extract encoding of the center gate (the XOR gate or similar)
            lkg = encoding_table.get(connected_gate.gate_type, 0)

            # Extract forward BFS values starting from the XOR gate (or similar)
            lf = bfs_extraction('Forward', connected_gate, encoding_table, forward_depth, fan_out)

            # Merge the values to form the locality vector
            locality_vector = lb + [lkg] + lf

            # Trim or pad the locality vector to 400 elements
            if len(locality_vector) < 400:
                locality_vector += [0] * (400 - len(locality_vector))
            else:
                locality_vector = locality_vector[:400]

            # Normalize values between 0 and 1
            max_value = max(locality_vector) if max(locality_vector) != 0 else 1
            locality_vector = [x / max_value for x in locality_vector]

            # Store the locality vector
            locality_vectors.append(locality_vector)

            # Store the corresponding label, if activation_key is provided
            if activation_key is not None:
                key_index = netlist.key_inputs.index(key_input)
                if key_index < len(activation_key):
                    labels.append(activation_key[key_index])
                    print(f"Added locality vector with label: {activation_key[key_index]}")
                else:
                    print(f"Warning: Index {key_index} out of range for activation_key. Skipping this gate.")
                    locality_vectors.pop()  # Remove the locality vector if it doesn't have a corresponding key
                    continue  # Skip this key_input if it doesn't have a corresponding activation key

    # Debug output
    print(f"Total locality vectors: {len(locality_vectors)}, Total labels: {len(labels)}")
    return locality_vectors, labels




# Define MLP Model
class MLP(nn.Module):
    def __init__(self, gss=False):
        super(MLP, self).__init__()
        if gss:  # GSS: 400x1000x256x2
            self.layer1 = nn.Linear(400, 1000)
            self.layer2 = nn.Linear(1000, 256)
        else:  # SRS: 400x512x128x2
            self.layer1 = nn.Linear(400, 512)
            self.layer2 = nn.Linear(512, 128)
        self.layer3 = nn.Linear(self.layer2.out_features, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.softmax(self.layer3(x))
        return x

# Define Key Prediction Accuracy (KPA)
def compute_kpa(predictions, labels):
    correct_predictions = (predictions == labels).sum().item()
    total_predictions = len(labels)
    return correct_predictions / total_predictions

# Define Parsing Function for ISCAS 85 Benchmarks
def parse_benchmark(file_path):
    gates = {}
    key_inputs = []
    outputs = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        if "INPUT" in line:
            gate_name = line.split('(')[1].split(')')[0]
            gate = Gate('INPUT')
            gates[gate_name] = gate
            key_inputs.append(gate)
        elif "OUTPUT" in line:
            gate_name = line.split('(')[1].split(')')[0]
            if gate_name in gates:
                outputs.append(gates[gate_name])
            else:
                gate = Gate('OUTPUT')
                gates[gate_name] = gate
                outputs.append(gate)
        else:
            parts = line.split('=')
            gate_name = parts[0].strip()
            expression = parts[1].strip()
            gate_type = expression.split('(')[0].strip()
            inputs = expression.split('(')[1].split(')')[0].split(',')

            # Ensure the input gates are properly linked
            input_gates = [gates[input_gate.strip()] for input_gate in inputs if input_gate.strip() in gates]
            gate = Gate(gate_type, inputs=input_gates)
            gates[gate_name] = gate

            # Link the output of each input gate to the current gate
            for input_gate in input_gates:
                input_gate.add_output(gate)

    return Netlist(gates=list(gates.values()), key_inputs=key_inputs, outputs=outputs)

# Main Script
if __name__ == "__main__":
    # Parameters
    Db = 5  # Backward Path Depth
    Df = 5  # Forward Path Depth
    Fin = 2  # Fan-In
    Fout = 3  # Fan-Out

    # Encoding Table for Gate Types
    encoding_table = {
        'NOT': 1, 'AND': 2, 'NAND': 3, 'OR': 4, 'XOR': 5,
        'NOR': 6, 'XNOR': 7, 'BUF': 8, 'FF': 9, 'Empty': 0
    }

    locked_benchmark_file_path = 'C:\\Users\\Johannes\\PycharmProjects\\Bachelorarbeit LogicLocks und Ki\\obfuscated benchmarks\\c5315_obfuscated.bench'

    activation_key = [1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]
    # Parse the locked ISCAS 85 benchmark
    netlist = parse_benchmark(locked_benchmark_file_path)
    # Extract labeled locality vectors for training (using the correct activation key)
    labeled_locality_vectors, labels = extract_locality_vectors(netlist, Db, Df, Fin, Fout, encoding_table, activation_key)

    # Extract unlabeled locality vectors for testing (without using the activation key)
    unlabeled_locality_vectors, _ = extract_locality_vectors(netlist, Db, Df, Fin, Fout, encoding_table)

    # Ensure labeled locality vectors and labels have the same length
    assert len(labeled_locality_vectors) == len(labels), (
        f"Mismatch between locality vectors ({len(labeled_locality_vectors)}) and labels ({len(labels)})"
    )

    # Prepare data for training and testing
    num_samples = len(labeled_locality_vectors)
    split_idx = int(0.8 * num_samples)

    # Training data
    train_x = torch.tensor(labeled_locality_vectors[:split_idx], dtype=torch.float32)
    train_y = torch.tensor(labels[:split_idx], dtype=torch.long)

    # Testing data
    test_x = torch.tensor(labeled_locality_vectors[split_idx:], dtype=torch.float32)
    test_y = torch.tensor(labels[split_idx:], dtype=torch.long)

    # Ensure sizes match before proceeding
    assert train_x.size(0) == train_y.size(0), "Mismatch between training data and labels size"
    assert test_x.size(0) == test_y.size(0), "Mismatch between testing data and labels size"

    print(f"Training data size: {train_x.size()}, Training labels size: {train_y.size()}")
    print(f"Testing data size: {test_x.size()}, Testing labels size: {test_y.size()}")

    # Create datasets and data loaders
    batch_size = 128

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize the MLP model
    model = MLP()  # Set gss=True for GSS
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0015)

    # Training loop
    num_epochs = 90
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Test the MLP model
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.numpy())
            all_labels.extend(batch_y.numpy())

    # Convert lists to tensors for accuracy computation
    all_predictions = torch.tensor(all_predictions)
    all_labels = torch.tensor(all_labels)

    accuracy = compute_kpa(all_predictions, all_labels)
    print(f"Key Prediction Accuracy (KPA) on test set: {accuracy * 100:.2f}%")