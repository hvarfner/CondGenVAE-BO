import hypermapper
from vae import quantity_of_interest
import json

with open('objective/mnist_scenario.json', 'r') as f:
    parameters_file = json.load(f)

print(parameters_file)