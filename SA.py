from simanneal import Annealer
import math
import random
import networkx as nx
from collections import defaultdict
from copy import deepcopy
import rustworkx as rx

from naqst import *


def dis(AG, a, b):
    return nx.shortest_path_length(AG, source = a, target = b)

class FindMinMapping(Annealer):
    
    def __init__(self, state, AG, mappre, topgraph):
        self.topgraph = topgraph
        self.AG = AG
        self.mappre = mappre
        super(FindMinMapping, self).__init__(state)  # important!
    
    def move(self):
        """Swaps two cities in the route."""
        initial_energy = self.energy()
        pn = len(self.AG.nodes()) - 1
        associated_edges = []
        for edge in list(self.AG.edges()):
            if (edge[0] in self.state) or (edge[1] in self.state):
                associated_edges.append(edge)

        random_edge = random.choice(associated_edges)
        if (random_edge[0] in self.state) and (random_edge[1] in self.state):
            idx1, idx2 = self.state.index(random_edge[0]), self.state.index(random_edge[1])
            self.state[idx1], self.state[idx2] = self.state[idx2], self.state[idx1]
        elif (random_edge[0] in self.state) and (random_edge[1] not in self.state):
            idx = self.state.index(random_edge[0])
            self.state[idx] = random_edge[1]
        elif (random_edge[0] not in self.state) and (random_edge[1] in self.state):
            idx = self.state.index(random_edge[1])
            self.state[idx] = random_edge[0]

        return self.energy() - initial_energy

    def energy(self):
        """Calculates the length of the route."""
        cost = 0
        #cost += distance_swap(self.mappre, self.state, self.AG)
        
        for ele in self.state:
            if ele == -1:
                continue
            if self.mappre[self.state.index(ele)] == -1:
                continue
            #print(self.state)
            #print(ele)
            cost += dis(self.AG, ele, self.mappre[self.state.index(ele)])
        
        penalty = 0
        for item in self.topgraph.edges():
            penalty += dis(self.AG, self.state[item[0]], self.state[item[1]])
      
        #penalty -= len(self.topgraph.edges())
        cost += (10 * penalty)
        return cost

def find_map_SA(mappre, graph_max, G):
    #tau = map2list(mappre, len(G.nodes()))
    tau = deepcopy(mappre)
    unoccupied = [item for item in mappre if item != -1]
    for node in graph_max.nodes():
        if tau[node] == -1:
            phy_node = random.choice(unoccupied)
            unoccupied.remove(phy_node)
            tau[node] = phy_node
    Fm = FindMinMapping(tau, G, mappre, graph_max)
    parameter = Fm.auto(minutes=2.0)
    Fm.set_schedule(parameter) 
    Fm.copy_strategy = "slice"
    tau1, cost1  = Fm.anneal()
    return tau1


if __name__ == "__main__":

    arch_size = 6
    Rb = math.sqrt(2)
    coupling_graph = generate_grid_with_Rb(arch_size,arch_size, Rb)
    G = nx.Graph()
    G.add_edges_from([(0,1),(1,2),(2,4),(4,3),(4,5)])
    mappre = [(0,1),(1,2),(2,1),(3,2),(2,4), -1]
    print(find_map_SA(mappre, G, coupling_graph))

