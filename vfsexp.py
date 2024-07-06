""" Find an embedding which is close to a given mapping """ 

import networkx as nx
import copy
import time
from collections import defaultdict


class Vf:
    def __init__(self,
                subgraph: nx.Graph, 
                graph: nx.Graph,
                curMap: dict, 
                stop: int,
                cost=None, 
                preMap=None, 
                upperbound=None
                ):
       # initial parameters
       self.graph = graph
       self.subgraph = subgraph
       self.curMap = {} #we can also start with a partial map
       
       # terminate the search earlier when time limit is reached
       self.start = time.time()
       self.stop = stop
       
       # Find the mapping that is closest to the preMap
       # self.cost = cost
       self.preMap = preMap 
       # self.last_bestMap = None
       self.upperbound = upperbound
       #self._order = dict()
       #self.order_list = []
       #self._order_graph = []
       #self.initOrder()
       #self.OrderForGraph()
       # self.result = []

    '''def OrderForGraph(self):
        graph_copy = self.graph.copy()
        while len(graph_copy.nodes()) != 0:
            degrees = dict(graph_copy.degree())
            min_degree = min(degrees.values())
            min_degree_nodes = [node for node, degree in degrees.items() if degree == min_degree]
            for nodes in min_degree_nodes:
                self._order_graph.append(nodes)
                graph_copy.remove_node(nodes)'''


  
    def dfsMatch(self, S): 
        """ S is the result to return. 
            In this method, it is either empty or the first complete map """ 

        # TODO: do we need S in this method?
        
        # print('**'*len(self.curMap), '!dfsMatch called')
        # print('**'*len(self.curMap), '!dfsMatch called', self.curMap)

        if self.is_complete():
            # print('The first complete map is found! \n', self.curMap)
            S = copy.copy(self.curMap)
            return S

        # TODO: is stop enforced correctly?
        if time.time()-self.start > self.stop: 
            # print('dfsmatch time exceeds', self.stop)
            return S

        '''Construct the current neighborhoods of the mapping: v is the next-vertex-to-be-mapped 
            X1 the set of candidates in self.graph '''

        nxt_vtx, subMNeighbor, gMNeighbor, gNMNeighbor = self.nxt_CandPairs()
        
        """ If we cannot find a correspondence for v then we backtrack """ 
        if (subMNeighbor and len(subMNeighbor) > len(gMNeighbor)) or not gNMNeighbor:
            # print('__'*len(self.curMap), 'End dfsMatch - I!', self.curMap, 'This branch failed!')
            return S
        
        for u in gNMNeighbor:
            if self.CandpairMeetsRules(nxt_vtx, u, subMNeighbor, gMNeighbor):

                """ Extend curMap with the candidate pair """
                self.curMap[nxt_vtx] = u
                #print("now the mapping", self.curMap)
                S = self.dfsMatch(S)
                
                """ The extension of curMap with {v:u} is failed."""
                self.curMap.pop(nxt_vtx)
                # print('__'*len(self.curMap), 'This extension failed and we consider the next candidate', self.curMap)
                    
        
        """ The call of dfsMatch (with the current map) is failed. """ 
        
        # print('__'*len(self.curMap), 'End dfsMatch!', self.curMap)
        return S
    
    """ Return the mapping which has minimal distance with preMap """ 
    def dfsMatchBest(self, S): 
        """ S is the current result. 
            In this method, it is at first empty {} and then the current best complete map. """ 
            
        if self.is_complete():

            if self.upperbound and self.mapdist() >= self.upperbound:
                #print("here 1")
                return S

            self.upperbound = self.mapdist()
            S = copy.copy(self.curMap)
            #print("here 2")
            return S

        if time.time() - self.start > self.stop:
            #print("the time is", time.time() - self.start)
            #print("here 3")
            return S

        '''Construct the current neighborhoods of the mapping: v is the next-vertex-to-be-mapped 
            X1 the set of candidates in self.graph '''

        nxt_vtx, subMNeighbor, gMNeighbor, gNMNeighbor = self.nxt_CandPairs()
        #print("nxt_vtx is", nxt_vtx)
        #print("subMNeighbor is", subMNeighbor)
        #print("gMneighbor is", gMNeighbor)
        #print("gNMNeighbor is", gNMNeighbor)
        
        """ If we cannot find a correspondence for v then we backtrack """ 
        if (subMNeighbor and len(subMNeighbor) > len(gMNeighbor)) or not gNMNeighbor:
            #print("here 4")
            return S
        
        for u in gNMNeighbor:
            if self.CandpairMeetsRules(nxt_vtx, u, subMNeighbor, gMNeighbor):

                self.curMap[nxt_vtx] = u
                #print("update curMap1", self.curMap)
                if self.upperbound and self.preMap and self.mapdist() >= self.upperbound:
                    #print("distance is", self.mapdist())
                    #print("self upperBound is ", self.upperbound)
                    self.curMap.pop(nxt_vtx)
                    
                    #print("update curMap2", self.curMap)
                    continue
                    
                S = self.dfsMatchBest(S)
                #print("S is ", S)
                
                """ We consider the next candidate."""
                self.curMap.pop(nxt_vtx)
                #print("update curMap3", self.curMap)
                          
        """ We backtrack and 
            consider the next candidate value of the previous key in curMap """ 
        #print("here 5")    
        return S 

    """ Return the mapping which has minimal distance with preMap """ 
    def dfsMatchAll(self, S):
        """ S is the current result. 
            In this method, it is at first empty and then the set if complete maps found so far. """ 
        # print('__'*len(self.curMap), 'Call dfsMatch', self.curMap, len(S))

        if self.is_complete():
            # print('__'*len(self.curMap), 'End dfsMatch!', '\n' f'Embedding {len(S)+1} found! \n', self.curMap)
            
            # self.curMap changes dynamically with search even if we put it in a list S.
            one_solution =  copy.copy(self.curMap)
            S.append(one_solution) 
            return S

        '''Construct the current neighborhoods of the mapping: v is the next-vertex-to-be-mapped 
            X1 the set of candidates in self.graph '''

        nxt_vtx, subMNeighbor, gMNeighbor, gNMNeighbor = self.nxt_CandPairs()
        
        """ If we cannot find a correspondence for v then we backtrack """ 
        if (subMNeighbor and len(subMNeighbor) > len(gMNeighbor)) or not gNMNeighbor:
            
            # print('__'*len(self.curMap), 'End dfsMatch!', self.curMap, 'This branch failed!', len(S))

            return S
        
        for u in gNMNeighbor:
            if self.CandpairMeetsRules(nxt_vtx, u, subMNeighbor, gMNeighbor):

                self.curMap[nxt_vtx] = u
                
                S = self.dfsMatchAll(S)
                
                """ The extension of curMap with {nxt_vtx:u} is failed."""
                self.curMap.pop(nxt_vtx)
                # print('__'*len(self.curMap), 'Backtrack', self.curMap, len(S))
                    
        
        """ The extension of with {nxt_vtx: } failed. We backtrack and 
            consider the next candidate value of the previous key in curMap """ 
        
        # print('__'*len(self.curMap), 'End dfsMatch!', self.curMap, len(S))
        return S
    
    """ Verify if {v:u} can extend curMap """
    def CandpairMeetsRules(self, v, u, subMNeighbor, gMNeighbor):
            
        if not self.curMap:
            return True     
        
        vNeighbor = list(nx.all_neighbors(self.subgraph, v))
        uNeighbor = list(nx.all_neighbors(self.graph, u))
                
        vPre, vSucc = preSucc(vNeighbor, self.curMap.keys())
        uPre, uSucc = preSucc(uNeighbor, self.curMap.values())
            
        ''' v cannot have more predecessors/successors than u'''
        if len(vPre) > len(uPre) or len(vSucc) > len(uSucc):
            return False
                       
        for pre in vPre:
            if self.curMap[pre] not in uPre:
                return False
        
        ''' v cannot have more successors in curMap than u does'''
        len1 = len(set(vNeighbor) & set(subMNeighbor)) #subMNeighborhood
        len2 = len(set(uNeighbor) & set(gMNeighbor))

        if len1 > len2:
            return False
        return True
        
        
    def is_complete(self):
        if len(self.curMap) == len(nx.nodes(self.subgraph)):
            return True
        return False

    def nxt_CandPairs(self):
        '''Construct the current neighborhoods of the mapping'''
        
        # print('u', self.curMap)
        # print('me', self.preMap)
        subMNeighbor = getNeiborhood(self.subgraph, self.curMap.keys())
        gMNeighbor = getNeiborhood(self.graph, self.curMap.values()) # unmapped neighbours in self.graph
        
        """ Select the next to-be-mapped node in self.subgraph and candidate nodes in self.graph. """ 
        X = subMNeighbor[:]
        gNMNeighbor = gMNeighbor[:]    

        if not subMNeighbor:

            """ If the subgraph is connected then curMap should be empty """
            if nx.is_connected(self.subgraph) and len(self.curMap) > 0: 
                raise Exception ('The subgraph is disconnected!')

            X = list(set(nx.nodes(self.subgraph)) - set(self.curMap.keys()))     
            
            subg_temp = self.subgraph.subgraph(X)
            X = max(nx.connected_components(subg_temp), key=len)
            
            gNMNeighbor = list(set(nx.nodes(self.graph)) - set(self.curMap.values()))

        # print(X, gNMNeighbor)

        """ Vf2++ rank vtx according to #(their neigbours in curMap), but it seems do not help. """
        # num_most_nghbrs = max([len(preSucc(list(nx.all_neighbors(self.subgraph, v)), self.curMap.keys())) for v in X])
        # X = [v for v in X if len(preSucc(list(nx.all_neighbors(self.subgraph, v)), self.curMap.keys())) == num_most_nghbrs]        
        
        '''Rank the unmapped neighbors by their degrees'''    
        max_deg = max([nx.degree(self.subgraph, v) for v in X])
        #min_deg = min([nx.degree(self.subgraph, v) for v in X])
 
        '''Select the node with the largest degree!'''
        nxt_vtx = [v for v in X if nx.degree(self.subgraph, v) ==  max_deg][0] #sl the next self.subgraph node to be mapped 
        #nxt_vtx = [v for v in X if nx.degree(self.subgraph, v) ==  min_deg][0]
        #indices = [self.order_list.index(x) for x in X]
        #nxt_vtx = self.order_list[sorted(indices)[0]]
        '''Remove those self.graph neighbours which cannot match the suggraph candidate node''' 
        gNMNeighbor = [ t for t in gNMNeighbor if\
                       nx.degree(self.subgraph, nxt_vtx) <= nx.degree(self.graph,t)]
        
        """ Rank nonmapped neighbours by distance to previous mapped node """
        if self.preMap:
            gNMN_deg = list([nx.shortest_path_length(self.graph, self.preMap[nxt_vtx], v), v] for v in gNMNeighbor)
            gNMN_deg.sort(key=lambda t: t[0])
            gNMNeighbor = list(t[1] for t in gNMN_deg)
        """ Using the edge bondary to sort the index"""

        #new_gNMNeighbor_1 = sorted(gNMNeighbor, key=lambda x: self._order_graph.index(x))
        #new_gNMNeighbor = new_gNMNeighbor_1[::-1]
        #print("new_g is ", new_gNMNeighbor)
        #print("g is ",gNMNeighbor )
        return nxt_vtx, subMNeighbor, gMNeighbor, gNMNeighbor
    
    def mapdist(self):
        distance = 0
        for p in self.subgraph.nodes():
            if p not in self.preMap.keys(): continue
            if p not in self.curMap.keys(): continue    
            distance += nx.shortest_path_length(self.graph, self.preMap[p], self.curMap[p])
        return distance

    '''def processBfsTree(self, source, orderIndex, added):
        self._order[orderIndex] = source
        added[source] = True

        endPosOfLevel = orderIndex
        startPosOfLevel = orderIndex
        lastAdded = orderIndex

        currConn = dict()
        for node in self.subgraph.nodes():
            currConn[node] = 0

        while orderIndex <= lastAdded:
            currNode = self._order[orderIndex]

            for oppositeNode in self.subgraph.neighbors(currNode):
                if not added[oppositeNode]:
                    lastAdded += 1
                    self._order[lastAdded] = oppositeNode
                    added[oppositeNode] = True

            if orderIndex > endPosOfLevel:
                for j in range(startPosOfLevel, endPosOfLevel + 1):
                    minInd = j
                    for i in range(j + 1, endPosOfLevel + 1):
                        if (currConn[self._order[i]] > currConn[self._order[minInd]] or \
                            (currConn[self._order[i]] == currConn[self._order[minInd]] and \
                                self.subgraph.degree(self._order[i]) >= self.subgraph.degree(self._order[minInd]))):
                            minInd = i

                    for oppositeNode in self.subgraph.neighbors(self._order[minInd]):
                        currConn[oppositeNode] += 1

                    self._order[j], self._order[minInd] = self._order[minInd], self._order[j]

                startPosOfLevel = endPosOfLevel + 1
                endPosOfLevel = lastAdded
            orderIndex += 1

        return

    def initOrder(self):
        added = dict()
        for node in self.subgraph.nodes():
            added[node] = False

        orderIndex = 0

        for node in self.subgraph.nodes():
            if not added[node]:
                minNode = node

                for node1 in self.subgraph.nodes():
                    if not added[node1] and self.subgraph.degree(minNode) < self.subgraph.degree(node1) :
                        minNode = node1

                self.processBfsTree(minNode, orderIndex, added)

        for key in sorted(self._order):
            self.order_list.append(self._order[key])

        return'''

#sl divide the neighborhood of a vertex into two disjoint parts: pre (in map) and succ (not in map)
def preSucc(Neighborhood, mapped_list):
    #vertexNeighbor and mapped_list can be empty
    
    Pre, Succ = [], []
    for vertex in Neighborhood:
        if vertex in mapped_list:            
            Pre.append(vertex)
        else:
            Succ.append(vertex)

    return Pre, Succ

def getNeiborhood(g, mapped_list):
    """Get non-mapped neighbours of curMap
        mapped_list: curMap.keys() or curMap.values()
        g: subgraph or graph
    """
    # print(mapped_list)
    Neighborhood = [q  for x in mapped_list for q in nx.all_neighbors(g,x)]
    Neighborhood = list(set(Neighborhood) - set(mapped_list)) 
    return Neighborhood





