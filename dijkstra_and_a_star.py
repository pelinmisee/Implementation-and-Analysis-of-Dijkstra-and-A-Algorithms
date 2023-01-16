import math
from heapq import *
from collections import defaultdict
class Graph():
    def __init__(self,node_amount):
       self.node_amount = node_amount
       self.g = self.graph()
       self.a_star_parents = {}

    # this function is for creating edges between vertices as it was wanted.
    def edges(self,node_amount):  # O(v^2) that travels all vertices and checks the condition for each vertex
        edges = []
        edge_amount = 0
        for i in range(1, node_amount + 1):
            for j in range(1, node_amount + 1):
                # if i and j are not same and their difference is less than or equal to 3, there exists an edge between them
                if i != j and abs(i - j) <= 3:
                    # weight is added to the edge as i+j
                    edges.append((i, j, i + j))
                    edge_amount += 1
        return edges, edge_amount

    # this function provides creating graph from edges
    def graph(self):  # O(E) that travels all edges at once
        edges, edge_amount = self.edges(self.node_amount)
        g = defaultdict(list)
        for l, r, c in edges:
            g[l].append((c, r))
            g[r].append((c, l))
        return g

    def initialize_single_source(self, source):  # O(v) it travels all vertices at once
        d = {}
        for node in self.g:
            d[node] = float('inf')
        d[source] = 0
        return d

    def build_min_heap(self, source):  # O(v) it travels all vertices at once and add them to the heap
        Q = []
        for node in self.g:
            if node == source:
                heappush(Q, (0, node, ()))
            else:
                heappush(Q, (float('inf'), node, ()))
        return Q

    def relaxation_and_update_minHeap(self, weight, vertice1, seen, mins, predecessor, path, Q,repetition):  # O(ElogV)
        for weight2, vertice2 in self.g.get(vertice1, ()):  # O(E) #for each vertex's adjacent vertexes
            repetition += 1
            if vertice2 in seen:  # if it is already in path, it is not needed to relax
                continue
            prev_cost = mins.get(vertice2, None)  # for getting the cost of the adjacent vertex
            next_cost = weight + weight2  # that is the cost of the path which is going to be checked
            if prev_cost is None or next_cost < prev_cost:  # if it is less than the cost of the adjacent vertex, it is needed to relax
                predecessor[vertice2] = vertice1  # if it is relaxed, it is needed to update predecessor
                mins[vertice2] = next_cost  # if it is relaxed, it is needed to update cost
                heappush(Q, (next_cost, vertice2, path))  # O(logV) if it is relaxed, it is needed to update min heap
                repetition += int(math.log2(self.node_amount))
        return repetition


    def dijkstra(self, source, destination):
        repetition = 0
        seen = set()  # for keeping the vertices which are already in path
        # O(v) for initializing single source like {1:0,2:inf,3:inf,4:inf,5:inf}
        mins = self.initialize_single_source(source)
        predecessor = {1: 1}  # for keeping the predecessor of each vertex
        Q = self.build_min_heap(source)  # O(v)

        while Q:  # O(v)
            weight, vertice1, path = heappop(Q)  # O(logv)
            if vertice1 not in seen:  # if it is not in path, it is needed to relax and update min heap
                seen.add(vertice1)  # it is added to seen
                path = (vertice1, path)  # it is added to path
                if vertice1 == destination:  # complete path
                    print("Predecessor List:", predecessor)
                    return (weight, path)  # return the cost and path
                # after then it was added to path, it is needed to relax for all adjacent vertexes
                # O(ElogV) for relaxing and updating min heap
                repetition = self.relaxation_and_update_minHeap(weight, vertice1, seen, mins, predecessor, path, Q,repetition)
        return float("inf"), None  # if there is no path, it returns infinity


    #heuristic function for a star algorithm
    def h_func(self, node, destination):
        return abs(node - destination)

    def a_star_algorithm(self, source, destination):
        #for deciding the source.
        distances = self.initialize_single_source(source) # O(v)
        self.a_star_parents = {source: None}  # initialize the parent mapping
        #unvisited and visited nodes are kept in two different sets
        open_set = set([source])
        closed_set = set()
        count = 0

        #while there is unvisited node
        while open_set: # O(v)
            count += 1
            #for finding the node with minimum cost f(n) = g(n) + h(n)
            node = min(open_set, key=lambda x: distances[x] + self.h_func(x, destination))
            #if the node is destination, it is returned
            if node == destination:
                path = [destination]
                #for finding the path
                while path[-1] != source: # O(v)
                    count += 1
                    #for finding the predecessor of the node
                    path.append(self.a_star_parents[path[-1]])

                path.reverse()
                print(count)
                return path, distances[destination]

            #if the node is not destination, it is removed from unvisited set and added to visited set
            open_set.remove(node)
            closed_set.add(node)

            #for each adjacent node of the node
            for weight,neighbor in self.g[node]:
                count += 1
                #if the node is already visited, it is not needed to be checked
                if neighbor in closed_set:
                    continue
                #new cost is calculated
                new_distance = distances[node] + weight

                #if the node is not in unvisited set, it is added to unvisited set
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    self.a_star_parents[neighbor] = node
                    open_set.add(neighbor)

        return [], float("inf")



if __name__ == '__main__':
    algorithm=int(input("Please enter the algorithm which you want to use: (A*:1, Dijkstra:2): "))
    if algorithm==1:
        try:
            N = int(input("Enter the number of nodes: "))
            S = int(input("Enter the source node: "))
            D = int(input("Enter the destination node: "))
            obj = Graph(N)
            graph = obj.graph()
            edges,edge_amount= obj.edges(N)
            teorik=int(edge_amount+N)
            reconst_path, total_cost = obj.a_star_algorithm(S, D)
            print("Total cost: ",total_cost)
            print("Shortest path: ",reconst_path)
            print(teorik)
        except ValueError:
            print("Please enter an integer!")
            exit()

    elif algorithm==2:
        try:
            N = int(input("Enter the number of nodes: "))
            S = int(input("Enter the source node: "))
            D = int(input("Enter the destination node: "))
            obj=Graph(N)
            graph=obj.graph()
            edges,edge_amount= obj.edges(N)
            theoretical=int(edge_amount*math.log2(N))
            out=(obj.dijkstra(S,D))
            data = {}
            data['cost']=out[0]
            reverse_path=[]
            while len(out)>1:
                reverse_path.append(out[0])
                out = out[1]
            reverse_path.remove(data['cost'])
            reverse_path.reverse()
            data['shortest_path']=reverse_path
            print("Shortest Path: ",data['shortest_path'])
            print("Total Cost", data['cost'])
            print("Theoretical Complexity: ",theoretical)
        except ValueError:
            print("Please enter an integer!")
            exit()
    else:
        print("Please enter a valid algorithm number!!")