# import pynauty as pn

# g1_adj = {0: [1, 2, 3, 4], 1: [0], 2: [0], 3: [0], 4: [0]}

# g1 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,2,3,4))])
# g2 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,2,3,4))])

# print("With same color ordering:")
# print(g1)
# print(g2)
# assert pn.isomorphic(g1, g2)

# g1 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,2,3,4))])
# g2 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((1,2,3,4)), set((0,))])

# print("With different color ordering:")
# print(g1)
# print(g2)
# assert pn.isomorphic(g1, g2), "Pynauty says not isomorphic when order of colors changes!!!"

# import pynauty as pn

# g1_adj = {0: [1], 1: [0]}

# g1 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,))])
# g2 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,))])

# print("With same color ordering:")
# print(g1)
# print(g2)
# assert pn.isomorphic(g1, g2)

# g1 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,))])
# g2 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((1,)), set((0,))])

# print("With different color ordering:")
# print(g1)
# print(g2)
# assert pn.isomorphic(g1, g2), "Pynauty says not isomorphic when order of colors changes!!!"


import pynauty as pn

g1_adj = {0: [1, 2], 1: [0], 2: [0]}

g1 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,2))])
g2 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,2))])

print("With same color ordering:")
print(g1)
print(g2)
assert pn.isomorphic(g1, g2)

g1 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((0,)), set((1,2))])
g2 = pn.Graph(number_of_vertices=5, adjacency_dict=g1_adj, vertex_coloring=[set((1,2)), set((0,))])

print("With different color ordering:")
print(g1)
print(g2)
assert pn.isomorphic(g1, g2), "Pynauty says not isomorphic when order of colors changes!!!"

