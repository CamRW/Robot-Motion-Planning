import numpy as np
from sklearn.neighbors import KDTree

rng = np.random.RandomState()

X = rng.random_sample((10,2))
print(X.shape)
tree = KDTree(X, leaf_size=2)
dist, ind = tree.query(X[:1],k=3)
print(ind)
print(dist)
print(X)







# class Kdtree:
#     def __init__(self):
#         pass

#     def generate_tree(self, node_list):
#         root = Node[node_list[0]]
#         root = node_list[0]
#         i = 0

#         for node in node_list:
#             if i%2 == 0:
#                 # CHECK X
#                 if node[0] <= root[0]:
#                     pass
#                     # ADD TO LEFT
#                 else:
#                     pass
#                     # ADD TO RIGHT
#             if i%2 == 1:
#                 # CHECK Y
#                 if node[1] <= root[1]:
#                     pass
#                     # ADD TO LEFT
#                 else:
#                     pass
#                     # ADD TO RIGHT

# class Node:
#    def __init__(self, data):
#       self.left = None
#       self.right = None
#       self.data = data
# # Insert Node
#    def insert(self, data):
#       if self.data:
#          if data < self.data:
#             if self.left is None:
#                self.left = Node(data)
#             else:
#                self.left.insert(data)
#          elif data > self.data:
#             if self.right is None:
#                self.right = Node(data)
#             else:
#                self.right.insert(data)
#       else:
#          self.data = data
# # Print the Tree
#    def PrintTree(self):
#       if self.left:
#          self.left.PrintTree()
#       print( self.data),
#       if self.right:
#          self.right.PrintTree()