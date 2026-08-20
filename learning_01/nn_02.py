import math
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# Backpropagation using the Chain Rule
#
# During the forward pass, mathematical operations (+, *, tanh, etc.) create
# a computation graph. Every node stores its value (data) and remembers which
# nodes were used to create it.
#
# During the backward pass, we traverse this graph from the output back to the
# input nodes and use the Chain Rule to compute gradients.
#
# Example:
#   e = a * b
#   d = e + c
#   L = d * f
#   dL/dL = 1
#   dL/dd = f
#   dL/df = d
#   dL/dc = dL/dd * dd/dc(= 1) = f -> this is the chain rule
#   dL/de = dL/dd * dd/de(= 1) = f
#   dL/da = dL/de(=f) * de/da(=b) = f*b
#   dL/db = f * a
#
# Each Value object stores:
#   data -> the numerical value at that node
#   grad -> how much the final output changes if this value changes slightly
#   if we want L to increase we need to increase leaf nodes in the direction of gradient say a.data += 0.01 * a.grad
#
# =============================================================================
# The neuron is generally built as below:
#
#   weighted_sum = x1*w1 + x2*w2 + bias
#   output = tanh(weighted_sum)
#
# Here:
#   - weights control the influence of each input
#   - bias shifts the weighted sum
#   - tanh is an activation function that transforms the weighted sum
# =============================================================================

# Activation function introduces non-linearity into the network.
#
# Without an activation function, a neural network would simply perform
# repeated additions and multiplications, which mathematically collapse
# into a single linear equation. In that case, even a deep network would
# behave like a single neuron.
#
# Activation functions (tanh, ReLU, sigmoid, etc.) allow the network to
# learn complex, non-linear relationships such as curves, boundaries,
# images, speech and language patterns.
#
# Typical neuron:
# weighted_sum = x1*w1 + x2*w2 + ... + bias
# output = activation(weighted_sum)
# =============================================================================
# NOTE:
# We accumulate gradients using += instead of =.
#
# Why?
# A node can contribute to the final output through multiple paths in the
# computation graph. According to multivariable calculus, the total derivative
# is the sum of the derivatives from every path.
#
# Example:
#
#       a
#      / \
#     *   +
#      \ /
#       L
#
# Here 'a' influences L through two different paths, so:
#
#   dL/da = (contribution from path 1) + (contribution from path 2)
#
# Therefore we accumulate:
#
#   self.grad += local_derivative * out.grad
#
# instead of overwriting:
#
#   self.grad = local_derivative * out.grad

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        # gradient of final output with respect to this node
        # Initially unknown, so set to 0
        self.grad = 0.0
        # every node should have a backward function
        # leaf nodes simply do nothing
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):

        # Forward pass
        # Compute new value
        other = other if isinstance(other, Value) else Value(other) 
        # To accomodate a + 1 where a is object of Value but 1 is an integer
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            # derivative of (a+b) wrt a = 1
            self.grad += 1.0 * out.grad
            # derivative of (a+b) wrt b = 1
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out
    def __mul__(self, other):

        other = other if isinstance(other, Value) else Value(other) 
        # Forward pass
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            # d(a*b)/da = b
            self.grad += other.data * out.grad
            # d(a*b)/db = a
            other.grad += self.data * out.grad

        out._backward = _backward

        return out
    def tanh(self):

        x = self.data
        # Forward pass
        t = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(t, (self,), 'tanh')
        def _backward():
            # derivative of tanh(x)
            # = 1-tanh²(x)
            self.grad += (1-t**2) * out.grad

        out._backward = _backward
        return out
  
    def backward(self):
        # Stores nodes in topological (forward computation) order
        topo = []
        # Keeps track of visited nodes so we don't process a node twice
        visited = set()

        # Depth First Search (DFS) to build topological ordering
        def build_topo(v):
            if v not in visited:
                visited.add(v)

                # Visit all nodes that were used to create this node
                for child in v._prev:
                    build_topo(child)

                # Add the current node after its children
                # This ensures children appear before parents
                topo.append(v)

        # Start DFS from the output node
        build_topo(self)

        # Gradient of output with respect to itself is always 1
        # Example: dL/dL = 1
        self.grad = 1.0

        # Traverse the graph in reverse topological order
        # (i.e., from output back to the leaf nodes)
        for node in reversed(topo):
            node._backward()
  
a = Value(2.0, label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b; e.label='e'
d = e+c; d.label='d'
f = Value(-2.0, label='f')
L = d * f; L.label = 'L'
print(d) 
print(d._prev)
print(d._op)

from graphviz import Digraph
def trace(root):
  # builds a set of all nodes and edges in a graph
  nodes, edges = set(), set()
  def build(v):
    if v not in nodes:
      nodes.add(v)
      for child in v._prev:
        edges.add((child, v))
        build(child)
  build(root)
  return nodes, edges

def draw_dot(root):
  dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
  
  nodes, edges = trace(root)
  for n in nodes:
    uid = str(id(n))
    # for any value in the graph, create a rectangular ('record') node for it
    dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
    if n._op:
      # if this value is a result of some operation, create an op node for it
      dot.node(name = uid + n._op, label = n._op)
      # and connect this node to it
      dot.edge(uid + n._op, uid)

  for n1, n2 in edges:
    # connect n1 to the op node of n2
    dot.edge(str(id(n1)), str(id(n2)) + n2._op)

  return dot

#dot = draw_dot(L)
#dot.render("computation_graph", view=True)


# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

o.backward()
dot1 = draw_dot(o)
dot1.render("computation_graph", view=True)