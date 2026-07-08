import math
import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return 3*x**2 - 4*x + 5

print(f(3.0))

xs = np.arange(-5, 5, 0.25)
ys = f(xs)
#plt.plot(xs, ys)
#plt.show()


h = 0.000001
x = 2/3
print((f(x + h) - f(x))/h) # Thsi gives me slope


# les get more complex
a = 2.0
b = -3.0
c = 10.0
d = a*b + c
print(d)

h = 0.0001

# inputs
a = 2.0
b = -3.0
c = 10.0

d1 = a*b + c
c += h # derivetive of d wrt to c
d2 = a*b + c

print('d1', d1)
print('d2', d2)
print('slope', (d2 - d1)/h)

#Value Object

class Value:
  def __init__(self, data):
    self.data = data
  def __repr__(self):
    return f"Value(data={self.data})"
  
a = Value(2.0)
print(a)  


class Value:
  def __init__(self, data):
    self.data = data
  def __repr__(self):
    return f"Value(data={self.data})"
  
a = Value(2.0)
b = Value(3.0)
#print(a+b) # Gives me error because python doesnt know how to add two value objects


class Value:
  def __init__(self, data):
    self.data = data
  def __repr__(self):
    return f"Value(data={self.data})"
  def __add__(self, other):
    out = Value(self.data + other.data)
    return out
  
a = Value(2.0)
b = Value(3.0)
print(a+b) # works python internally calls a.__add__(b)

class Value:
  def __init__(self, data):
    self.data = data
  def __repr__(self):
    return f"Value(data={self.data})"
  def __add__(self, other):
    out = Value(self.data + other.data)
    return out
  def __mul__(self, other):
    out = Value(self.data * other.data)
    return out
  
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
print(a*b+c) 

# Now we have defined ourselves a Value object how add, mul works there
# What's missing here is conencting like what variable depends on what

class Value:
  def __init__(self, data, _children=(), _op='', label=''):
    self.data = data
    self.grad = 0.0 #derivative
    self._prev = set(_children)
    self._op = _op
    self.label = label
  def __repr__(self):
    return f"Value(data={self.data})"
  def __add__(self, other):
    out = Value(self.data + other.data, (self, other), '+')
    return out
  def __mul__(self, other):
    out = Value(self.data * other.data, (self, other), '*')
    return out
  
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

dot = draw_dot(L)
dot.render("computation_graph", view=True)