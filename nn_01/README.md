# Neural Networks from Scratch: Build a Tiny Automatic Differentiation Engine (Micrograd Guide)

Welcome to the foundational workspace for learning how neural networks work at their deepest, most fundamental level. This directory is heavily inspired by and follows Andrej Karpathy's famous **Micrograd** tutorial, where we build a tiny automatic differentiation (Autograd) engine from scratch.

By building our own math engine (`Value` class) and neural network components (`Neuron`, `Layer`, `MLP`) using pure Python, we pull back the curtain on "magic" deep learning libraries like PyTorch. This guide is written for beginners—no prior machine learning experience or advanced calculus knowledge is assumed!

---

## 1. Overview

The purpose of this folder is to demystify the inner workings of deep learning. High-level frameworks like PyTorch, TensorFlow, and JAX make it incredibly easy to train complex neural networks in just a few lines of code. However, relying on these libraries without understanding what happens under the hood can lead to a shallow understanding and difficulty debugging training failures.

This folder serves as an educational sandbox where you will:
- **Build a tiny autograd engine** from scratch that supports addition, multiplication, exponentiation, negation, subtraction, division, and the hyperbolic tangent (`tanh`) activation function.
- **Trace and visualize mathematical operations** as computation graphs using Graphviz.
- **Assemble a complete neural network stack** (Neurons $\rightarrow$ Layers $\rightarrow$ Multi-Layer Perceptrons).
- **Implement backpropagation** using the chain rule to automatically compute gradients.
- **Train a neural network** using gradient descent on a binary classification dataset.

**The Ultimate Goal:** Gain a crystalline, intuitive grasp of how data flows forward, how gradients flow backward, and how parameters update before graduating to PyTorch.

---

## 2. Learning Objectives

By exploring the scripts in this folder, you will master the following fundamental concepts of deep learning:

| Concept | High-Level Intuition | Why It Matters in Neural Networks |
| :--- | :--- | :--- |
| **Derivatives** | Measures how much the output of a function changes when its input is tweaked by a tiny amount. | Helps us determine whether to increase or decrease a parameter to reduce error. |
| **Gradients** | A vector (list) of derivatives showing the direction of steepest increase for a multi-variable function. | Represents the collective "push" required for every weight and bias in the network. |
| **Numerical Differentiation** | Approximating derivatives using a tiny step $h$: $\frac{f(x+h) - f(x)}{h}$. | Ideal for explaining the concept and verifying that our automated calculations are exact. |
| **Computation Graphs** | A flowchart where nodes are numbers (values) and edges represent mathematical operations. | Essential for systematically tracking dependencies and calculating derivatives. |
| **Forward Pass** | Feeding inputs through the network to calculate intermediate results and final predictions. | Generates predictions and measures how far off the network is from the truth (loss). |
| **Backpropagation** | Walking backward through the computation graph to calculate the derivative of the output with respect to every input. | The algorithmic engine that automatically finds how to fix errors across all parameters. |
| **Chain Rule** | A rule from calculus that allows us to find overall rates of change by multiplying local rates of change. | The mathematical glue that makes backpropagation in deep, stacked networks possible. |
| **Automatic Differentiation** | The software technique of dynamically tracking operations to calculate exact derivatives. | Spares human engineers from manually writing derivative formulas for complex models. |
| **Gradient Descent** | An optimization algorithm that shifts parameters in the opposite direction of the gradient. | The fundamental mechanism by which a model learns from its mistakes and improves. |
| **Neurons** | The basic unit of a network that computes a weighted sum of inputs, adds a bias, and applies an activation. | Mimics biological brain cells by acting as a flexible detector of input features. |
| **Layers** | A collection of parallel neurons that process the same input stream independently. | Increases the capacity of the model to detect multiple distinct patterns at once. |
| **Multi-Layer Perceptrons (MLPs)**| Stacking multiple layers sequentially so the outputs of one layer become the inputs of the next. | Enables the network to learn deep, hierarchical features (e.g., edges $\rightarrow$ shapes $\rightarrow$ objects). |
| **Loss Functions** | A mathematical formula that outputs a single number measuring "how wrong" the network's predictions are. | Serves as the guiding compass (or score) that the network tries to minimize. |
| **Training Loop** | An iterative loop of: Forward Pass $\rightarrow$ Reset Gradients $\rightarrow$ Backpropagation $\rightarrow$ Parameter Update. | The repetitive process that refines the network's parameters over time. |

---

## 3. File-by-File Explanation

Rather than looking at a completed library, this directory contains a step-by-step learning progression. Here is how each file builds on the previous one:

### 🚀 Step 1: `nn_01.py` — The Foundation of Calculus & The Value Object
- **Why it exists:** To introduce the concept of derivatives numerically before transitioning to an object-oriented representation of variables.
- **What new concept it introduces:**
  - **Numerical Differentiation:** Using a tiny value $h = 0.000001$ to calculate the slope of $f(x) = 3x^2 - 4x + 5$ at $x = 2/3$.
  - **The `Value` Class:** Introducing a custom class to wrap raw data and overload operators (`__add__`, `__mul__`) so we can write equations like `a * b + c` using custom objects.
  - **Graph Tracking:** Updating `Value` to track its children (`_prev`) and the operation (`_op`) that created it.
  - **Visualization:** Using Graphviz (`trace` and `draw_dot`) to render the full computation graph as an SVG image.
- **How it builds:** It sets up the raw tracking structure. We can now build deep math expressions, but we can't perform backpropagation yet.

### 🧠 Step 2: `nn_02_gradient.py` — Implementing the Chain Rule & The First Neuron
- **Why it exists:** To teach our custom `Value` class how to automatically backpropagate gradients using the calculus chain rule.
- **What new concept it introduces:**
  - **The Gradient (`grad`):** Every `Value` now stores a `.grad` (initialized to $0.0$), representing the rate of change of the final output with respect to this value.
  - **Local Backpropagation (`_backward`):** Every mathematical operation defines a local backward function. For example, for $out = a \times b$, the local derivatives are $\frac{\partial out}{\partial a} = b$ and $\frac{\partial out}{\partial b} = a$.
  - **Topological Sorting:** Using Depth-First Search (DFS) to order the computation graph so we can propagate gradients in the correct reverse sequence (from output back to inputs) by calling `.backward()`.
  - **Non-Linear Activations (`tanh`):** Implementing the hyperbolic tangent activation function to squish values between $-1$ and $+1$ and explain non-linearity.
  - **Accumulating Gradients (`+=`):** Changing the gradient assignment from `=` to `+=` to handle nodes that are reused multiple times in a graph.
- **How it builds upon previous files:** It upgrades the passive tracking tree from `nn_01.py` into an active autograd engine. We build a single, manual neuron (`x1*w1 + x2*w2 + b`) and successfully run backpropagation on it.

### 🛠️ Step 3: `nn_03.py` — Building a Robust Mathematical Toolkit
- **Why it exists:** To expand the mathematical capabilities of our `Value` class so it can support any mathematical formula.
- **What new concept it introduces:**
  - **Exponentiation (`__pow__`):** Allowing our class to handle exponents ($x^y$), which is critical for division and squared errors.
  - **Right-Handed Operations (`__radd__`, `__rmul__`):** Allowing operations like `2 * a` and `1 + a` where a primitive number is on the left.
  - **Division, Subtraction, and Negation (`__truediv__`, `__sub__`, `__neg__`):** Implementing these by combining existing primitive operations (e.g., $a / b = a \times b^{-1}$ and $a - b = a + (-b)$).
  - **Exponential (`exp()`):** Implementing $e^x$ and using it to build `tanh` manually: $\tanh(x) = \frac{e^{2x} - 1}{e^{2x} + 1}$.
- **How it builds upon previous files:** It completes the mathematical engine. Instead of using built-in shortcuts, we can now represent any complex function (including loss functions and custom activation equations) as a graph of primitive operations.

### 🔬 Step 4: `nn_04_torch.py` — Verification with Industry Standards
- **Why it exists:** To validate that our tiny custom `Value` class works exactly like PyTorch, the gold standard of deep learning.
- **What new concept it introduces:**
  - **PyTorch Tensors (`torch.Tensor`):** Instantiating scalar tensors in PyTorch.
  - **Gradient Enablement (`requires_grad=True`):** Telling PyTorch to build a computation graph.
  - **Double Precision (`.double()`):** Matching Python's 64-bit float precision.
- **How it builds upon previous files:** It does not modify our engine, but it verifies our math. By running the same neuron calculation through PyTorch and printing out the gradients, we confirm that our engine's gradient calculations match PyTorch's to the last decimal place!

### 🏋️ Step 5: `nn_05_neuralnet.py` — Constructing & Training an MLP
- **Why it exists:** To scale our scalar `Value` class up into structural neural network components and execute a complete training loop.
- **What new concept it introduces:**
  - **`Neuron` Class:** Encapsulates random weight initializations, bias, and the forward calculation $y = \tanh(\sum w_i x_i + b)$.
  - **`Layer` Class:** Manages multiple neurons in parallel.
  - **`MLP` Class:** Stacks layers sequentially to create a multi-layer network.
  - **Zero-Grading (`p.grad = 0.0`):** Resetting gradients before every backward pass to prevent gradients from accumulating across training iterations.
  - **Loss Computation & Parameter Updates:** Computing Sum Squared Error and adjusting parameters in-place using Gradient Descent: `p.data += -learning_rate * p.grad`.
- **How it builds upon previous files:** This is the culmination of the entire folder. It takes the mathematical `Value` class from `nn_03.py` and organizes it into a modular, object-oriented neural network capable of learning a dataset from scratch.

---

## 4. Theory Section

### 📈 What is a derivative?
Imagine you are driving a car and you tap the gas pedal. A **derivative** measures how much your speed increases with a tiny tap of the pedal. 
Mathematically, for a function $y = f(x)$, if we increase $x$ by a microscopic step $h$, the output changes to $f(x+h)$. The derivative is the slope of this change:

$$\text{Slope} = \frac{f(x + h) - f(x)}{h}$$

If the derivative is **positive**, increasing $x$ increases $y$. If it is **negative**, increasing $x$ decreases $y$. If it is **zero**, we are at a flat spot (a peak or a valley).

---

### ⛰️ What is a gradient?
While a derivative measures the slope of a single variable, a **gradient** is simply the collection of derivatives for a function with *many* variables. 

If you are standing on a foggy mountain, the gradient is a vector that points in the direction of the steepest slope upwards. In machine learning, our loss function is a massive landscape of errors. The gradient points in the direction that increases the error fastest. Therefore, to minimize the error, we do the opposite: we take steps in the **negative gradient** direction to walk downhill.

---

### ⚖️ Difference between derivative and gradient
- **Derivative:** Applies to single-input functions ($y = f(x)$). It is a single number.
- **Gradient:** Applies to multi-input functions ($y = f(w_1, w_2, w_3, \dots)$). It is a vector (a list of numbers), where each number is the derivative with respect to one specific input (called a partial derivative).

---

### 🕸️ What is a computation graph?
A **computation graph** is a visual diagram of mathematical operations. 
- **Nodes** represent numerical values (inputs, weights, biases, and intermediate outputs).
- **Edges (lines)** show how numbers flow into operations (like $+$, $*$, or $\tanh$).

By organizing mathematical expressions as a graph, a computer can easily trace how any given parameter impacts the final output.

---

### ➡️ What is a forward pass?
The **forward pass** is the process of feeding inputs into the network and propagating them forward (from left to right) through all operations to calculate the final prediction and the loss.

```text
Inputs (x) ──> [ Layer 1 ] ──> [ Layer 2 ] ──> Prediction (ypred) ──> Compute Loss (L)
```

---

### ⬅️ What is backpropagation?
**Backpropagation** (backward propagation of errors) is the process of traversing the computation graph backward (from right to left) to compute the gradient of the loss with respect to every single parameter in the network.

We start at the final loss $L$. The rate of change of $L$ with respect to itself is always $1.0$ ($\frac{\partial L}{\partial L} = 1.0$). We then step backward through each operation, using the **Chain Rule** to compute gradients for earlier nodes.

---

### 🔗 What is the chain rule?
The **chain rule** is a calculus formula for computing the derivative of composite functions. 

If $A$ influences $B$, and $B$ influences $C$, then the rate of change of $C$ with respect to $A$ is the product of their local rates of change:

$$\frac{\partial C}{\partial A} = \frac{\partial C}{\partial B} \times \frac{\partial B}{\partial A}$$

#### Intuitive Example:
- If a gears system is set up such that turning Gear A 1 time turns Gear B 3 times ($\frac{\partial B}{\partial A} = 3$), and turning Gear B 1 time turns Gear C 2 times ($\frac{\partial C}{\partial B} = 2$).
- Turning Gear A 1 time will turn Gear C: $2 \times 3 = 6$ times ($\frac{\partial C}{\partial A} = 6$).

---

### 🤖 Why does automatic differentiation exist?
For simple equations, we can calculate derivatives manually on paper. However, modern neural networks have millions or billions of parameters. Writing out derivative equations for billions of variables is humanly impossible. 

**Automatic Differentiation** automates this. By writing code where every basic operation (addition, multiplication) automatically knows how to compute its own *local* derivative, we can compute exact gradients for mathematical operations of arbitrary complexity with a single function call (`.backward()`).

---

### ➕ Why gradients are accumulated using `+=` instead of `=`
In many computation graphs, a single variable can influence the final output through multiple paths. 

```text
       [ a ]
      /     \
   [ e ]   [ f ]
      \     /
       [ L ]
```

Here, `a` influences `L` through both `e` and `f`. According to multivariable calculus, the total derivative of `L` with respect to `a` is the sum of the derivatives from all paths:

$$\frac{\partial L}{\partial a} = \left( \text{path 1: } \frac{\partial L}{\partial e} \frac{\partial e}{\partial a} \right) + \left( \text{path 2: } \frac{\partial L}{\partial f} \frac{\partial f}{\partial a} \right)$$

If we used simple assignment (`self.grad = ...`), the gradient from the second path would overwrite the gradient from the first path. Using **`+=`** ensures we correctly sum up the contributions of all paths.

---

### 🧠 What is a neuron?
A **neuron** is the fundamental computational building block of a neural network. It takes several input values, weighs them according to their importance, adds an offset value (bias), and squeezes the result through an activation function to introduce non-linearity.

```text
       Inputs       Weights
         x₁ --------> w₁ -----\
                              |
         x₂ --------> w₂ ----> [ Σ ] -------> [ tanh ] -------> Output (y)
                              |  (Weighted      (Activation
         x₃ --------> w₃ -----/   Sum + Bias)    Function)
                                 ^
                                 |
         b (Bias) ---------------/
```

---

### 🎛️ Why neurons have weights and bias
- **Weights ($w$):** Control the strength of the input signals. If a weight is large and positive, that input strongly excites the neuron. If it is negative, it inhibits it. If it is zero, the input is ignored.
- **Bias ($b$):** Represents how easy it is for the neuron to "fire" (produce a positive activation). It shifts the activation function left or right. Without a bias, a neuron's activation would always be centered at zero, severely limiting its flexibility.

---

### ⚡ Why we use an activation function (tanh in this project)
Without an activation function, every operation in a neural network is just a linear equation (additions and multiplications). No matter how many layers you stack, a sequence of linear equations mathematically collapses into a single linear equation:

$$y = w_2(w_1 x + b_1) + b_2 \implies y = W x + B$$

Therefore, a deep network without activations behaves exactly like a single neuron! 
An **activation function** like $\tanh$ introduces a non-linear curve. This allows the network to learn complex, curved boundaries, enabling it to model complex patterns like shapes, audio frequencies, and language syntax.

---

### 🥞 What is a layer?
A **layer** is a collection of parallel neurons that process the same input stream. Each neuron in the layer has its own unique, independent weights and biases, allowing them to detect different features from the same input simultaneously.

```text
                     +---------------------------------------+
                     |                LAYER                  |
                     |                                       |
                     |     +----------+                      |
                     |  /->| Neuron 1 |──> Output 1          |
                     |  |  +----------+                      |
                     |  |                                    |
          Inputs ────+────>+----------+                      |
            [x]      |  |─>| Neuron 2 |──> Output 2          |
                     |  |  +----------+                      |
                     |  |                                    |
                     |  \─>| Neuron 3 |──> Output 3          |
                     |     +----------+                      |
                     +---------------------------------------+
```

---

### 🕸️ What is a Multi-Layer Perceptron (MLP)?
A **Multi-Layer Perceptron (MLP)** is a fully connected neural network made of multiple layers stacked one after another.

The outputs of Layer 1 (Hidden Layer) become the inputs of Layer 2, whose outputs feed into Layer 3 (Output Layer), and so on.

```text
      INPUTS          LAYER 1          LAYER 2          LAYER 3
     (3 units)      (4 Neurons)      (4 Neurons)       (1 Neuron)

       (x₁) ───────> [Neuron] ───────> [Neuron] ───────\
             \     /          \     /          \       v
              \   /            \   /            \
       (x₂) ────+──> [Neuron] ──+──> [Neuron] ────+──> [Neuron] ───> Output (y)
              /   \            /   \            /
             /     \          /     \          /
       (x₃) ───────> [Neuron] ───────> [Neuron] ───────/
                    /                /
             [Neuron] ───────> [Neuron]
```

---

### 🎯 What is a loss function?
A **loss function** is a mathematical formula that acts as a referee or scorekeeper. It takes the network's predictions and compares them to the true targets, outputting a single number representing the "error". A loss of $0.0$ represents perfect predictions.

---

### 📐 What is Sum Squared Error?
In this project, we use Sum Squared Error (SSE) as our loss function:

$$\text{Loss} = \sum (\text{prediction} - \text{target})^2$$

1. **Subtraction ($\text{prediction} - \text{target}$):** Computes the error.
2. **Squaring ($^2$):** Ensures that all errors are positive (so negative and positive errors don't cancel each other out) and heavily penalizes larger errors.
3. **Summation ($\sum$):** Aggregates all errors across the dataset into a single scalar value.

---

### 🏂 What is gradient descent?
**Gradient Descent** is an optimization algorithm used to minimize the loss. 

The gradient tells us which way to walk to **increase** the loss fastest. Therefore, we update our parameters by moving in the **opposite** direction. We subtract a small fraction of the gradient from each weight and bias:

$$\theta_{\text{new}} = \theta_{\text{old}} - \text{learning\_rate} \times \text{gradient}$$

It is like walking down a mountain in a heavy fog by feeling the slope of the ground beneath your feet and taking small steps downhill.

---

### 👟 What is a learning rate?
The **learning rate** is a small multiplier (e.g., $0.1$ or $0.01$) that controls the size of the steps we take during gradient descent.
- **Too large:** We might overshoot the valley and cause the training to fail.
- **Too small:** The network will take an eternity to learn.

---

### 🔍 Parameter, Weight, Bias, Activation, Output, and Prediction

| Term | Technical Meaning | Layman Analogy |
| :--- | :--- | :--- |
| **Parameter** | Any learnable variable in the network that is updated during training. | The knobs and sliders we adjust to tune a machine. |
| **Weight** | A parameter that scales an input value, determining its importance. | The volume knob for a specific input channel. |
| **Bias** | A parameter added to the weighted sum to shift the activation function. | The threshold setting how easy it is to trigger the neuron. |
| **Activation**| The intermediate weighted sum value *before* applying the activation function. | The raw electrical signal inside a neuron before it fires. |
| **Output** | The value produced by a single neuron *after* applying the activation function. | The signal that is actually fired and transmitted to the next layer. |
| **Prediction**| The final output of the entire multi-layer network. | The model's final guess for a given input. |

---

## 5. Code Architecture

Our neural network implementation is structured as a clear, object-oriented hierarchy:

```text
  [ Value ]            <-- Basic computational unit (holds data, grad, _backward)
      ↓
  [ Neuron ]           <-- Consists of multiple weight Values and one bias Value
      ↓
  [ Layer ]            <-- Stacks multiple Neurons to process inputs in parallel
      ↓
  [ MLP ]              <-- Cascades multiple Layers sequentially (Input -> Hidden -> Output)
      ↓
  [ Training Loop ]    <-- Feeds data to MLP, computes loss, backpropagates, updates Values
```

### Class Responsibilities:

1. **`Value` (`nn_03.py`)**
   - Wraps a single scalar float (`self.data`).
   - Keeps track of gradients (`self.grad`).
   - Stores the mathematical history (`self._prev`, `self._op`).
   - Defines mathematical operations using operator overloading.
   - Triggers the global backward pass (`self.backward()`).

2. **`Neuron` (`nn_05_neuralnet.py`)**
   - Initializes a list of random weight `Value` objects (`self.w`) and a bias `Value` (`self.b`).
   - Computes $y = \tanh(\sum w_i x_i + b)$.
   - Returns all parameters of the neuron (`self.parameters()`).

3. **`Layer` (`nn_05_neuralnet.py`)**
   - Stacks multiple `Neuron` objects in parallel.
   - Passes the input vector $x$ to each neuron and returns their outputs as a list.
   - Collects all parameters from all neurons in the layer.

4. **`MLP` (`nn_05_neuralnet.py`)**
   - Stacks multiple `Layer` objects sequentially.
   - Feeds inputs through the layers in sequence (Forward Pass).
   - Collects all parameters across all layers into a single flat list.

---

## 6. Training Flow

The complete process of training our neural network operates in an iterative loop:

```text
+--------------------------------------------------------------------------+
|                            THE TRAINING LOOP                             |
+--------------------------------------------------------------------------+
|                                                                          |
|       +-----------------+                                                |
|  ==>  | 1. Input Data   | (Features: xs)                                 |
|       +-----------------+                                                |
|                |                                                         |
|                v                                                         |
|       +-----------------+                                                |
|       | 2. Forward Pass | (Pass inputs through MLP layers)               |
|       +-----------------+                                                |
|                |                                                         |
|                v                                                         |
|       +-----------------+                                                |
|       | 3. Prediction   | (Model outputs: ypred)                         |
|       +-----------------+                                                |
|                |                                                         |
|                v                                                         |
|       +-----------------+                                                |
|       | 4. Compute Loss | (Compare predictions with targets: ys)         |
|       +-----------------+                                                |
|                |                                                         |
|                v                                                         |
|       +-----------------+                                                |
|       | 5. Zero Gradients| (Set p.grad = 0.0 for all parameters)         |
|       +-----------------+                                                |
|                |                                                         |
|                v                                                         |
|       +-----------------+                                                |
|       | 6. Backprop     | (Call loss.backward() to compute gradients)    |
|       +-----------------+                                                |
|                |                                                         |
|                v                                                         |
|       +-----------------+                                                |
|       | 7. Grad Descent | (Update: p.data -= learning_rate * p.grad)     |
|       +-----------------+                                                |
|                |                                                         |
|                +=========( Repeat for next epoch )=======================+
|                                                                          |
+--------------------------------------------------------------------------+
```

### Trace of a Single Step:
1. **Input:** We feed the input features `xs` (e.g., `[2.0, 3.0, -1.0]`) into the network.
2. **Forward Pass:** The MLP processes the data layer-by-layer:
   $$\text{Input } x \rightarrow \text{Layer 1} \rightarrow \text{Layer 2} \rightarrow \text{Layer 3} \rightarrow \text{Prediction } \hat{y}$$
3. **Loss Calculation:** We evaluate how good our predictions are compared to targets `ys` using the Sum Squared Error loss:
   $$\text{Loss} = \sum (\hat{y} - y)^2$$
4. **Zeroing Gradients:** Crucially, we reset all parameter gradients to `0.0`. If we forget this, gradients from the previous iteration will be added to the new ones, ruining the calculations.
5. **Backpropagation:** We call `loss.backward()`. This starts at the Loss node and traverses the entire graph in reverse, populating the `.grad` field for every single weight and bias in the network.
6. **Gradient Descent:** We update all parameters in-place:
   $$\text{parameter.data} \mathrel{-}= 0.1 \times \text{parameter.grad}$$
7. **Repeat:** We run this loop for multiple iterations (e.g., 20 epochs) until the total loss approaches $0.0$.

---

## 7. Important Python Concepts Used

Our implementation leverages several key Python features:

### 🐍 Special Magic Methods
- **`__call__`**
  - Allows an object to be called like a regular function.
  - Used in `Neuron`, `Layer`, and `MLP` so we can write `mlp(x)` instead of `mlp.forward(x)`.
- **`__repr__`**
  - Defines how an object is represented as a string when printed.
  - Instead of printing `<__main__.Value object at 0x...>`, printing a `Value` displays `Value(data=2.0)`.

### ⚡ Operator Overloading
Methods like `__add__`, `__mul__`, and `__pow__` allow us to intercept basic arithmetic operators and define custom behaviors for our objects:
```python
# When we write:
d = a * b + c

# Python translates it to:
d = a.__mul__(b).__add__(c)
```
This lets us write natural-looking mathematical formulas while building a computation graph behind the scenes.

### 🧩 Core Language Utilities
- **List Comprehensions:** A compact way to create lists in Python.
  ```python
  self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
  ```
- **`zip()`:** Iterates over multiple lists in parallel, pairing matching elements.
  ```python
  # Pairs weights with input values: [(w1, x1), (w2, x2), ...]
  zip(self.w, x)
  ```
- **`sum(generator, start)`:** Computers the sum of items. By default, `sum()` starts with the integer `0`. Since we cannot add an integer `0` to a custom `Value` object, we provide the bias `self.b` as the starting value:
  ```python
  act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
  ```
- **`*args`:** Used to pass a variable number of positional arguments to a function as a tuple. (Though not used directly in our scripts, it is the standard way mathematical libraries accept arbitrary dimensions).
- **`lambda`:** Small, anonymous, inline functions. We use a lambda to define a default "do nothing" backward pass for our leaf nodes:
  ```python
  self._backward = lambda: None
  ```
- **`isinstance`:** Checks if an object is of a specific type. We use this to support operations between `Value` objects and regular Python floats/integers:
  ```python
  other = other if isinstance(other, Value) else Value(other)
  ```

---

## 8. Mathematical Equations

Here are the mathematical formulas implemented in this directory:

### 1. Weighted Sum (Pre-activation)
$$\text{act} = \sum_{i=1}^{n} w_i x_i + b = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b$$

### 2. Hyperbolic Tangent ($\tanh$) Activation
$$\text{output} = \tanh(\text{act}) = \frac{e^{2 \cdot \text{act}} - 1}{e^{2 \cdot \text{act}} + 1}$$

### 3. Sum Squared Error Loss
$$\text{Loss} = \sum_{j} (\hat{y}_j - y_j)^2$$

### 4. Gradient Descent Parameter Update
$$\theta \leftarrow \theta - \eta \times \frac{\partial \text{Loss}}{\partial \theta}$$
*(Where $\theta$ is any weight or bias, and $\eta$ is the learning rate)*

### 5. Local Derivative Examples (Chain Rule)
- **Addition ($z = x + y$):**
  $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \times 1.0, \quad \frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \times 1.0$$
- **Multiplication ($z = x \times y$):**
  $$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial z} \times y, \quad \frac{\partial L}{\partial y} = \frac{\partial L}{\partial z} \times x$$
- **Activation ($o = \tanh(n)$):**
  $$\frac{\partial L}{\partial n} = \frac{\partial L}{\partial o} \times (1 - o^2)$$

---

## 9. Key Takeaways

By building and running the scripts in this folder, you have learned that:
1. **Neural Networks are just large mathematical formulas:** There is no magic. Underneath the fancy terminology, a neural network is simply a sequence of additions, multiplications, and activations.
2. **Backpropagation is just the chain rule in code:** By storing mathematical operations in a directed graph, we can automatically compute how any weight affects our final error by working backward.
3. **Activation functions are mandatory:** Without non-linear functions like $\tanh$, stacking multiple layers is mathematically useless, as they collapse into a single straight line.
4. **Learning is just walking downhill:** By calculating the gradient of the loss, we know exactly which way to nudge every knob (weights and biases) in the network to make the loss smaller.

---

## 10. Next Steps

Now that you have built a neural network from scratch, understood how computation graphs work, and verified your math against PyTorch, you are ready to transition!

In the next section, we will:
- Stop working with single scalars (`Value` class).
- Start working with multi-dimensional arrays called **Tensors**.
- Use **PyTorch's built-in layers and automatic differentiation engine**, which does exactly what we built here, but optimized in C++ to run blazingly fast on GPUs!

---

## 🛠️ Setup and How to Run

To run the scripts in this directory and visualize your own computation graphs:

### 1. Install Dependencies
You need `torch`, `numpy`, and `matplotlib`. To visualize graphs, you also need Graphviz:
```bash
# Activate your environment
source .venv/bin/activate

# Install Python packages
pip install torch numpy matplotlib graphviz
```

### 2. Install Graphviz System Binary (Required for rendering graphs)
- **macOS:** `brew install graphviz`
- **Ubuntu/Debian:** `sudo apt-get install graphviz`
- **Windows:** Download and install from [Graphviz Downloads](https://graphviz.org/download/).

### 3. Execute any script
```bash
python nn_01.py
python nn_02_gradient.py
python nn_03.py
python nn_04_torch.py
python nn_05_neuralnet.py
```
This will run the calculations, print out intermediate results, and generate a `computation_graph.svg` file showing your neural network's visual graph!