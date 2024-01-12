# Infermo🔥

### Tensors and dynamic Neural Networks in pure Mojo

Infermo is a Mojo library that provides two high-level features:
- Tensor computation
- Automatic Differentiation

##
- ### NOT READY FOR USE YET - UNDER CONSTRUCTION - Updates are on the way :)
##

Mojo currently operates on CPU only. GPU support will come soon! 

Infermo is still a Proof-of-Concept, if you encounter any bugs, feel free to create an issue or a PR. Thank you for your contribution. 😊

## A tiny Example
```python
# Lets's build a simple neural network that learns to approximate sin(15x)
# Dynamic Computation Graph (with conditional model architecture!!!) (static execution is also possible)

fn main() raises:

    # init params
    let W1 = Tensor(shape(1,64)).randhe().requires_grad()
    let W2 = Tensor(shape(64,64)).randhe().requires_grad()
    let W3 = Tensor(shape(64,1)).randhe().requires_grad()
    let W_opt = Tensor(shape(64,64)).randhe().requires_grad()
    let b1 = Tensor(shape(64)).randhe().requires_grad()
    let b2 = Tensor(shape(64)).randhe().requires_grad()
    let b3 = Tensor(shape(1)).randhe().requires_grad()
    let b_opt = Tensor(shape(64)).randhe().requires_grad()

    var avg_loss = Float32(0.0)
    let every = 1000
    let num_epochs = 20000

    # training
    for epoch in range(1,num_epochs+1):

        # set input and true values
        let input = Tensor(shape(32,1)).randu(0,1).dynamic()
        let true_vals = sin(15.0 * input)

        # define model architecture
        var x = relu(input @ W1 + b1)
        x = relu(x @ W2 + b2)
        if epoch < 100:
            x = relu(x @ W_opt + b_opt) 
        x = x @ W3 + b3
        let loss = mse(x,true_vals).forward()

        # print progress
        avg_loss += loss[0]
        if epoch%every == 0:
            print("Epoch:",epoch," Avg Loss: ",avg_loss/every)
            avg_loss = 0.0   
       
        # # compute gradients and optimize
        loss.backward()
        loss.optimize(0.01,"sgd")

        # clear graph
        loss.clear() 
        input.free()
```

## Unique Feature
- Memory Sharing
- Gradient Checkpointing
- Choose between static and dynamic graph execution


## Coming soon...
- More optimized memory management
- GPU support
- More operators, activiations, optimizers

#### We are focusing on building the engine right before adding more features. Stay tuned for more updates!

