from layers import EmbedPosition, TransformerBlock, LinearLayer, Softmax
from numba.types import List
from numba.experimental import jitclass

transformer_block_type = TransformerBlock.class_type.instance_type
@jitclass([
    ('embedding', EmbedPosition.class_type.instance_type),
    ('transformer_blocks', List(transformer_block_type)),
    ('lm_head', LinearLayer.class_type.instance_type),
    ('out_softmax', Softmax.class_type.instance_type)
])
class NeuralNetwork: 
    ''' 
    Neural network that combines the layers: EmbedPosition, Transformerblock, LinearLayer and Softmax.
    Includes forward and backward pass, as well as gradient descent step.

    '''

    def __init__(self, r: int = 5, d: int = 10, m: int = 2, L: int = 5, p: int = 128, k: int = 8):

        n_max = 2 * r - 1
        self.embedding = EmbedPosition(n_max, m, d, 0.1)
        self.transformer_blocks = [TransformerBlock(d, k, p)]
        for _ in range(L - 1):
            self.transformer_blocks.append(TransformerBlock(d, k, p))

        # Unembedding
        self.lm_head = LinearLayer(d, m, True, 0.1)  
        self.out_softmax = Softmax(1e-8)
    
    def forward(self, x):
        '''
        Performs forward pass through the network.

        '''
        x = self.embedding.forward(x)
        for block in self.transformer_blocks:
            x = block.forward(x)
        x = self.lm_head.forward(x)
        x = self.out_softmax.forward(x)
        return x
    
    def backward(self, grad):
        '''
        Perform backward pass through the network.
        From grad calculate derivative of the loss wrt the final output from the forward pass.

        '''
        dL_dx = self.out_softmax.backward(grad)
        dL_dx = self.lm_head.backward(dL_dx)
        for i in range(len(self.transformer_blocks) - 1, -1, -1):
            dL_dx = self.transformer_blocks[i].backward(dL_dx)
        dL_dx = self.embedding.backward(dL_dx)
        return dL_dx
    
    def step_gd(self, optimizer):
        '''
        Perform a gradient descent step for each layer.
        
        '''

        self.embedding.step_gd(optimizer)
        for block in self.transformer_blocks:
            block.step_gd(optimizer)
        self.lm_head.step_gd(optimizer)