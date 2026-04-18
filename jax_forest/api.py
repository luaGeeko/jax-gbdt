import jax
import jax.numpy as jnp
import numpy as np
import time

# Internal Compiler Imports
from src.compiler.parser import TreeParser
from src.compiler.soft_sigmoid_parser import SoftParser
from src.compiler.laplacian_parser import LaplacianParser

# Internal Kernel Imports
from kernels.no_branch_inference import jax_forest_predict
from kernels.soft_sigmoid_inference import soft_node_activations, calculate_paths_iterative, calculate_paths_dense
from kernels.laplacian_inference import calculate_laplacian_dense


class JAXForestRegressor:
    """
    High-performance JAX compiler for XGBoost Regressor ensembles.
    Converts trained discrete trees into differentiable tensor operations.
    """
    
    def __init__(self, model_path: str, method: str = 'soft_dense'):
        """
        Args:
            model_path (str): Path to the saved XGBoost JSON model.
            method (str): The mathematical backend to use. 
                          Options: 'no_branch', 'soft_iterative', 'soft_dense', 'laplacian_dense'
        """
        self.model_path = model_path
        self.method = method.lower()
        self._is_compiled = False
        
        valid_methods = ['no_branch', 'soft_iterative', 'soft_dense', 'laplacian_dense']
        if self.method not in valid_methods:
            raise ValueError(f"Method '{self.method}' not recognized. Choose from {valid_methods}")
            
        self._load_and_parse()
        self._build_kernel()

    def _load_and_parse(self):
        """Internal method to parse the XGBoost JSON into the requested mathematical topology."""
        if self.method == 'no_branch':
            parser = TreeParser(trained_model_path=self.model_path)
            self._params = parser.parse()
            
        elif self.method in ['soft_iterative', 'soft_dense']:
            parser = SoftParser(trained_model_path=self.model_path)
            backend = 'dense' if self.method == 'soft_dense' else 'iterative'
            self._params = parser.routing_matrices(method_type=backend)
            
        elif self.method == 'laplacian_dense':
            parser = LaplacianParser(trained_model_path=self.model_path)
            W, tau = parser.get_routing_matrices()
            top = parser.get_graph_topology()
            self._params = {'W': W, 'tau': tau, 'dense_A_template': top['dense_A_template'], 'leaf_weights': top['leaf_weights']}

    def _build_kernel(self):
        """Internal method to wrap the appropriate JAX math with @jax.jit."""
        if self.method == 'no_branch':
            @jax.jit
            def _kernel(X):
                return jax_forest_predict(X, self._params['features'], self._params['thresholds'], 
                                          self._params['left_children'], self._params['right_children'], 
                                          self._params['max_depth'])
                
        elif self.method == 'soft_iterative':
            @jax.jit
            def _kernel(X):
                lp, rp = soft_node_activations(X, self._params['W'], self._params['tau'], temperature=10.0)
                return calculate_paths_iterative(lp, rp, self._params['lefts'], self._params['rights'], self._params['thresholds'])
                
        elif self.method == 'soft_dense':
            @jax.jit
            def _kernel(X):
                lp, rp = soft_node_activations(X, self._params['W'], self._params['tau'], temperature=10.0)
                return calculate_paths_dense(lp, rp, self._params['A'], self._params['leaf_wts'])
                
        elif self.method == 'laplacian_dense':
            @jax.jit
            def _kernel(X):
                lp, rp = soft_node_activations(X, self._params['W'], self._params['tau'], temperature=10.0)
                return calculate_laplacian_dense(lp, rp, self._params['dense_A_template'], self._params['leaf_weights'])
                
        self._predict_fn = _kernel

    def compile(self, data: np.ndarray):
        """
        Triggers XLA compilation (Cold Start) and locks in the expected batch size.
        
        Args:
            dummy_X (np.ndarray): A sample batch of data matching the expected shape for inference.
        """
        jax_X = jnp.array(data)
        _ = self._predict_fn(jax_X).block_until_ready()
        self._is_compiled = True
        self._compiled_shape = data.shape

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Executes ultra-fast inference using the compiled JAX kernel.
        
        Args:
            X (np.ndarray): Input features. Must match the shape passed to compile().
            
        Returns:
            np.ndarray: Predicted regression values.
        """
        if not self._is_compiled:
            print("Warning: Model not explicitly compiled. Triggering cold-start compilation now...")
            self.compile(X)
            
        if X.shape != self._compiled_shape:
            print(f"Warning: Input shape {X.shape} differs from compiled shape {self._compiled_shape}. This will trigger a re-compilation.")
            
        jax_X = jnp.array(X)
        predictions = self._predict_fn(jax_X).block_until_ready()
        return np.array(predictions)