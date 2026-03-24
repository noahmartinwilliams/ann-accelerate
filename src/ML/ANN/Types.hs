module ML.ANN.Types where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import Prelude as P

data One = One
data Inp = Inp
data Outp = Outp

type Vect a b = AccMat a b One
type Weights = AccMat Double Outp Inp 
type Biases = Vect Double Outp

data LayerType = SGD

data Layer = Layer { lweights :: Weights, lbiases :: Biases , ltype :: LayerType}
