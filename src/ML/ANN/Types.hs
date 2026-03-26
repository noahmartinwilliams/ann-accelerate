module ML.ANN.Types where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import Prelude as P

data One = One
data Inp = Inp
data Outp = Outp

type Vect a b = AccMat a b One
type Weights = AccMat Double Outp Inp 
type VWeights = Vect Double Outp 
type Biases = Vect Double Outp

data ActFunc = Sigmoid | Relu | SoftMax

type LSpec = [(Int, ActFunc)]

data Layer = Layer { lnumInputs :: Int, lweights :: Weights, lbiases :: Biases , llspec :: LSpec} | 
    InpLayer { vweights :: VWeights, vbiases :: Biases, vlspec :: LSpec}

data LLayer = LLayer { llprevInput :: (AccMat Double Inp One), llayer :: Layer }

data Optim = SGDOptim (Exp Double) 
