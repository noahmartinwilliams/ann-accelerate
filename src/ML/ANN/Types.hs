{-# LANGUAGE DeriveGeneric #-}
module ML.ANN.Types where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import Prelude as P

data One = One deriving(Generic)
data Inp = Inp deriving(Generic)
data Outp = Outp deriving(Generic)

type Vect a b = AccMat a b One
type Weights = AccMat Double Outp Inp 
type VWeights = Vect Double Outp 
type Biases = Vect Double Outp

data ActFunc = Sigmoid | Relu | SoftMax deriving(Read)

type LSpec = [(Int, ActFunc)]

data Layer = Layer { lnumTimes :: Exp Int, lnumInputs :: Int, lweights :: Weights, lbiases :: Biases , llspec :: LSpec, lweightsMom :: Weights, lbiasesMom :: Biases, lweightsVel :: Weights, lbiasesVel :: Biases} | 
    InpLayer { vnumTimes :: Exp Int, vweights :: VWeights, vbiases :: Biases, vlspec :: LSpec, vweightsMom :: VWeights, vbiasesMom :: Biases, vweightsVel :: VWeights, vbiasesVel :: Biases }

data LLayer = LLayer { llprevInput :: (AccMat Double Inp One), llayer :: Layer }

data Optim = SGDOptim (Exp Double) | AdamOptim (Exp Double) (Exp Double) (Exp Double)

data Network = Network [Layer] Optim ErrorFnT

data LNetwork = LNetwork [LLayer] Optim ErrorFnT

type AccBlock = Acc (Vector Int, Vector Double)

type Block = (Vector Int, Vector Double)

data LayerInfo = LayerInfo Bool LSpec Int 

data BLInfo = BLSGD [LayerInfo] ErrorFnT | BLAdam [LayerInfo] ErrorFnT

data ErrorFnT = MSEErrorFn | CrossEntropyErrorFn 

type ErrorFn = ((Acc (Matrix Double) -> Acc (Matrix Double) -> Acc (Matrix Double)), (Acc (Matrix Double) -> Acc (Matrix Double) -> Acc (Matrix Double)))

