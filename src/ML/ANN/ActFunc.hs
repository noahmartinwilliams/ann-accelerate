module ML.ANN.ActFunc where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.ANN.Types
import Prelude as P

sigmoid :: Exp Double -> Exp Double
sigmoid x = let one = constant 1.0 in one / (one + (exp (-x)))

dsigmoid :: Exp Double -> Exp Double
dsigmoid x = do
    let one = constant 1.0
        e = exp (-x)
        es = (e + one) * (e + one)
    e / es

relu :: Exp Double -> Exp Double
relu x = A.max (constant 0.0) x

softmax :: Acc (Matrix Double) -> Acc (Matrix Double)
softmax x = do
    let e = A.map (\y -> exp y) x
        s = A.sum (A.flatten e)
        s' = A.the s
    A.map (\y -> y / s') e

data Interm = Interm
data Interm2 = Interm2

actFunc :: ActFunc -> AccMat Double a b -> AccMat Double a b
actFunc Sigmoid (AccMat inp a b) = AccMat (A.map sigmoid inp) a b
actFunc Relu (AccMat inp a b) = AccMat (A.map relu inp) a b
actFunc SoftMax (AccMat inp a b) = AccMat (softmax inp) a b

actFuncs :: LSpec -> AccMat Double One Outp -> AccMat Double One Outp
actFuncs [] (AccMat m _ _ ) = AccMat m One Outp
actFuncs ((i, af) : rest) m = do
    let m' = matTake (constant i) m Interm
        m'' = actFunc af m'
        (AccMat mrest One _) = matDrop (constant i) m Interm2
        m2 = actFuncs rest (AccMat mrest One Outp)
    (matAppend m'' m2 Outp)

dactFunc :: ActFunc -> AccMat Double a b -> AccMat Double a b
dactFunc Sigmoid (AccMat m a b) = AccMat (A.map dsigmoid m) a b

dactFuncs :: LSpec -> AccMat Double One Outp -> AccMat Double One Outp
dactFuncs [] (AccMat m _ _) = AccMat m One Outp
dactFuncs ((i, af) : rest) m = do
    let m' = matTake (constant i) m Interm
        m'' = dactFunc af m'
        (AccMat mrest One _) = matDrop (constant i) m Interm2
        m2 = dactFuncs rest (AccMat mrest One Outp)
    (matAppend m'' m2 Outp)
