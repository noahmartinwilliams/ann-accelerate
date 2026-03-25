module ML.ANN.InfLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix as M
import ML.ANN.ActFunc
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P

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

inferLayer :: Acc (Matrix Double) -> Layer -> Acc (Matrix Double)
inferLayer inp (Layer { lweights = w, lbiases = b, llspec = lspec }) = do
    let inp' = AccMat inp Inp One
        m = (w `matMul` inp') `matAdd` b
        (AccMat m' One Outp) = actFuncs lspec (matTransp m )
    A.transpose m'
