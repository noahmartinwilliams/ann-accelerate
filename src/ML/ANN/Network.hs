module ML.ANN.Network where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import Data.Random.Normal
import ML.ANN.Block
import ML.ANN.BPLayer
import ML.ANN.ErrorFn
import ML.ANN.InfLayer
import ML.ANN.LLayer
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P
import System.Random

mkNetwork :: StdGen -> [LSpec] -> Optim -> Network
mkNetwork gen (first : lspecLs) optim = do
    let norms = normals gen
        (inpLayer, norms') = mkInpLayer first norms 
    (Network (inpLayer : (restOfLayers first lspecLs norms')) optim) where

        numIns :: LSpec -> Int
        numIns l = P.foldr (+) 0 (P.map P.fst l)

        restOfLayers :: LSpec -> [LSpec] -> [Double] -> [Layer]
        restOfLayers _ [] _ = []
        restOfLayers lspec [lspec'] rands = let (l, _) = mkLayer (numIns lspec) lspec' rands in [l]
        restOfLayers lspec (lspec' : rest) rands = do
            let (l, rands') = mkLayer (numIns lspec) lspec' rands
            l : (restOfLayers lspec' rest rands')

inferNetwork :: Network -> Acc (Matrix Double) -> Acc (Matrix Double)
inferNetwork (Network [l] _ ) x = inferLayer l x
inferNetwork (Network (l : rest) o) x = do
    let y = inferLayer l x
    inferNetwork (Network rest o) y

learnNetwork :: Network -> Acc (Matrix Double) -> (LNetwork, Acc (Matrix Double))
learnNetwork (Network [] optim) m = (LNetwork [] optim, m)
learnNetwork (Network (h : t) optim) m = do
    let (l, m') = learnLayer h m
        ((LNetwork l' _), m'') = learnNetwork (Network t optim) m'
    (LNetwork (l : l') optim, m'')

batchLearnNetwork :: Network -> [Acc (Matrix Double)] -> ([LNetwork], [Acc (Matrix Double)])
batchLearnNetwork n inpLs = P.unzip (P.map (learnNetwork n) inpLs)


bpNetwork :: LNetwork -> Acc (Matrix Double) -> (Network, Acc (Matrix Double))
bpNetwork (LNetwork layers optim) bp = do
    let bp' = AccMat bp Outp One
        (n, (AccMat e Outp One)) = intern (P.reverse layers) bp' optim 
    (n, e) where

        intern :: [LLayer] -> AccMat Double Outp One -> Optim -> (Network, AccMat Double Outp One)
        intern [] a o = ((Network [] o), a)
        intern ( h : t) a opt = do
            let (l, e) = bpLayer h opt a
                ((Network l' _), e') = intern t e opt
            (Network (l' P.++ [l]) opt, e')

trainOnce :: BLInfo -> AccBlock -> Acc (Matrix Double, Matrix Double) -> Acc (Matrix Double, Matrix Double, Vector Int, Vector Double)
trainOnce blinfo block sample = do
    let net = block2network blinfo block
        (inp, outp) = A.unlift sample :: (Acc (Matrix Double), Acc (Matrix Double))
        (ln, netOut) = learnNetwork net inp
        err = mseErrorFn netOut outp
        derr = dmseErrorFn netOut outp
        (net', bp) = bpNetwork ln derr
        (_, block') = network2block net'
        (blockI, blockD) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        ret = A.lift (err, bp, blockI, blockD)
    ret
