module ML.ANN.Block where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.ANN.Types
import Prelude as P

getllspec :: Layer -> LSpec
getllspec (Layer { llspec = l }) = l
getllspec (InpLayer { vlspec = l }) = l

isInpLayer :: Layer -> Bool
isInpLayer (Layer {}) = False
isInpLayer (InpLayer {} ) = True

network2block :: Network -> (BLInfo, AccBlock)
network2block (Network layers optim@(SGDOptim lr)) = do
    let lr' = A.unit lr
        lr'' = A.replicate (constant (Z:.(1 :: Int))) lr' :: Acc (Vector Double)
        lspecs = P.map getllspec layers
        numOuts = P.map getLSpecNumOuts lspecs
        bools = P.map isInpLayer layers
        (aints, adoubles) = P.unzip (P.map (layer2block optim) layers)
        numLayers = P.length layers
        aints' = P.foldl (A.++) (use (fromList (Z:.1) [numLayers])) aints
        adoubles' = P.foldl (A.++) lr'' adoubles
        blinfo = P.zipWith3 LayerInfo bools lspecs numOuts
    (BLSGD blinfo, A.lift (aints', adoubles')) where

        getLSpecNumOuts :: LSpec -> Int
        getLSpecNumOuts l = P.foldr (+) 0 (P.map P.fst l)

        layer2block :: Optim -> Layer -> (Acc (Vector Int), Acc (Vector Double))
        layer2block (SGDOptim _) (InpLayer { vweights = (AccMat w _ _) , vbiases = (AccMat b _ _)}) = do
            let ds = (A.flatten w) A.++ (A.flatten b)
                is = use (A.fromList (Z:.0) [])
            (is, ds)
        layer2block (SGDOptim _) (Layer { lweights = (AccMat w _ _), lbiases = (AccMat b _ _ )}) = do
            let ds = (A.flatten w) A.++ (A.flatten b)
                is = use (A.fromList (Z:.0) [])
            (is, ds)
