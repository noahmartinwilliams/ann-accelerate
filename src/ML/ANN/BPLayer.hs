module ML.ANN.BPLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.ActFunc
import ML.ANN.LLayer
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P

batchBPLayers :: [LLayer] -> Optim -> [AccMat Double Outp One] -> (Layer, AccMat Double Outp One)
batchBPLayers llayers (SGDOptim lr) bpLayers = do
    let (layers, (fbps : bps)) = P.unzip (P.zipWith (\x -> \bp -> bpLayer x (SGDOptim lr) bp) llayers bpLayers)
        ws = P.map lweights layers
        bs = P.map lbiases layers
        numLayers = P.length llayers
        numLayersD = constant (1.0 / (P.fromIntegral numLayers :: Double))
        (firstWs : wsRest) = ws
        (firstBs : bsRest) = bs
        ws' = avgMs firstWs wsRest numLayersD
        bs' = avgMs firstBs bsRest numLayersD
        firstLayer = layers P.!! 0
    (firstLayer { lweights = ws', lbiases = bs'}, avgMs fbps bps numLayersD) where

        avgMs :: AccMat Double a b -> [AccMat Double a b] -> Exp Double -> AccMat Double a b
        avgMs m l d = d `matScale` (P.foldl matAdd m l )


bpLayer :: LLayer -> Optim -> AccMat Double Outp One -> (Layer, AccMat Double Outp One)
bpLayer (LLayer { llprevInput = prevInput, llayer = l@(Layer { lnumInputs = ni, lweights = w, lbiases = b, llspec = lspec})}) (SGDOptim lr) bp = do
    let wT = matTransp w
        x = (w `matMul` prevInput ) `matAdd` b
        deriv = matTransp (dactFuncs lspec (matTransp x))
        onesV = AccMat (use (fromList (Z:.ni:.1) (P.repeat 1.0))) Inp One
        bp' = bp `matMul` (matTransp onesV)
        dw = ((x `matZipMul` deriv) `matZipMul` bp) `matMul` (matTransp onesV)
        w' = w `matSub` (lr `matScale` dw)
        db = (deriv `matZipMul` bp)
        b' = b `matSub` (lr `matScale` db)
        (AccMat bp'' Inp One) = wT `matMul` bp
    (l { lweights = w', lbiases = b'}, AccMat bp'' Outp One)
bpLayer (LLayer { llprevInput = prev, llayer = layer@(InpLayer { vweights = w, vbiases = b, vlspec = lspec })}) (SGDOptim lr) bp = do
    let (AccMat prev' Inp One) = prev
        prev'' = AccMat (A.transpose prev') One Outp
        prev2 = matTransp prev''
        deriv = matTransp (dactFuncs lspec prev'')
        w' = w `matSub` (lr `matScale` (deriv `matZipMul` (bp `matZipMul` prev2)))
        b' = b `matSub` (lr `matScale` (deriv `matZipMul` bp))
        bp' = w `matZipMul` (deriv `matZipMul` bp)
    ( layer { vweights = w', vbiases = b' }, bp')
