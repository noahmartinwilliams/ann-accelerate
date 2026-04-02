module ML.ANN.BPLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.ActFunc
import ML.ANN.LLayer
import ML.ANN.LSpec
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
bpLayer (LLayer { llprevInput = prevInput, llayer = l@(Layer { lnumInputs = ni, lweights = w, lbiases = b, llspec = lspec})}) (AdamOptim lr beta1 beta2) bp = do
    let wT = matTransp w
        wm = lweightsMom l
        wv = lweightsVel l
        bm = lbiasesMom l
        bv = lbiasesVel l
        one = constant 1.0
        epsilon = constant 0.00000001
        x = (w `matMul` prevInput ) `matAdd` b
        deriv = matTransp (dactFuncs lspec (matTransp x))
        onesV = AccMat (use (fromList (Z:.ni:.1) (P.repeat 1.0))) Inp One
        bp' = bp `matMul` (matTransp onesV)

        dw = ((x `matZipMul` deriv) `matZipMul` bp) `matMul` (matTransp onesV)
        wm' = (beta1 `matScale` wm) `matAdd` ((one - beta1) `matScale` dw)
        wv' = (beta2 `matScale` wv) `matAdd` ((one - beta2) `matScale` (dw `matZipMul` dw))
        wmhat = (one / (one - beta1)) `matScale` wm'
        wvhat = (one / (one - beta2)) `matScale` wv'
        wvhatsqrt = wvhat `matMap` (\x -> one / ((sqrt x) + epsilon))
        w' = w `matSub` (lr `matScale` (wmhat `matZipMul` wvhatsqrt))


        db = (deriv `matZipMul` bp)
        bm' = (beta1 `matScale` bm) `matAdd` ((one - beta1) `matScale` db)
        bv' = (beta2 `matScale` bv) `matAdd` ((one - beta2) `matScale` (db `matZipMul` db))
        bmhat = (one / (one - beta1)) `matScale` bm'
        bvhat = (one / (one - beta2)) `matScale` bv'
        bvhatsqrt = bvhat `matMap` (\x -> one / ((sqrt x) + epsilon))
        b' = b `matSub` (lr `matScale` (bmhat `matZipMul` bvhatsqrt))
        (AccMat bp'' Inp One) = wT `matMul` bp
    (l { lweights = w', lbiases = b', lweightsMom = wm', lweightsVel = wv', lbiasesMom = bm', lbiasesVel = bv' }, AccMat bp'' Outp One)

bpLayer (LLayer { llprevInput = (AccMat prev _ _), llayer = l@(InpLayer { vweights = w, vbiases = b, vlspec = lspec })}) (AdamOptim lr beta1 beta2) bp = do
    let wm = vweightsMom l
        wv = vweightsVel l
        bm = vbiasesMom l
        bv = vbiasesVel l
        one = constant 1.0
        epsilon = constant 0.00000001
        prev' = AccMat prev Outp One
        ni = getLSpecNumOuts lspec
        x = (w `matZipMul` prev' ) `matAdd` b
        deriv = matTransp (dactFuncs lspec (matTransp x))

        dw = ((x `matZipMul` deriv) `matZipMul` bp) 
        wm' = (beta1 `matScale` wm) `matAdd` ((one - beta1) `matScale` dw)
        wv' = (beta2 `matScale` wv) `matAdd` ((one - beta2) `matScale` (dw `matZipMul` dw))
        wmhat = (one / (one - beta1)) `matScale` wm'
        wvhat = (one / (one - beta2)) `matScale` wv'
        wvhatsqrt = wvhat `matMap` (\x -> one / ((sqrt x) + epsilon))
        w' = w `matSub` (lr `matScale` (wmhat `matZipMul` wvhatsqrt))


        db = (deriv `matZipMul` bp)
        bm' = (beta1 `matScale` bm) `matAdd` ((one - beta1) `matScale` db)
        bv' = (beta2 `matScale` bv) `matAdd` ((one - beta2) `matScale` (db `matZipMul` db))
        bmhat = (one / (one - beta1)) `matScale` bm'
        bvhat = (one / (one - beta2)) `matScale` bv'
        bvhatsqrt = bvhat `matMap` (\x -> one / ((sqrt x) + epsilon))
        b' = b `matSub` (lr `matScale` (bmhat `matZipMul` bvhatsqrt))
        (AccMat bp' Outp One) = w `matZipMul` bp
    (l { vweights = w', vbiases = b', vweightsMom = wm', vweightsVel = wv', vbiasesMom = bm', vbiasesVel = bv' }, AccMat bp' Outp One)
