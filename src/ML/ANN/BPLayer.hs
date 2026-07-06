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
        deriv = (dactFuncs lspec x)
        dw = (((x `matZipMul` deriv) `matZipMul` bp) `matMul` (matTransp prevInput))  
        w' = w `matSub` (lr `matScale` dw)
        db = (deriv `matZipMul` bp)
        b' = b `matSub` (lr `matScale` db)
        (AccMat bp' Inp One) = wT `matMul` ((bp `matZipMul` deriv) `matZipMul` x)
    (l { lweights = w', lbiases = b'}, AccMat bp' Outp One)

bpLayer (LLayer { llprevInput = prev, llayer = layer@(InpLayer { vweights = w, vbiases = b, vlspec = lspec })}) (SGDOptim lr) bp = do
    let (AccMat prev' Inp One) = prev
        prev'' = AccMat prev' Outp One
        x = (w `matZipMul` prev'') `matAdd` b
        deriv = (dactFuncs lspec x)
        w' = w `matSub` (lr `matScale` (deriv `matZipMul` (bp `matZipMul` x)))
        b' = b `matSub` (lr `matScale` (deriv `matZipMul` bp))
        bp' = w `matZipMul` ((deriv `matZipMul` bp) `matZipMul` x)
    ( layer { vweights = w', vbiases = b' }, bp')

bpLayer (LLayer { llprevInput = prevInput, llayer = l@(Layer { lweights = w, lbiases = b, llspec = lspec})}) (AdamOptim lr beta1 beta2) bp = do
    let wT = matTransp w
        wm = lweightsMom l
        wv = lweightsVel l
        bm = lbiasesMom l
        bv = lbiasesVel l
        t = lnumTimes l
        one = constant 1.0
        epsilon = constant 0.0000001
        x = (w `matMul` prevInput ) `matAdd` b
        deriv = (dactFuncs lspec x)

        dw = ((((bp `matZipMul` deriv) `matZipMul` x) `matMul` (matTransp prevInput)) ) 
        wm' = (beta1 `matScale` wm) `matAdd` ((one - beta1) `matScale` dw)
        wv' = (beta2 `matScale` wv) `matAdd` ((one - beta2) `matScale` (dw `matZipMul` dw))
        b1t = beta1 A.^ t
        b2t = beta2 A.^ t
        wmhat = (one / (one - b1t)) `matScale` wm'
        wvhat = (one / (one - b2t)) `matScale` wv'
        wvhatsqrt = wvhat `matMap` (\y -> one / ((sqrt y) + epsilon))
        w' = w `matSub` (lr `matScale` (wmhat `matZipMul` wvhatsqrt))

        db = (deriv `matZipMul` bp)
        bm' = (beta1 `matScale` bm) `matAdd` ((one - beta1) `matScale` db)
        bv' = (beta2 `matScale` bv) `matAdd` ((one - beta2) `matScale` (db `matZipMul` db))
        bmhat = (one / (one - beta1)) `matScale` bm'
        bvhat = (one / (one - beta2)) `matScale` bv'
        bvhatsqrt = bvhat `matMap` (\y -> one / ((sqrt y) + epsilon))
        b' = b `matSub` (lr `matScale` (bmhat `matZipMul` bvhatsqrt))
        (AccMat bp' Inp One) = wT `matMul` ((bp `matZipMul` deriv) `matZipMul` x)
    (l { lweights = w', lbiases = b', lweightsMom = wm', lweightsVel = wv', lbiasesMom = bm', lbiasesVel = bv', lnumTimes = (t + (constant 1)) }, AccMat bp' Outp One)

bpLayer (LLayer { llprevInput = (AccMat prev _ _), llayer = l@(InpLayer { vweights = w, vbiases = b, vlspec = lspec })}) (AdamOptim lr beta1 beta2) bp = do
    let wm = vweightsMom l
        wv = vweightsVel l
        bm = vbiasesMom l
        bv = vbiasesVel l
        one = constant 1.0
        t = vnumTimes l
        epsilon = constant 0.0000001
        prev' = AccMat prev Outp One
        x = (w `matZipMul` prev' ) `matAdd` b
        deriv = (dactFuncs lspec x)

        dw = ((x `matZipMul` deriv) `matZipMul` bp) 
        wm' = (beta1 `matScale` wm) `matAdd` ((one - beta1) `matScale` dw)
        wv' = (beta2 `matScale` wv) `matAdd` ((one - beta2) `matScale` (dw `matZipMul` dw))
        b1t = beta1 A.^ t
        b2t = beta2 A.^ t
        wmhat = (one / (one - b1t)) `matScale` wm'
        wvhat = (one / (one - b2t)) `matScale` wv'
        wvhatsqrt = wvhat `matMap` (\y -> one / ((sqrt y) + epsilon))
        w' = w `matSub` (lr `matScale` (wmhat `matZipMul` wvhatsqrt))


        db = (deriv `matZipMul` bp)
        bm' = (beta1 `matScale` bm) `matAdd` ((one - beta1) `matScale` db)
        bv' = (beta2 `matScale` bv) `matAdd` ((one - beta2) `matScale` (db `matZipMul` db))
        bmhat = (one / (one - beta1)) `matScale` bm'
        bvhat = (one / (one - beta2)) `matScale` bv'
        bvhatsqrt = bvhat `matMap` (\y -> one / ((sqrt y) + epsilon))
        b' = b `matSub` (lr `matScale` (bmhat `matZipMul` bvhatsqrt))
        (AccMat bp' Outp One) = w `matZipMul` ((bp `matZipMul` deriv) `matZipMul` x)
    (l { vweights = w', vbiases = b', vweightsMom = wm', vweightsVel = wv', vbiasesMom = bm', vbiasesVel = bv', vnumTimes = (t + (constant 1)) }, AccMat bp' Outp One)

