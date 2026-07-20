{-# LANGUAGE TypeOperators #-}
module ML.ANN.BPLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.ActFunc
import ML.ANN.LLayer
import ML.ANN.LSpec
import ML.ANN.MkLayer
import ML.ANN.Types
import Prelude as P

bpLayer :: LLayer -> Optim -> AccMat Double Outp One -> (Layer, AccMat Double Outp One)
bpLayer (LLayer { llprevInput = prevInput, llayer = l@(Layer { lweights = w, lbiases = b, llspec = lspec})}) (SGDOptim lr) bp = do
    let wT = matTransp w
        x = (w `matMul` prevInput ) `matAdd` b
        deriv = (dactFuncs lspec x)
        dw = (((x `matZipMul` deriv) `matZipMul` bp) `matMul` (matTransp prevInput))  
        w' = w `matSub` (lr `matScale` dw)
        db = ((deriv `matZipMul` bp) `matZipMul` x)
        b' = b `matSub` (lr `matScale` db)
        (AccMat bp' Inp One) = wT `matMul` ((bp `matZipMul` deriv) `matZipMul` x)
    (l { lweights = w', lbiases = b'}, AccMat bp' Outp One)

bpLayer (LLayer { llprevInput = prev, llayer = layer@(InpLayer { vweights = w, vbiases = b, vlspec = lspec })}) (SGDOptim lr) bp = do
    let (AccMat prev' Inp One) = prev
        prev'' = AccMat prev' Outp One
        x = (w `matZipMul` prev'') `matAdd` b
        deriv = (dactFuncs lspec x)
        w' = w `matSub` (lr `matScale` ((deriv `matZipMul` (bp `matZipMul` x)) `matZipMul` prev''))
        b' = b `matSub` (lr `matScale` ((bp `matZipMul` deriv)) `matZipMul` x)
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
        b1t = beta1 A.^ t
        b2t = beta2 A.^ t
        epsilon = constant 0.0000001
        x = (w `matMul` prevInput ) `matAdd` b
        deriv = (dactFuncs lspec x)

        dw = ((((bp `matZipMul` deriv) `matZipMul` x) `matMul` (matTransp prevInput)) ) 
        db = (bp `matZipMul` deriv) `matZipMul` x

        wm' = (beta1 `matScale` wm) `matAdd` ((one - beta1) `matScale` dw)
        bm' = (beta1 `matScale` bm) `matAdd` ((one - beta1) `matScale` db)

        wv' = (beta2 `matScale` wv) `matAdd` ((one - beta2) `matScale` (dw `matZipMul` dw))
        bv' = (beta2 `matScale` bv) `matAdd` ((one - beta2) `matScale` (db `matZipMul` db))

        wmhat = (one / (one - b1t)) `matScale` wm' 
        wvhat = (one / (one - b2t)) `matScale` wv'

        bmhat = (one / (one - b1t)) `matScale` bm' 
        bvhat = (one / (one - b2t)) `matScale` bv'

        w' = w `matSub` (lr `matScale` (wmhat `matZipDiv` (wvhat `matMap` (\y -> (sqrt y) + epsilon))))
        b' = b `matSub` (lr `matScale` (bmhat `matZipDiv` (bvhat `matMap` (\y -> (sqrt y) + epsilon))))

        (AccMat bp' Inp One) = wT `matMul` ((bp `matZipMul` deriv) `matZipMul` x)
    (l { lweights = w', lbiases = b', lweightsMom = wm', lweightsVel = wv', lbiasesMom = bm', lbiasesVel = bv', lnumTimes = (t + (constant 1)) }, AccMat bp' Outp One)

bpLayer (LLayer { llprevInput = (AccMat prev _ _), llayer = l@(InpLayer { vweights = w, vbiases = b, vlspec = lspec })}) (AdamOptim lr beta1 beta2) bp = do
    let wm = vweightsMom l
        wv = vweightsVel l
        bm = vbiasesMom l
        bv = vbiasesVel l
        one = constant 1.0
        epsilon = constant 0.00000001
        t = vnumTimes l
        prev' = AccMat prev Outp One
        x = (w `matZipMul` prev' ) `matAdd` b
        deriv = (dactFuncs lspec x)

        dw = ((x `matZipMul` deriv) `matZipMul` bp) `matZipMul` prev'
        db = (deriv `matZipMul` bp) `matZipMul` x
        b1t = beta1 A.^ t
        b2t = beta2 A.^ t
        wm' = (beta1 `matScale` wm ) `matAdd` ((one - beta1) `matScale` dw)
        wv' = (beta2 `matScale` wv ) `matAdd` ((one - beta2) `matScale` (dw `matZipMul` dw)) 

        bm' = (beta1 `matScale` bm) `matAdd` ((one - beta1) `matScale` db)
        bv' = (beta2 `matScale` bv) `matAdd` ((one - beta2) `matScale` (db `matZipMul` db))
        wmhat = (one / (one - b1t)) `matScale` wm' 
        bmhat = (one / (one - b1t)) `matScale` bm'

        wvhat = (one / (one - b2t)) `matScale` wv'
        bvhat = (one / (one - b2t)) `matScale` bv'
        w' = w `matSub` (lr `matScale` (wmhat `matZipDiv` (wvhat `matMap` (\y -> (sqrt y) + epsilon))))
        b' = b `matSub` (lr `matScale` (bmhat `matZipDiv` (bvhat `matMap` (\y -> (sqrt y) + epsilon))))
        (AccMat bp' Outp One) = w `matZipMul` ((bp `matZipMul` deriv) `matZipMul` x)
    (l { vweights = w', vbiases = b', vweightsMom = wm', vweightsVel = wv', vbiasesMom = bm', vbiasesVel = bv', vnumTimes = (t + (constant 1)) }, AccMat bp' Outp One)

incNumTimes :: AccBlock -> AccBlock
incNumTimes block = do
    let (vi, vd) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        sh = A.shape vi
        toAdd = A.generate sh (\(I1 x) -> (x A./= (constant 0)) A.? ((constant 1), (constant 0)))
    A.lift (A.zipWith (+) vi toAdd, vd) 
