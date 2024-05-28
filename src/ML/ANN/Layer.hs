{-# LANGUAGE GADTs, DataKinds, KindSignatures, TypeOperators #-}
module ML.ANN.Layer (
    Layer(..), 
    LLayer(..), 
    LSpec(), 
    calcLayer, 
    lspecGetNumOutputs, 
    mkSGDLayer, 
    mkSGDInpLayer, 
    learnLayer, 
    backpropLayer, 
    module ML.ANN.MkLayer ) where

import Data.Array.Accelerate as A
import Prelude as P

import ML.ANN.Mat 
import ML.ANN.Vect
import ML.ANN.ActFuncs
import ML.ANN.Optim
import ML.ANN.MkLayer

calcLayerIntern :: Mat OutputSize InputSize -> Vect OutputSize -> LSpec -> Acc (Matrix Double) -> Acc (Matrix Double)
calcLayerIntern weights biases lspec x = do
    let x2 = (weights `mmulv` (VectI x)) `vaddv` biases
        (VectO output) = applyActFuncs lspec x2
    output

calcLayer :: Layer -> Acc (Matrix Double) -> Acc (Matrix Double)
calcLayer (SGDInpLayer weights bias lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output
calcLayer (SGDLayer _ weights bias lspec) x = calcLayerIntern weights bias lspec x
calcLayer (MomLayer _ weights _ biases _ lspec) x = calcLayerIntern weights biases lspec x
calcLayer (MomInpLayer weights _  bias _ lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output

calcLayer (RMSLayer _ weights _ biases _ lspec) x = calcLayerIntern weights biases lspec x
calcLayer (RMSInpLayer weights _ bias _ lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output

calcLayer (AdagradLayer _ weights _ biases _ lspec) x = calcLayerIntern weights biases lspec x
calcLayer (AdagradInpLayer weights _ bias _ lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output

calcLayer (AdamLayer _ weights _ _ biases _ _ lspec) x = calcLayerIntern weights biases lspec x

calcLayer (AdamInpLayer weights _ _ bias _ _ lspec) x = do
    let x2 = (weights `vmulv` (VectO x)) `vaddv` bias
        (VectO output) = applyActFuncs lspec x2
    output

learnLayer :: Layer -> Acc (Matrix Double) -> (LLayer, Acc (Matrix Double))
learnLayer (SGDInpLayer weights biases lspec) input = do
    let x = VectO input
        output = applyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
    ((LInpLayer (SGDInpLayer weights biases lspec) x), (extractVect output))

learnLayer (SGDLayer numInputs weights biases lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LLayer (SGDLayer numInputs weights biases lspec) x), (extractVect output))

learnLayer (MomLayer numInputs weights weightsMom biases biasesMom lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LLayer (MomLayer numInputs weights weightsMom biases biasesMom lspec) x), (extractVect output))

learnLayer (MomInpLayer weights weightsMom biases biasesMom lspec) input = do
    let x = VectO input
        output = applyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
    ((LInpLayer (MomInpLayer weights weightsMom biases biasesMom lspec) x), (extractVect output))

learnLayer (RMSLayer numInputs weights weightsMom biases biasesMom lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LLayer (RMSLayer numInputs weights weightsMom biases biasesMom lspec) x), (extractVect output))

learnLayer (RMSInpLayer weights weightsMom biases biasesMom lspec) input = do
    let x = VectO input
        output = applyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
    ((LInpLayer (RMSInpLayer weights weightsMom biases biasesMom lspec) x), (extractVect output))

learnLayer (AdagradLayer numInputs weights weightsSummed biases biasesSummed lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LLayer (AdagradLayer numInputs weights weightsSummed biases biasesSummed lspec) x), (extractVect output))

learnLayer (AdagradInpLayer weights weightsSummed biases biasesSummed lspec) input = do
    let x = VectO input
        output = applyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
    ((LInpLayer (AdagradInpLayer weights weightsSummed biases biasesSummed lspec) x), (extractVect output))

learnLayer (AdamLayer numInputs weights weightsMom weightsVel biases biasesMom biasesVel lspec) input = do
    let x = VectI input
        output = applyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
    ((LLayer (AdamLayer numInputs weights weightsMom weightsVel biases biasesMom biasesVel lspec) x), (extractVect output))

learnLayer (AdamInpLayer weights weightsMom weightsVel biases biasesMom biasesVel lspec) input = do
    let x = VectO input
        output = applyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
    ((LInpLayer (AdamInpLayer weights weightsMom weightsVel biases biasesMom biasesVel lspec) x), (extractVect output))

backpropLayer :: LLayer -> Optim -> Acc (Matrix Double) -> Exp Int -> (Layer, Acc (Matrix Double))
backpropLayer (LInpLayer (SGDInpLayer weights biases lspec) x) (SGD learnRate) bp _ = do
    let lr = constant learnRate
        bp2 = VectO bp
        deriv = dapplyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
        weights2 = weights `vsubv` (lr `smulv` (x `vmulv` (deriv `vmulv` bp2)))
        biases2 = biases `vsubv` (lr `smulv` (deriv `vmulv` bp2))
        bp3 = weights `vmulv` (deriv `vmulv` bp2)
        (VectO bp4) = bp3
    ((SGDInpLayer weights2 biases2 lspec), bp4)

backpropLayer (LLayer (SGDLayer numInputs weights biases lspec) x) (SGD learnRate) bp _ = do
    let lr = constant learnRate
        bp2 = VectO bp
        deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        weights2 = weights `msubm` (lr `smulm` (x `vxv` (deriv `vmulv` bp2 ) ))
        biases2 = biases `vsubv` (lr `smulv` (deriv `vmulv` bp2))
        bp3 = ((transp weights) `mmulv`  (deriv `vmulv` bp2))
        (VectI bp4) = bp3
    ((SGDLayer numInputs weights2 biases2 lspec), bp4)

backpropLayer (LLayer (MomLayer numInputs weights weightsMom biases biasesMom lspec) x) (Mom alpha mom) bpInp _ = do
    let deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        bp = VectO bpInp
        momExp = constant mom
        alphaExp = constant alpha
        fprimeW = x `vxv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        changeWeights = (alphaExp `smulm` fprimeW) `maddm` (momExp `smulm` weightsMom)
        changeBiases = (alphaExp `smulv` fprimeB) `vaddv` (momExp `smulv` biasesMom)
        weights2 = weights `msubm` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = (transp weights) `mmulv` (bp `vmulv` deriv)
        (VectI bp3) = bp2
    ((MomLayer numInputs weights2 changeWeights biases2 changeBiases lspec), bp3)

backpropLayer (LInpLayer (MomInpLayer weights weightsMom biases biasesMom lspec) x) (Mom alpha mom) bpInp _ = do
    let deriv = dapplyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
        bp = VectO bpInp
        momExp = constant mom
        alphaExp = constant alpha
        fprimeW = x `vmulv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        changeWeights = (alphaExp `smulv` fprimeW) `vaddv` (momExp `smulv` weightsMom)
        changeBiases = (alphaExp `smulv` fprimeB) `vaddv` (momExp `smulv` biasesMom)
        weights2 = weights `vsubv` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = weights `vmulv` (bp `vmulv` deriv)
        (VectO bp3) = bp2
    ((MomInpLayer weights2 changeWeights biases2 changeBiases lspec), bp3)

backpropLayer (LLayer (RMSLayer numInputs weights weightsV biases biasesV lspec) x) (RMSProp alpha beta) bpInp _ = do
    let deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        bp = VectO bpInp
        betaExp = constant beta
        alphaExp = constant alpha
        fprimeW = x `vxv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        vdw = (betaExp `smulm` weightsV) `maddm` ((constant (1.0 - beta)) `smulm` (mzipw (*) fprimeW fprimeW))
        vdb = (betaExp `smulv` biasesV) `vaddv` ((constant (1.0 - beta)) `smulv` (vzipw (*) fprimeB fprimeB))
        epsilon = constant 0.00001
        changeWeights = alphaExp `smulm` (mzipw (\z -> \y -> z / ((sqrt y) + epsilon)) fprimeW weightsV)
        changeBiases = alphaExp `smulv` (vzipw (\z -> \y -> z / ((sqrt y) + epsilon)) fprimeB biasesV)
        weights2 = weights `msubm` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = (transp weights) `mmulv` (bp `vmulv` deriv)
        (VectI bp3) = bp2
    ((RMSLayer numInputs weights2 vdw biases2 vdb lspec), bp3)

backpropLayer (LInpLayer (RMSInpLayer weights weightsV biases biasesV lspec) x) (RMSProp alpha beta) bpInp _ = do
    let deriv = dapplyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
        bp = VectO bpInp
        betaExp = constant beta
        alphaExp = constant alpha
        fprimeW = x `vmulv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        vdw = (betaExp `smulv` weightsV) `vaddv` ((constant (1.0 - beta)) `smulv` (vzipw (*) fprimeW fprimeW))
        vdb = (betaExp `smulv` biasesV) `vaddv` ((constant (1.0 - beta)) `smulv` (vzipw (*) fprimeB fprimeB))
        epsilon = constant 0.00001
        changeWeights = alphaExp `smulv` (vzipw (\z -> \y -> z / ((sqrt y) + epsilon)) fprimeW weightsV)
        changeBiases = alphaExp `smulv` (vzipw (\z -> \y -> z / ((sqrt y) + epsilon)) fprimeB biasesV)
        weights2 = weights `vsubv` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = weights `vmulv` (bp `vmulv` deriv)
        (VectO bp3) = bp2
    ((RMSInpLayer weights2 vdw biases2 vdb lspec), bp3)

backpropLayer (LInpLayer (AdagradInpLayer weights weightsSummed biases biasesSummed lspec) x) (Adagrad learnRate) bp _ = do
    let bp2 = VectO bp
        lr = constant learnRate
        epsilon = constant 0.00001
        lrW = vmap (\z -> lr / (sqrt (z + epsilon))) weightsSummed
        lrB = vmap (\z -> lr / (sqrt (z + epsilon))) biasesSummed
        deriv = dapplyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
        weightsDelta = x `vmulv` (deriv `vmulv` bp2)
        biasesDelta = deriv `vmulv` bp2
        weightsChange = vzipw (*) lrW weightsDelta
        biasesChange = vzipw (*) lrB biasesDelta
        weights2 = weights `vsubv` weightsChange
        biases2 = biases `vsubv` biasesChange
        weightsSummed2 = vzipw (*) weightsDelta weightsDelta
        biasesSummed2 = vzipw (*) biasesDelta biasesDelta
        bp3 = weights `vmulv` (deriv `vmulv` bp2)
        (VectO bp4) = bp3
    ((AdagradInpLayer weights2 weightsSummed2 biases2 biasesSummed2 lspec), bp4)

backpropLayer (LLayer (AdagradLayer numInputs weights weightsSummed biases biasesSummed lspec) x) (Adagrad learnRate) bp _ = do
    let bp2 = VectO bp
        lr = constant learnRate
        epsilon = constant 0.00001
        lrW = mmap (\z -> lr / (sqrt (z + epsilon))) weightsSummed
        lrB = vmap (\z -> lr / (sqrt (z + epsilon))) biasesSummed
        deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        weightsDelta = x `vxv` (deriv `vmulv` bp2)
        biasesDelta = deriv `vmulv` bp2
        weightsChange = mzipw (*) lrW weightsDelta
        biasesChange = vzipw (*) lrB biasesDelta
        weights2 = weights `msubm` weightsChange
        biases2 = biases `vsubv` biasesChange
        weightsSummed2 = msquare weightsDelta 
        biasesSummed2 = vsquare biasesDelta
        bp3 = (transp weights) `mmulv` (deriv `vmulv` bp2)
        (VectI bp4) = bp3
    ((AdagradLayer numInputs weights2 weightsSummed2 biases2 biasesSummed2 lspec), bp4)

-- TODO: figure out how to incorporate the number of samples used so far into the algorithm so that beta1 and beta2 can be taken to the power of t.
backpropLayer (LLayer (AdamLayer numInputs weights weightsM weightsV biases biasesM biasesV lspec) x) (Adam alpha beta1 beta2) bpInp t = do
    let deriv = dapplyActFuncs lspec ((weights `mmulv` x) `vaddv` biases)
        bp = VectO bpInp
        beta2Exp = constant beta2
        beta1Exp = constant beta1
        alphaExp = constant alpha
        fprimeW = x `vxv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        one = constant 1.0 :: Exp Double
        vdw = (beta2Exp `smulm` weightsV) `maddm` ((constant (1.0 - beta2)) `smulm` (msquare fprimeW)) -- velocity of delta weights
        vdb = (beta2Exp `smulv` biasesV) `vaddv` ((constant (1.0 - beta2)) `smulv` (vsquare fprimeB))
        mdw = (beta1Exp `smulm` weightsM) `maddm` ((constant (1.0 - beta1)) `smulm` fprimeW) -- momentum of delta weights
        mdb = (beta1Exp `smulv` biasesM) `vaddv` ((constant (1.0 - beta1)) `smulv` fprimeB)
        mhw = (one / (one - (beta1Exp A.^ t))) `smulm` mdw -- momentum hat of weights
        mhb = (one / (one - (beta1Exp A.^ t))) `smulv` mdb
        vhw = (one / (one - (beta2Exp A.^ t))) `smulm` vdw -- velocity hat of weights
        vhb = (one / (one - (beta2Exp A.^ t))) `smulv` vdb 
        epsilon = constant 0.00001
        changeWeights = alphaExp `smulm` (mzipw (\z -> \y -> z / ((sqrt y) + epsilon)) mhw vhw)
        changeBiases = alphaExp `smulv` (vzipw (\z -> \y -> z / ((sqrt y) + epsilon)) mhb vhb)
        weights2 = weights `msubm` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = (transp weights) `mmulv` (bp `vmulv` deriv)
        (VectI bp3) = bp2
    ((AdamLayer numInputs weights2 mdw vdw biases2 mdb vdb lspec), bp3)

backpropLayer (LInpLayer (AdamInpLayer weights weightsM weightsV biases biasesM biasesV lspec) x) (Adam alpha beta1 beta2) bpInp t = do
    let deriv = dapplyActFuncs lspec ((weights `vmulv` x) `vaddv` biases)
        bp = VectO bpInp
        beta2Exp = constant beta2
        beta1Exp = constant beta1
        alphaExp = constant alpha
        fprimeW = x `vmulv` (bp `vmulv` deriv)
        fprimeB = bp `vmulv` deriv
        one = constant 1.0 :: Exp Double
        vdw = (beta2Exp `smulv` weightsV) `vaddv` ((constant (1.0 - beta2)) `smulv` (vsquare fprimeW)) -- velocity of delta weights
        vdb = (beta2Exp `smulv` biasesV) `vaddv` ((constant (1.0 - beta2)) `smulv` (vsquare fprimeB))
        mdw = (beta1Exp `smulv` weightsM) `vaddv` ((constant (1.0 - beta1)) `smulv` fprimeW) -- momentum of delta weights
        mdb = (beta1Exp `smulv` biasesM) `vaddv` ((constant (1.0 - beta1)) `smulv` fprimeB)
        mhw = (one / (one - (beta1Exp A.^ t))) `smulv` mdw -- momentum hat of weights
        mhb = (one / (one - (beta1Exp A.^ t))) `smulv` mdb
        vhw = (one / (one - (beta2Exp A.^ t))) `smulv` vdw -- velocity hat of weights
        vhb = (one / (one - (beta2Exp A.^ t))) `smulv` vdb 
        epsilon = constant 0.00001
        changeWeights = alphaExp `smulv` (vzipw (\z -> \y -> z / ((sqrt y) + epsilon)) mhw vhw)
        changeBiases = alphaExp `smulv` (vzipw (\z -> \y -> z / ((sqrt y) + epsilon)) mhb vhb)
        weights2 = weights `vsubv` changeWeights
        biases2 = biases `vsubv` changeBiases
        bp2 = weights `vmulv` (bp `vmulv` deriv)
        (VectO bp3) = bp2
    ((AdamInpLayer weights2 mdw vdw biases2 mdb vdb lspec), bp3)


