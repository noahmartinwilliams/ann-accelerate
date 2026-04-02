module ML.ANN.Block where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix
import ML.ANN.LSpec
import ML.ANN.Types
import Prelude as P

getllspec :: Layer -> LSpec
getllspec (Layer { llspec = l }) = l
getllspec (InpLayer { vlspec = l }) = l

isInpLayer :: Layer -> Bool
isInpLayer (Layer {}) = False
isInpLayer (InpLayer {} ) = True

network2block :: Network -> (BLInfo, AccBlock)
network2block (Network layers optim@(SGDOptim lr) errfn) = do
    let lr' = A.unit lr
        lr'' = A.replicate (constant (Z:.(1 :: Int))) lr' :: Acc (Vector Double)
        lspecs = P.map getllspec layers
        numIns = getNumIns layers
        bools = P.map isInpLayer layers
        (aints, adoubles) = P.unzip (P.map (layer2block optim) layers)
        numLayers = P.length layers
        aints' = P.foldl (A.++) (use (fromList (Z:.1) [numLayers])) aints
        adoubles' = P.foldl (A.++) lr'' adoubles
        blinfo = P.zipWith3 LayerInfo bools lspecs numIns
    (BLSGD blinfo errfn, A.lift (aints', adoubles')) 
network2block (Network layers optim@(AdamOptim lr beta1 beta2) errFn) = do
    let lr' = A.unit lr
        lr'' = A.replicate (constant (Z:.(1 :: Int))) lr' :: Acc (Vector Double)
        beta1' = A.unit beta1
        beta1'' = A.replicate (constant (Z:.(1 :: Int))) beta1' :: Acc (Vector Double)
        beta2' = A.unit beta2
        beta2'' = A.replicate (constant (Z:.(1 :: Int))) beta2' :: Acc (Vector Double)
        lspecs = P.map getllspec layers
        numIns = getNumIns layers
        bools = P.map isInpLayer layers
        (aints, adoubles) = P.unzip (P.map (layer2block optim) layers)
        numLayers = P.length layers
        aints' = P.foldl (A.++) (use (fromList (Z:.1) [numLayers])) aints
        adoubles' = P.foldl (A.++) lr'' (beta1'' : beta2'' : adoubles)
        blinfo = P.zipWith3 LayerInfo bools lspecs numIns
    (BLSGD blinfo errFn, A.lift (aints', adoubles')) 

getNumIns :: [Layer] -> [Int]
getNumIns [] = []
getNumIns ((InpLayer { vlspec = vl}) : rest) = (getLSpecNumOuts vl) : (getNumIns rest)
getNumIns ((Layer { lnumInputs = ni}) : rest) = ni : (getNumIns rest)

layer2block :: Optim -> Layer -> (Acc (Vector Int), Acc (Vector Double))
layer2block (SGDOptim _) (InpLayer { vweights = (AccMat w _ _) , vbiases = (AccMat b _ _)}) = do
    let ds = (A.flatten w) A.++ (A.flatten b)
        is = use (A.fromList (Z:.0) [])
    (is, ds)
layer2block (SGDOptim _) (Layer { lweights = (AccMat w _ _), lbiases = (AccMat b _ _ )}) = do
    let ds = (A.flatten w) A.++ (A.flatten b)
        is = use (A.fromList (Z:.0) [])
    (is, ds)
layer2block (AdamOptim _ _ _) l@(Layer {}) = do
    let (AccMat w _ _) = lweights l
        (AccMat b _ _ ) = lbiases l
        (AccMat wv _ _ ) = lweightsVel l
        (AccMat wm _ _) = lweightsMom l
        (AccMat bv _ _) = lbiasesVel l
        (AccMat bm _ _) = lbiasesMom l
        ds = (A.flatten w) A.++ (A.flatten b) A.++ (A.flatten wm) A.++ (A.flatten bm) A.++ (A.flatten wv) A.++ (A.flatten bv)
        is = use (A.fromList (Z:.0) [])
    (is, ds)
layer2block (AdamOptim _ _ _) l@(InpLayer {}) = do
    let (AccMat w _ _) = vweights l
        (AccMat b _ _ ) = vbiases l
        (AccMat wv _ _ ) = vweightsVel l
        (AccMat wm _ _ ) = vweightsMom l
        (AccMat bv _ _ ) = vbiasesVel l
        (AccMat bm _ _ ) = vbiasesMom l
        ds = (A.flatten w) A.++ (A.flatten b) A.++ (A.flatten wm) A.++ (A.flatten bm) A.++ (A.flatten wv) A.++ (A.flatten bv)
        is = use (A.fromList (Z:.0) [])
    (is, ds)

block2network :: BLInfo -> AccBlock -> Network
block2network (BLSGD ls errfn) accblock = do
    let (accIs, accDs) = A.unlift accblock
        accIs' = A.drop (constant 1) accIs
        lr = A.the (A.reshape (constant Z) (A.take (constant 1) accDs ))
        accDs' = A.drop (constant 1) accDs
        layers = intern ls accIs' accDs'
    Network layers (SGDOptim lr) errfn where

        intern :: [LayerInfo] -> Acc (Vector Int) -> Acc (Vector Double) -> [Layer] 
        intern [] _ _ = []
        intern ((LayerInfo True lspec numIns) : rest) ints doubles = do
            let weightsV = A.take (constant numIns) doubles
                weightsR = A.drop (constant numIns) doubles
                biasesV = A.take (constant numIns) weightsR
                biasesR = A.drop (constant numIns) weightsR
                weightsM = AccMat (A.reshape (constant (Z:.numIns:.1)) weightsV) Outp One
                biasesM = AccMat (A.reshape (constant (Z:.numIns:.1)) biasesV) Outp One
                zerosV = AccMat (use (A.fromList (Z:.numIns:.1) (P.repeat 0.0))) Outp One
                layer = InpLayer { vlspec = lspec, vweights = weightsM, vbiases = biasesM, vweightsMom = zerosV, vbiasesMom = zerosV, vweightsVel = zerosV}
            (layer : (intern rest ints biasesR))

        intern ((LayerInfo False lspec numIns) : rest) ints doubles = do
            let numOuts = getLSpecNumOuts lspec
                numWeights = constant (numIns * numOuts)
                weightsV = A.take numWeights doubles
                restWeights = A.drop numWeights doubles
                biasesV = A.take (constant numOuts ) restWeights
                weightsM = AccMat (A.reshape (constant (Z:.numOuts:.numIns)) weightsV) Outp Inp
                biasesM = AccMat (A.reshape (constant (Z:.numOuts:.1)) biasesV) Outp One
                restDoubles = A.drop (constant numOuts) restWeights
                zerosM = AccMat (use (A.fromList (Z:.numOuts:.numIns) (P.repeat 0.0))) Outp Inp
                zerosV = AccMat (use (A.fromList (Z:.numOuts:.1) (P.repeat 0.0))) Outp One
                layer = Layer { llspec = lspec, lnumInputs = numIns, lweights = weightsM, lbiases = biasesM, lbiasesMom = zerosV, lbiasesVel = zerosV, lweightsMom = zerosM, lweightsVel = zerosM}
            (layer : (intern rest ints restDoubles))

block2network (BLAdam ls errFn) accblock = do
    let (accIs, accDs) = A.unlift accblock :: (Acc (Vector Int), Acc (Vector Double))
        accIs' = A.drop (constant 1) accIs
        one = constant 1
        lr = A.the (A.reshape (constant Z) (A.take one accDs ))
        beta1 = A.the (A.reshape (constant Z) (A.take one (A.drop one accDs)))
        beta2 = A.the (A.reshape (constant Z) (A.take one (A.drop (constant 2) accDs)))
        accDs' = A.drop (constant 3) accDs
        layers = intern ls accIs' accDs'
    (Network layers (AdamOptim lr beta1 beta2) errFn) where

        intern :: [LayerInfo] -> Acc (Vector Int) -> Acc (Vector Double) -> [Layer] 
        intern [] _ _ = []
        intern ((LayerInfo True lspec numIns) : rest) ints doubles = do
            let numIns' = constant numIns
            let weightsV = A.take numIns' doubles
                weightsR = A.drop numIns' doubles
                biasesV = A.take numIns' weightsR
                biasesR = A.drop numIns' weightsR
                weightsM = AccMat (A.reshape (constant (Z:.numIns:.1)) weightsV) Outp One
                biasesM = AccMat (A.reshape (constant (Z:.numIns:.1)) biasesV) Outp One
                weightsMom = A.reshape (constant (Z:.numIns:.1)) (A.take numIns' biasesR)
                weightsMomR = A.drop numIns' biasesR
                biasesMom = A.reshape (constant (Z:.numIns:.1)) (A.take numIns' weightsMomR)
                biasesMomR = A.drop numIns' weightsMomR
                weightsVel = A.reshape (constant (Z:.numIns:.1)) (A.take numIns' biasesMomR)
                weightsVelR = A.drop numIns' biasesMomR
                biasesVel = A.reshape (constant (Z:.numIns:.1)) (A.take numIns' weightsVelR)
                biasesVelR = A.drop numIns' weightsVelR
                layer = InpLayer { vweights = weightsM, vbiases = biasesM, vbiasesMom = (AccMat biasesMom Outp One), vbiasesVel = (AccMat biasesVel Outp One), vweightsMom = (AccMat weightsMom Outp One), vweightsVel = (AccMat weightsVel Outp One)}
            (layer : (intern rest ints biasesVelR))

        intern ((LayerInfo True lspec numIns) : rest) ints doubles = do
            let numOuts = getLSpecNumOuts lspec
                numOuts' = constant numOuts
                numWeights = constant (numIns * numOuts)
                weightsV = A.take numWeights doubles
                weightsR = A.drop numWeights doubles
                biasesV = A.take numOuts' weightsR
                biasesR = A.drop numOuts' weightsR
                weightsM = AccMat (A.reshape (constant (Z:.numOuts:.numIns)) weightsV) Outp Inp
                biasesM = AccMat (A.reshape (constant (Z:.numOuts:.1)) biasesV) Outp One
                weightsMom = A.reshape (constant (Z:.numOuts:.numIns)) (A.take numWeights biasesR)
                weightsMomR = A.drop numWeights biasesR
                biasesMom = A.reshape (constant (Z:.numOuts:.1)) (A.take numOuts' weightsMomR)
                biasesMomR = A.drop numOuts' weightsMomR
                weightsVel = A.reshape (constant (Z:.numOuts:.numIns)) (A.take numWeights biasesMomR)
                weightsVelR = A.drop numWeights biasesMomR
                biasesVel = A.reshape (constant (Z:.numOuts:.1)) (A.take numOuts' weightsVelR)
                biasesVelR = A.drop numOuts' weightsVelR
                layer = Layer { lweights = weightsM, lbiases = biasesM, lbiasesMom = (AccMat biasesMom Outp One), lbiasesVel = (AccMat biasesVel Outp One), lweightsMom = (AccMat weightsMom Outp Inp), lweightsVel = (AccMat weightsVel Outp Inp)}
            (layer : (intern rest ints biasesVelR))

