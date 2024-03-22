{-# LANGUAGE DeriveGeneric, GADTs, TypeFamilies #-}
module ML.ANN.Block(BlInfo(..), BlockInfo(..), BlockA, BlockV, layer2block, block2layer, network2block, block2network) where

import ML.ANN.Network
import ML.ANN.Mat
import ML.ANN.Vect
import ML.ANN.Layer
import ML.ANN.Optim

import Data.Array.Accelerate as A
import Prelude as P

import Data.Serialize

data BlInfo = SGDBlInfo Int Int LSpec | -- numInputs numOutputs lspec
    SGDBlInpInfo LSpec |
    MomBlInfo Int Int LSpec | --numInputs numOutputs lspec
    MomBlInpInfo LSpec |
    RMSBlInfo Int Int LSpec |
    RMSBlInpInfo LSpec |
    AdagradBlInfo Int Int LSpec | 
    AdagradBlInpInfo LSpec |
    AdamBlInfo Int Int LSpec | 
    AdamBlInpInfo LSpec
    deriving(Show, P.Eq, Generic) -- lspec

data BlockInfo = BlockInfo Optim [BlInfo ] deriving(Show, P.Eq, Generic)

type BlockA = Acc (Vector Int, Vector Double) 
type BlockV = (Vector Int, Vector Double)


instance Serialize BlInfo
instance Serialize BlockInfo

layer2block :: Layer -> (BlInfo, BlockA)
layer2block (SGDInpLayer weights bias lspec) = do
    let (VectO weightsV) = weights
        (VectO biasV) = bias
        weightsFlat = A.flatten weightsV
        biasFlat = A.flatten biasV
        integers = use (fromList (Z:.1) [(lspecGetNumOutputs lspec)])
        lifted = A.lift (integers, weightsFlat A.++ biasFlat)
    ((SGDBlInpInfo lspec), lifted)

layer2block (SGDLayer numInputs weights bias lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractMat weights)
        biasFlat = A.flatten (extractVect bias)
        integers = use (fromList (Z:.2) [numInputs, numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ biasFlat)
    ((SGDBlInfo numInputs numOutputs lspec), lifted)

layer2block (MomLayer numInputs weights weightsMom biases biasesMom lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractMat weights)
        weightsMomFlat = A.flatten (extractMat weightsMom)
        biasesFlat = A.flatten (extractVect biases)
        biasesMomFlat = A.flatten (extractVect biasesMom)
        integers = use (fromList (Z:.2) [numInputs, numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsMomFlat A.++ biasesFlat A.++ biasesMomFlat)
    (MomBlInfo numInputs numOutputs lspec, lifted)

layer2block (MomInpLayer weights weightsMom biases biasesMom lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractVect weights)
        weightsMomFlat = A.flatten (extractVect weightsMom)
        biasesFlat = A.flatten (extractVect biases)
        biasesMomFlat = A.flatten (extractVect biasesMom)
        integers = use (fromList (Z:.1) [numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsMomFlat A.++ biasesFlat A.++ biasesMomFlat)
    (MomBlInpInfo lspec, lifted)

layer2block (RMSLayer numInputs weights weightsMom biases biasesMom lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractMat weights)
        weightsMomFlat = A.flatten (extractMat weightsMom)
        biasesFlat = A.flatten (extractVect biases)
        biasesMomFlat = A.flatten (extractVect biasesMom)
        integers = use (fromList (Z:.2) [numInputs, numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsMomFlat A.++ biasesFlat A.++ biasesMomFlat)
    (RMSBlInfo numInputs numOutputs lspec, lifted)

layer2block (RMSInpLayer weights weightsMom biases biasesMom lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractVect weights)
        weightsMomFlat = A.flatten (extractVect weightsMom)
        biasesFlat = A.flatten (extractVect biases)
        biasesMomFlat = A.flatten (extractVect biasesMom)
        integers = use (fromList (Z:.1) [numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsMomFlat A.++ biasesFlat A.++ biasesMomFlat)
    (RMSBlInpInfo lspec, lifted)

layer2block (AdagradLayer numInputs weights weightsSummed biases biasesSummed lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractMat weights)
        weightsSummedFlat = A.flatten (extractMat weightsSummed)
        biasesFlat = A.flatten (extractVect biases)
        biasesSummedFlat = A.flatten (extractVect biasesSummed)
        integers = use (fromList (Z:.2) [numInputs, numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsSummedFlat A.++ biasesFlat A.++ biasesSummedFlat)
    (AdagradBlInfo numInputs numOutputs lspec, lifted)

layer2block (AdagradInpLayer weights weightsSummed biases biasesSummed lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractVect weights)
        weightsSummedFlat = A.flatten (extractVect weightsSummed)
        biasesFlat = A.flatten (extractVect biases)
        biasesSummedFlat = A.flatten (extractVect biasesSummed)
        integers = use (fromList (Z:.1) [numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsSummedFlat A.++ biasesFlat A.++ biasesSummedFlat)
    (AdagradBlInpInfo lspec, lifted)

layer2block (AdamLayer numInputs weights weightsM weightsV biases biasesM biasesV lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractMat weights)
        weightsMomFlat = A.flatten (extractMat weightsM)
        weightsVelFlat = A.flatten (extractMat weightsV)
        biasesFlat = A.flatten (extractVect biases)
        biasesMomFlat = A.flatten (extractVect biasesM)
        biasesVelFlat = A.flatten (extractVect biasesV)
        integers = use (fromList (Z:.2) [numInputs, numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsMomFlat A.++ weightsVelFlat A.++ biasesFlat A.++ biasesMomFlat A.++ biasesVelFlat )
    (AdamBlInfo numInputs numOutputs lspec, lifted)

layer2block (AdamInpLayer weights weightsM weightsV biases biasesM biasesV lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractVect weights)
        weightsMomFlat = A.flatten (extractVect weightsM)
        weightsVelFlat = A.flatten (extractVect weightsV)
        biasesFlat = A.flatten (extractVect biases)
        biasesMomFlat = A.flatten (extractVect biasesM)
        biasesVelFlat = A.flatten (extractVect biasesV)
        integers = use (fromList (Z:.1) [numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ weightsMomFlat A.++ weightsVelFlat A.++ biasesFlat A.++ biasesMomFlat A.++ biasesVelFlat )
    (AdamBlInpInfo lspec, lifted)


block2layer :: (BlInfo, BlockA) -> (Layer, BlockA)
block2layer ((SGDBlInpInfo lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numOutputs = lspecGetNumOutputs lspec
        weightsAV = A.take (constant numOutputs) doubles
        biasesAV = A.take (constant numOutputs) (A.drop (constant numOutputs) doubles)
        retBlockD = A.drop (constant (2 * numOutputs)) doubles
        retBlockI = A.drop (constant 1) integers
        weightsAM = VectO (A.replicate (A.lift (Z:.All:.(1 :: Int))) weightsAV)
        biasesAM = VectO (A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV)
        retBlock = A.lift (retBlockI, retBlockD)
    ((SGDInpLayer weightsAM biasesAM lspec), retBlock)

block2layer ((SGDBlInfo numInputs numOutputs lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = numInputs * numOutputs
        numBiases = numOutputs
        weightsAV = A.take (constant numWeights) doubles
        biasesAV = A.take (constant numBiases) (A.drop (constant numWeights) doubles)
        retBlockD = A.drop (constant (numWeights + numBiases)) doubles
        retBlockI = A.drop (constant 2) integers -- Ignore the integers, those are there in case we someday want to store integer data in there that actually means something.
        weightsAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        retBlock = A.lift (retBlockI, retBlockD)
    ((SGDLayer numInputs (MatOI weightsAM) (VectO biasesAM) lspec), retBlock)

block2layer ((MomBlInfo numInputs numOutputs lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = numInputs * numOutputs
        numBiases = numOutputs
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsMomAV, rest1) = takeDrop numWeights rest0
        (biasesAV, rest2) = takeDrop numBiases rest1
        (biasesMomAV, rest3) = takeDrop numBiases rest2
        weightsAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsAV
        weightsMomAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsMomAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        biasesMomAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesMomAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest3)
    ((MomLayer numInputs (MatOI weightsAM) (MatOI weightsMomAM) (VectO biasesAM) (VectO biasesMomAM) lspec), retBlock)

block2layer ((MomBlInpInfo lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = lspecGetNumOutputs lspec
        numBiases = numWeights
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsMomAV, rest1) = takeDrop numWeights rest0
        (biasesAV, rest2) = takeDrop numBiases rest1
        (biasesMomAV, rest3) = takeDrop numBiases rest2
        weightsAM = A.reshape (constant (Z:.numWeights:.1)) weightsAV
        weightsMomAM = A.reshape (constant (Z:.numWeights:.1)) weightsMomAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        biasesMomAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesMomAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest3)
    ((MomInpLayer (VectO weightsAM) (VectO weightsMomAM) (VectO biasesAM) (VectO biasesMomAM) lspec), retBlock)

block2layer ((RMSBlInfo numInputs numOutputs lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = numInputs * numOutputs
        numBiases = numOutputs
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsMomAV, rest1) = takeDrop numWeights rest0
        (biasesAV, rest2) = takeDrop numBiases rest1
        (biasesMomAV, rest3) = takeDrop numBiases rest2
        weightsAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsAV
        weightsMomAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsMomAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        biasesMomAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesMomAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest3)
    ((RMSLayer numInputs (MatOI weightsAM) (MatOI weightsMomAM) (VectO biasesAM) (VectO biasesMomAM) lspec), retBlock)

block2layer ((RMSBlInpInfo lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = lspecGetNumOutputs lspec
        numBiases = numWeights
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsMomAV, rest1) = takeDrop numWeights rest0
        (biasesAV, rest2) = takeDrop numBiases rest1
        (biasesMomAV, rest3) = takeDrop numBiases rest2
        weightsAM = A.reshape (constant (Z:.numWeights:.1)) weightsAV
        weightsMomAM = A.reshape (constant (Z:.numWeights:.1)) weightsMomAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        biasesMomAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesMomAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest3)
    ((RMSInpLayer (VectO weightsAM) (VectO weightsMomAM) (VectO biasesAM) (VectO biasesMomAM) lspec), retBlock)

block2layer ((AdagradBlInfo numInputs numOutputs lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = numInputs * numOutputs
        numBiases = numOutputs
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsSummedAV, rest1) = takeDrop numWeights rest0
        (biasesAV, rest2) = takeDrop numBiases rest1
        (biasesSummedAV, rest3) = takeDrop numBiases rest2
        weightsAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsAV
        weightsSummedAM = A.reshape (constant (Z:.numOutputs:.numInputs)) weightsSummedAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        biasesSummedAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesSummedAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest3)
    ((AdagradLayer numInputs (MatOI weightsAM) (MatOI weightsSummedAM) (VectO biasesAM) (VectO biasesSummedAM) lspec), retBlock)

block2layer ((AdagradBlInpInfo lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = lspecGetNumOutputs lspec
        numBiases = numWeights
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsSummedAV, rest1) = takeDrop numWeights rest0
        (biasesAV, rest2) = takeDrop numBiases rest1
        (biasesSummedAV, rest3) = takeDrop numBiases rest2
        weightsAM = A.reshape (constant (Z:.numWeights:.1)) weightsAV
        weightsSummedAM = A.reshape (constant (Z:.numWeights:.1)) weightsSummedAV
        biasesAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesAV
        biasesSummedAM = A.replicate (A.lift (Z:.All:.(1 :: Int))) biasesSummedAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest3)
    ((AdagradInpLayer (VectO weightsAM) (VectO weightsSummedAM) (VectO biasesAM) (VectO biasesSummedAM) lspec), retBlock)

block2layer ((AdamBlInfo numInputs numOutputs lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = numInputs * numOutputs
        numBiases = numOutputs
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsMomAV, rest1) = takeDrop numWeights rest0
        (weightsVelAV, rest2) = takeDrop numWeights rest1
        (biasesAV, rest3) = takeDrop numBiases rest2
        (biasesMomAV, rest4) = takeDrop numBiases rest3
        (biasesVelAV, rest5) = takeDrop numBiases rest4
        weightsSh = constant (Z:.numOutputs:.numInputs)
        biasesSh = Z:.All:.(1 :: Int)
        weightsAM = A.reshape weightsSh weightsAV
        weightsMomAM = A.reshape weightsSh weightsMomAV
        weightsVelAM = A.reshape weightsSh weightsVelAV
        biasesAM = A.replicate (A.lift biasesSh) biasesAV
        biasesMomAM = A.replicate (A.lift biasesSh) biasesMomAV
        biasesVelAM = A.replicate (A.lift biasesSh) biasesVelAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest5)
    ((AdamLayer numInputs (MatOI weightsAM) (MatOI weightsMomAM) (MatOI weightsVelAM) (VectO biasesAM) (VectO biasesMomAM) (VectO biasesVelAM) lspec), retBlock)

block2layer ((AdamBlInpInfo lspec), block) = do
    let (integers, doubles) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        numWeights = lspecGetNumOutputs lspec
        numBiases = numWeights
        (weightsAV, rest0) = takeDrop numWeights doubles
        (weightsMomAV, rest1) = takeDrop numWeights rest0
        (weightsVelAV, rest2) = takeDrop numWeights rest1
        (biasesAV, rest3) = takeDrop numBiases rest2
        (biasesMomAV, rest4) = takeDrop numBiases rest3
        (biasesVelAV, rest5) = takeDrop numBiases rest4
        weightsSh = constant (Z:.numWeights:.1)
        biasesSh = A.lift (Z:.All:.(1 :: Int))
        weightsAM = A.reshape weightsSh weightsAV
        weightsMomAM = A.reshape weightsSh weightsMomAV
        weightsVelAM = A.reshape weightsSh weightsVelAV
        biasesAM = A.replicate biasesSh biasesAV
        biasesMomAM = A.replicate biasesSh biasesMomAV
        biasesVelAM = A.replicate biasesSh biasesVelAV
        retBlockI = A.drop (constant 2) integers
        retBlock = A.lift (retBlockI, rest5)
    ((AdamInpLayer (VectO weightsAM) (VectO weightsMomAM) (VectO weightsVelAM) (VectO biasesAM) (VectO biasesMomAM) (VectO biasesVelAM) lspec), retBlock)
        

takeDrop :: Int -> Acc (Vector Double) -> (Acc (Vector Double), Acc (Vector Double))
takeDrop x inp = (A.take (constant x) inp, A.drop (constant x) inp)

network2block :: Network -> (BlockInfo, BlockA)
network2block (Network layers (RMSProp lr beta) _) = do
    let lr2 = A.reshape (constant (Z:.1)) (A.unit (constant lr))
        beta2 = A.reshape (constant (Z:.1)) (A.unit (constant beta))
        (blinfo, retblock) = layers2block layers
        (vint, vdouble) = A.unlift retblock :: (Acc (Vector Int), Acc (Vector Double))
        vdouble2 = lr2 A.++ beta2 A.++ vdouble
        retblock2 = A.lift (vint, vdouble2)
    (BlockInfo (RMSProp lr beta) blinfo, retblock2) where

        layers2block :: [Layer] -> ([BlInfo], BlockA)
        layers2block [] = let emptyi = use (fromList (Z:.0) []) in let emptyd = use (fromList (Z:.0) []) in ([], A.lift (emptyi, emptyd))
        layers2block (h : t) = do
            let (info, intdoubles) = layer2block h
                (integers, doubles) = A.unlift intdoubles
                (infoRest, intdoublesRest) = layers2block t
                (integersRest, doublesRest) = A.unlift intdoublesRest
                ret = A.lift ((integers A.++ integersRest), (doubles A.++ doublesRest))
            (info : infoRest, ret)

network2block (Network layers (Mom lr beta) _) = do
    let lr2 = A.reshape (constant (Z:.1)) (A.unit (constant lr))
        beta2 = A.reshape (constant (Z:.1)) (A.unit (constant beta))
        (blinfo, retblock) = layers2block layers
        (vint, vdouble) = A.unlift retblock :: (Acc (Vector Int), Acc (Vector Double))
        vdouble2 = lr2 A.++ beta2 A.++ vdouble
        retblock2 = A.lift (vint, vdouble2)
    (BlockInfo (Mom lr beta) blinfo, retblock2) where

        layers2block :: [Layer] -> ([BlInfo], BlockA)
        layers2block [] = let emptyi = use (fromList (Z:.0) []) in let emptyd = use (fromList (Z:.0) []) in ([], A.lift (emptyi, emptyd))
        layers2block (h : t) = do
            let (info, intdoubles) = layer2block h
                (integers, doubles) = A.unlift intdoubles
                (infoRest, intdoublesRest) = layers2block t
                (integersRest, doublesRest) = A.unlift intdoublesRest
                ret = A.lift ((integers A.++ integersRest), (doubles A.++ doublesRest))
            (info : infoRest, ret)

network2block (Network layers (SGD lr) _) = do
    let lr2 = A.reshape (constant (Z:.1)) (A.unit (constant lr))
        (blinfo, retblock) = layers2block layers
        (vint, vdouble) = A.unlift retblock :: (Acc (Vector Int), Acc (Vector Double))
        vdouble2 = lr2 A.++ vdouble
        retblock2 = A.lift (vint, vdouble2)
    (BlockInfo (SGD lr) blinfo, retblock2) where

        layers2block :: [Layer] -> ([BlInfo], BlockA)
        layers2block [] = let emptyi = use (fromList (Z:.0) []) in let emptyd = use (fromList (Z:.0) []) in ([], A.lift (emptyi, emptyd))
        layers2block (h : t) = do
            let (info, intdoubles) = layer2block h
                (integers, doubles) = A.unlift intdoubles
                (infoRest, intdoublesRest) = layers2block t
                (integersRest, doublesRest) = A.unlift intdoublesRest
                ret = A.lift ((integers A.++ integersRest), (doubles A.++ doublesRest))
            (info : infoRest, ret)

network2block (Network layers (Adagrad lr) _) = do
    let lr2 = A.reshape (constant (Z:.1)) (A.unit (constant lr))
        (blinfo, retblock) = layers2block layers
        (vint, vdouble) = A.unlift retblock :: (Acc (Vector Int), Acc (Vector Double))
        vdouble2 = lr2 A.++ vdouble
        retblock2 = A.lift (vint, vdouble2)
    (BlockInfo (Adagrad lr) blinfo, retblock2) where

        layers2block :: [Layer] -> ([BlInfo], BlockA)
        layers2block [] = let emptyi = use (fromList (Z:.0) []) in let emptyd = use (fromList (Z:.0) []) in ([], A.lift (emptyi, emptyd))
        layers2block (h : t) = do
            let (info, intdoubles) = layer2block h
                (integers, doubles) = A.unlift intdoubles
                (infoRest, intdoublesRest) = layers2block t
                (integersRest, doublesRest) = A.unlift intdoublesRest
                ret = A.lift ((integers A.++ integersRest), (doubles A.++ doublesRest))
            (info : infoRest, ret)

network2block (Network layers (Adam alpha beta1 beta2) numTimes) = do
    let lr2 = use (fromList (Z:.3) [alpha, beta1, beta2])
        (blinfo, retblock) = layers2block layers
        (vint, vdouble) = A.unlift retblock :: (Acc (Vector Int), Acc (Vector Double))
        vdouble2 = lr2 A.++ vdouble
        numTimes' = A.reshape (constant (Z:.1)) numTimes
        retblock2 = A.lift (numTimes' A.++ vint, vdouble2)
    (BlockInfo (Adam alpha beta1 beta2) blinfo, retblock2) where

        layers2block :: [Layer] -> ([BlInfo], BlockA)
        layers2block [] = let emptyi = use (fromList (Z:.0) []) in let emptyd = use (fromList (Z:.0) []) in ([], A.lift (emptyi, emptyd))
        layers2block (h : t) = do
            let (info, intdoubles) = layer2block h
                (integers, doubles) = A.unlift intdoubles
                (infoRest, intdoublesRest) = layers2block t
                (integersRest, doublesRest) = A.unlift intdoublesRest
                ret = A.lift ((integers A.++ integersRest), (doubles A.++ doublesRest))
            (info : infoRest, ret)

block2network :: (BlockInfo, BlockA) -> Network
block2network (BlockInfo (SGD lr) blinfos, blocka) = do
    let (integers, doubles) = A.unlift blocka :: (Acc (Vector Int), Acc (Vector Double))
        restDoubles = A.drop (constant 1) doubles
        restV = A.lift (integers, restDoubles) :: Acc (Vector Int, Vector Double)
    Network (intern blinfos restV) (SGD lr) (use (fromList (Z) [0])) where

        intern :: [BlInfo] -> BlockA -> [Layer]
        intern [] _ = []
        intern (h : rest) blockA = do
            let (layer, blockRest) = block2layer (h, blockA)
            layer : (intern rest blockRest)

block2network (BlockInfo (Mom lr beta) blinfos, blocka) = do
    let (integers, doubles) = A.unlift blocka :: (Acc (Vector Int), Acc (Vector Double))
        restDoubles = A.drop (constant 2) doubles
        restV = A.lift (integers, restDoubles) :: Acc (Vector Int, Vector Double)
    Network (intern blinfos restV) (Mom lr beta) (use (fromList (Z) [0])) where

        intern :: [BlInfo] -> BlockA -> [Layer]
        intern [] _ = []
        intern (h : rest) blockA = do
            let (layer, blockRest) = block2layer (h, blockA)
            layer : (intern rest blockRest)

block2network (BlockInfo (RMSProp lr beta) blinfos, blocka) = do
    let (integers, doubles) = A.unlift blocka :: (Acc (Vector Int), Acc (Vector Double))
        restDoubles = A.drop (constant 2) doubles
        restV = A.lift (integers, restDoubles) :: Acc (Vector Int, Vector Double)
    Network (intern blinfos restV) (RMSProp lr beta) (use (fromList (Z) [0])) where

        intern :: [BlInfo] -> BlockA -> [Layer]
        intern [] _ = []
        intern (h : rest) blockA = do
            let (layer, blockRest) = block2layer (h, blockA)
            layer : (intern rest blockRest)

block2network (BlockInfo (Adagrad lr) blinfos, blocka) = do
    let (integers, doubles) = A.unlift blocka :: (Acc (Vector Int), Acc (Vector Double))
        restDoubles = A.drop (constant 1) doubles
        restV = A.lift (integers, restDoubles) :: Acc (Vector Int, Vector Double)
    Network (intern blinfos restV) (Adagrad lr) (use (fromList (Z) [0])) where

        intern :: [BlInfo] -> BlockA -> [Layer]
        intern [] _ = []
        intern (h : rest) blockA = do
            let (layer, blockRest) = block2layer (h, blockA)
            layer : (intern rest blockRest)

block2network (BlockInfo (Adam alpha beta1 beta2) blinfos, blocka) = do
    let (integers, doubles) = A.unlift blocka :: (Acc (Vector Int), Acc (Vector Double))
        time = A.reshape (constant Z) (A.take (constant 1) integers)
        restIntegers = A.drop (constant 1) integers
        restDoubles = A.drop (constant 3) doubles
        restV = A.lift (restIntegers, restDoubles) :: Acc (Vector Int, Vector Double)
    Network (intern blinfos restV) (Adam alpha beta1 beta2) time where

        intern :: [BlInfo] -> BlockA -> [Layer]
        intern [] _ = []
        intern (h : rest) blockA = do
            let (layer, blockRest) = block2layer (h, blockA)
            layer : (intern rest blockRest)
