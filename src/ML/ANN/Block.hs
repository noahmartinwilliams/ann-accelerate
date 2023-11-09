module ML.ANN.Block(BLInfo(..), BlockInfo(..), BlockA, BlockV, layer2block, block2layer, network2block) where

import ML.ANN.Network
import ML.ANN.Mat
import ML.ANN.Vect
import ML.ANN.Layer
import ML.ANN.Optim

import Data.Array.Accelerate as A
import Prelude as P

data BLInfo = SGDBLInfo Int Int LSpec deriving(Show, P.Eq) -- numInputs numOutputs lspec

data BlockInfo = BlockInfo Optim [BLInfo ] deriving(Show)

type BlockA = Acc (Vector Int, Vector Double)
type BlockV = (Vector Int, Vector Double)


layer2block :: Layer -> (BLInfo, BlockA)
layer2block (SGDLayer numInputs weights bias lspec) = do
    let numOutputs = lspecGetNumOutputs lspec
        weightsFlat = A.flatten (extractMat weights)
        biasFlat = A.flatten (extractVect bias)
        integers = use (fromList (Z:.2) [numInputs, numOutputs])
        lifted = A.lift (integers, weightsFlat A.++ biasFlat)
    ((SGDBLInfo numInputs numOutputs lspec), lifted)

block2layer :: (BLInfo, BlockA) -> (Layer, BlockA)
block2layer ((SGDBLInfo numInputs numOutputs lspec), block) = do
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

network2block :: Network -> (BlockInfo, BlockA)
network2block (SGDNetwork layers lr) = do
    let lr2 = A.reshape (constant (Z:.1)) (A.unit lr)
        (blinfo, retblock) = layers2block layers
    (BlockInfo (SGD lr) blinfo, retblock) where
        layers2block :: [Layer] -> ([BLInfo], BlockA)
        layers2block [] = let emptyi = use (fromList (Z:.0) []) in let emptyd = use (fromList (Z:.0) []) in ([], A.lift (emptyi, emptyd))
        layers2block (head : tail) = do
            let (info, intdoubles) = layer2block head
                (integers, doubles) = A.unlift intdoubles
                (infoRest, intdoublesRest) = layers2block tail
                (integersRest, doublesRest) = A.unlift intdoublesRest
                ret = A.lift ((integers A.++ integersRest), (doubles A.++ doublesRest))
            (info : infoRest, ret)
