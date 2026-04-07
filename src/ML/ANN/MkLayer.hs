module ML.ANN.MkLayer where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Matrix 
import ML.ANN.Types
import Prelude as P

heWeightInit :: Int -> [Double] -> [Double]
heWeightInit numIns rands = P.map (\x -> x * 2.0 / (sqrt (P.fromIntegral numIns :: Double))) rands

mkWeights :: Int -> Int -> [Double] -> (Weights, [Double])
mkWeights numIns numOuts rands = do
    let m = use (fromList (Z:.numOuts:.numIns) (heWeightInit numIns rands))
        r = P.drop (numOuts * numIns) rands
    (AccMat m Outp Inp, r )

mkBiases :: Int -> Int -> [Double] -> (Biases, [Double])
mkBiases numIns numOuts rands = do
    let b = use (fromList (Z:.numOuts:.1) (heWeightInit numIns rands))
        r = P.drop (numOuts) rands
    (AccMat b Outp One, r)

mkLayer :: Int -> LSpec -> [Double] -> (Layer, [Double])
mkLayer numIns lspec rands = do
    let numOuts = P.foldr (+) 0 (P.map (\(i, _) -> i) lspec)
        (weights, rands') = mkWeights numIns numOuts rands
        (biases, rands'') = mkBiases numIns numOuts rands'
        zerosW = use (A.fromList (Z:.numOuts:.numIns) (P.repeat 0.0))
        zerosV = use (A.fromList (Z:.numOuts:.1) (P.repeat 0.0))
        wM = AccMat zerosW Outp Inp
        wV = AccMat zerosW Outp Inp
        bM = AccMat zerosV Outp One
        bV = AccMat zerosV Outp One
        l = Layer { lnumTimes = 0, lnumInputs = numIns, lweights = weights, lbiases = biases, lweightsVel = wV, lbiasesVel = bV, lweightsMom = wM, lbiasesMom = bM, llspec = lspec}
    (l, rands'')

mkInpLayer :: LSpec -> [Double] -> (Layer, [Double])
mkInpLayer lspec rands = do
    let numOuts = P.foldr (+) 0 (P.map P.fst lspec)
        (weights, rands') = mkBiases numOuts numOuts rands
        (biases, rands'') = mkBiases numOuts numOuts rands'
        zerosV = AccMat (use (A.fromList (Z:.numOuts:.1) (P.repeat 0.0))) Outp One
        l = InpLayer {vnumTimes = 0, vweights = weights, vbiases = biases, vlspec = lspec, vweightsMom = zerosV, vweightsVel = zerosV, vbiasesMom = zerosV, vbiasesVel = zerosV}
    (l, rands'')
