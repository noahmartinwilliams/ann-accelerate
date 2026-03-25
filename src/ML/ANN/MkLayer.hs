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

mkBiases :: Int -> [Double] -> (Biases, [Double])
mkBiases numOuts rands = do
    let b = use (fromList (Z:.numOuts:.1) (heWeightInit numOuts rands))
        r = P.drop (numOuts) rands
    (AccMat b Outp One, r)

mkLayer :: Int -> LSpec -> [Double] -> LayerType -> (Layer, [Double])
mkLayer numIns lspec rands (SGD) = do
    let numOuts = P.foldr (+) 0 (P.map (\(i, _) -> i) lspec)
    let (weights, rands') = mkWeights numIns numOuts rands
        (biases, rands'') = mkBiases numOuts rands'
    (Layer { lweights = weights, lbiases = biases, ltype = SGD, llspec = lspec}, rands'')
