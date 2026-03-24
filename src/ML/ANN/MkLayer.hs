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

