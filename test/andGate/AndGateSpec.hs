module Main where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import ML.ANN.ActFunc
import ML.ANN.Block
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.Exit
import System.Random

genSamples :: StdGen -> [(Matrix Double, Matrix Double)]
genSamples g = do
    let ns = randoms g :: [Double]
    intern ns where
        intern :: [Double] -> [(Matrix Double, Matrix Double)]
        intern ( a : b : rest) | (a P.>= 0.0 ) P.&& (b P.>= 0.0) = (fromList (Z:.2:.1) [a, b], fromList (Z:.1:.1) [1.0]) : (intern rest)
        intern ( a : b : rest ) = (fromList (Z:.2:.1) [a, b], fromList (Z:.1:.1) [0.0]) : (intern rest)

main :: IO ()
main = do
    let g = mkStdGen 100
        n = mkNetwork g [[(2, Sigmoid)], [(5, Sigmoid)], [(1, Sigmoid)]] (SGDOptim (constant 0.01))
        (blinfo, blockA) = network2block n
        block = run blockA
        fn = runN (trainOnce blinfo )
        numSamples = 2048
        samples = P.take numSamples (genSamples g)
        (err0, _, _, _) = fn block (samples P.!! 0)
        (errs, _, _, _) = runner fn block samples
        last = errs P.!! (numSamples - 1)
    if (P.sum (A.toList last)) P.> (P.sum (A.toList err0))
    then
        die ("Expected last error to be less than first error. Got: " P.++ (show err0) P.++ " and " P.++ (show last) P.++ ".\n")
    else
        exitSuccess

runner :: ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Matrix Double, Matrix Double, Vector Int, Vector Double)) -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> ([Matrix Double], [Matrix Double], Vector Int, Vector Double)
runner fn block (sample : rest) = do
    let (err, bp, blockI, blockD) = fn block sample
        (err', bp', blockI', blockD') = runner fn (blockI, blockD) rest
    (err : err', bp : bp', blockI', blockD')
