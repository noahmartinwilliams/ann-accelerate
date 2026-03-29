module Main where

import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter
import ML.ANN.Block
import ML.ANN.BPLayer
import ML.ANN.ErrorFn
import ML.ANN.InfLayer
import ML.ANN.LLayer
import ML.ANN.MkLayer
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.Exit
import System.Random

main :: IO ()
main = do
    g <- getStdGen
    let n = mkNetwork g [[(2, Sigmoid)], [(3, Sigmoid)], [(1, Sigmoid)]] (SGDOptim (constant 0.001))
        (blinfo, block) = network2block n
        (_, rblock) = A.unlift block :: (Acc (Vector Int), Acc (Vector Double))
        n' = block2network blinfo block
        (_, block') = network2block n'
        (_, rblock') = A.unlift block' :: (Acc (Vector Int), Acc (Vector Double))
        diff = A.zipWith (-) rblock rblock'
        summed = A.sum (A.zipWith (*) diff diff)
        summed' = run summed
    if ((A.toList summed') P.!! 0) P.>= 0.001
    then
        exitFailure
    else
        exitSuccess
