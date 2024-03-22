module Main where

import Prelude as P
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter

import System.Random
import System.Console.GetOpt
import System.Environment
import Data.Maybe
import ML.ANN
import Data.ByteString.Lazy as B

data Options = Options { optOutput :: String, optRandSeed :: Int, optTrueRand :: Bool, optLayers :: [LSpec], optOptimizer :: Optim }

startOptions :: Options
startOptions = Options { optOutput = "output.txt", optRandSeed = 100, optTrueRand = False, optLayers = [], optOptimizer = (SGD 0.001)}

options :: [OptDescr (Options -> IO Options)]
options = [ Option "o" ["output"] (ReqArg (\x -> \opt -> return opt { optOutput = x }) "output.ann") "output file",
            Option "s" ["seed"] (ReqArg (\arg -> \opt -> return opt { optRandSeed = (read arg :: Int) }) "100") "random seed",
            Option "r" ["rand"] (NoArg (\opt -> return opt { optTrueRand = True})) "use truly random seed",
            Option "l" ["layers"] (ReqArg (\arg -> \opt -> return opt { optLayers = (read arg :: [LSpec]) }) "[[Sigmoid 2], [Sigmoid 3], [Sigmoid 1]]") "specify layers",
            Option "O" ["optim"] (ReqArg (\arg -> \opt -> return opt { optOptimizer = (read arg :: Optim) }) "SGD 0.0001") "specify optimizer"]

main :: IO ()
main = do
    rand <- getStdGen
    args <- getArgs
    let (actions, _, errors) = getOpt RequireOrder options args
    opts <- P.foldl (>>=) (return startOptions) actions
    let Options { optOutput = outputStr, optRandSeed = seed, optTrueRand = trueRand, optLayers = layers, optOptimizer = optim } = opts
    let seed2 = if trueRand then rand else mkStdGen seed
        net = mkNetwork seed2 layers optim 
        (blinfo, blocka) = network2block net
        blockv = run blocka
        bs = block2bs (blinfo, blockv)
    B.writeFile outputStr bs
