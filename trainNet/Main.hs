module Main where

import Prelude as P
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX
import System.Console.GetOpt
import System.Environment
import System.IO
import Data.Maybe
import ML.ANN
import Data.ByteString.Lazy as B
import Data.List.Split
import Text.Printf

data Options = Options { optNet :: String, optInput :: String, optOutput :: String, optSave :: String }

startOptions :: Options
startOptions = Options { optNet = "network.ann", optInput = "input.txt", optOutput = "output.txt", optSave = "network-trained.ann" }

options :: [OptDescr (Options -> IO Options)]
options = [ Option "I" ["input-net"] (ReqArg (\arg -> \opt -> return opt { optNet = arg }) "network.ann") "input neural net file.",
            Option "i" ["input-data"] (ReqArg (\arg -> \opt -> return opt { optInput = arg }) "input.txt") "input data file.",
            Option "o" ["output-data"] (ReqArg (\arg -> \opt -> return opt { optOutput = arg }) "output.txt") "output data file.",
            Option "O" ["output-net"] (ReqArg (\arg -> \opt -> return opt { optSave = arg }) "network-trained.ann") "output neural net file." ]


train :: (Vector Int, Vector Double) -> [(Vector Double, Vector Double)] -> ((Vector Int, Vector Double) -> (Vector Double, Vector Double) -> (Vector Double, Vector Int, Vector Double)) -> ([Vector Double], (Vector Int, Vector Double))
train block [] _ = ([], block)
train block ( head : rest ) fn = do
    let (err, ints, doubles) = fn block head 
        (rest2, block2) = train (ints, doubles) rest fn
    (err : rest2, block2)

getSamplesLL :: [String] -> [([Double], [Double])]
getSamplesLL inplist = do
    let (inputs, outputs) = P.unzip (P.map (\x -> ((endBy "#" x) P.!! 0, (endBy "#" x) P.!! 1)) inplist)
        inputsl = P.map (\x -> endBy "," x) inputs
        outputsl = P.map (\x -> endBy "," x) outputs
        inputsld = P.map (\x -> P.map (\y -> read y :: Double) x) inputsl
        outputsld = P.map (\x -> P.map (\y -> read y :: Double) x) outputsl
    P.zip inputsld outputsld

writer :: [String] -> IO ()
writer [] = return ()
writer ( head : tail ) = do
    System.IO.putStr head
    writer tail

main :: IO ()
main = do
    hSetBuffering stdout LineBuffering
    args <- getArgs
    let (actions, _, errors) = getOpt RequireOrder options args
    opts <- P.foldl (>>=) (return startOptions) actions
    let Options { optNet = netfilename, optInput = inputfilename, optOutput = outputfilename, optSave = outnetfilename} = opts
    annInBS <- B.readFile netfilename 
    input <- System.IO.getContents
    let inplines = endBy "\n" input
    (blinfo, blockv) <- bs2block annInBS
    let fn = runN (\x -> \y -> trainOnce (AccANN blinfo x) mseCFn y)
        samplesll = getSamplesLL inplines
        samplesvv = P.map (\(x, y) -> ((A.fromList (Z:.(P.length x)) x :: Vector Double), (A.fromList (Z:.(P.length y)) y :: Vector Double))) samplesll
        (errorsVs, blockv2) = train blockv samplesvv fn 
        errorsStr = P.map (\x -> (printf "%.7F" ((toList x) P.!! 0) )  P.++ "\n") errorsVs
    writer errorsStr
    B.writeFile outnetfilename (block2bs (blinfo, blockv2))
