module Main where

import ML.ANN
import Data.Array.Accelerate as A
import Data.Array.Accelerate.LLVM.PTX
import Prelude as P

import System.Console.GetOpt
import System.Environment
import Data.Maybe
import Data.ByteString.Lazy as B
import Text.Printf
import System.IO
import Data.List.Split

data Options = Options { optInputNet :: String }

startOptions :: Options
startOptions = Options { optInputNet = "input.ann" }

options :: [OptDescr (Options -> IO Options)]
options = [ Option "I" ["input-net"] (ReqArg (\x -> \opt -> return opt { optInputNet = x } ) "input.ann") "input neural net file name"]

getInputs :: [String] -> [Vector Double]
getInputs list = do
    let listStrings = P.map (\x -> endBy "," x) list
        listDoubles = P.map (\x -> (P.map (\y -> read y :: Double) x)) listStrings
        listVects = P.map (\x -> fromList (Z:.(P.length x)) x) listDoubles
    listVects

stringList2string :: [String] -> String -> String
stringList2string [] _ = ""
stringList2string [last] _ = last
stringList2string ( h : t) inBetween = h P.++ inBetween P.++ (stringList2string t inBetween)


main :: IO ()
main = do
    hSetBuffering stdout (BlockBuffering Nothing)
    args <- getArgs
    let (actions, _, errors) = getOpt RequireOrder options args
    opts <- P.foldl (>>=) (return startOptions) actions
    let Options { optInputNet = inputNet } = opts
    annInBS <- B.readFile inputNet
    input <- System.IO.getContents
    let inpLines = endBy "\n" input
    (blinfo, blockv) <- bs2block annInBS
    let fn = runN (calcNetwork (block2network (blinfo, (use blockv))))
        inputs = getInputs inpLines
        outputs = P.map (\x -> fn x) inputs
        outputsDoubleLists = P.map (\x -> toList x) outputs
        outputsStringsLists = P.map (\x -> P.map (\y -> printf "%0.5F" y) x) outputsDoubleLists
        strings = P.map (\x -> stringList2string x "," ) outputsStringsLists
        output = stringList2string strings "\n"
    P.putStr output
