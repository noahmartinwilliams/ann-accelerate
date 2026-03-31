{-# LANGUAGE DeriveGeneric, OverloadedStrings #-}
module Main where

import Control.Monad.Reader
import Data.Aeson 
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.LLVM.PTX as PTX
import qualified Data.ByteString as B
import Data.List.Split
import Data.Maybe
import GHC.Generics
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import System.IO
import System.Random

data Conf = Conf { inputAF :: String, miniBatchSize :: Int, layers :: String, optimizer :: String, lr :: Double, beta1 :: Double, beta2 :: Double, costF :: String } deriving(Generic, Show)

instance ToJSON Conf where
    toEncoding = genericToEncoding defaultOptions

instance FromJSON Conf 

getConf :: String -> Maybe Conf
getConf inp = do
    let bs = fromString inp
    decode bs :: Maybe Conf

main :: IO ()
main = do
    c <- readFile "configsMnist.txt"
    let lines = endBy "\n" c
        g = 100
    trainImgs <- B.readFile "train-images-idx3-ubyte"
    trainAnswers <- B.readFile "train-labels-idx1-ubyte"
    let configs = P.filter (isJust) (P.map getConf lines)
        configs' = P.map (\(Just x) -> x) configs
        fnums = [0..]
        strs = P.zipWith (\x -> \y -> runReader (runNeural g x trainImgs trainAnswers ) y) fnums configs' 
    saveResults strs

saveResults :: [(String, String)] -> IO ()
saveResults [] = return ()
saveResults ((fname, fcontents) : tail) = do
    writeFile fname fcontents
    saveResults tail

runNeural :: Int -> Int -> B.ByteString -> B.ByteString -> Reader Conf (String, String)
runNeural seed lineNo imgs answers = do
    let gen = mkStdGen seed
    cnf <- ask
    neural <- getNeural gen ( optimizer cnf)
    let imgs' = B.drop 16 imgs
        answers' = B.drop 8 answers
        samps = mkSamps (miniBatchSize cnf) imgs' answers'
    (errs, _, _, _) <- runNeural' neural samps
    let (err' : errs') = P.map (show) errs
        retStr = P.foldl (\x -> \y -> x P.++ "\n" P.++ y) err' errs'
        retName = "results/" P.++ (show lineNo) P.++ ".txt"
    (return (retName, retStr))

bsSplitEvery :: Int -> B.ByteString -> [B.ByteString]
bsSplitEvery i bs | (B.null bs) = []
bsSplitEvery i bs = (B.take i bs ) : (bsSplitEvery i (B.drop i bs))

bs2Mat :: Int -> B.ByteString -> Matrix Double
bs2Mat mbs bs = do
    let uped = B.unpack bs
        asDoubles = P.map (\x -> P.fromIntegral x :: Double) uped
        scaled = P.map (\x -> (x - 128.0) / 128.0) asDoubles
    A.fromList (Z:.(28*28):.mbs) scaled

bs2Answer :: B.ByteString -> Matrix Double
bs2Answer bs | (B.head bs) P.== 0 = fromList (Z:.10:.1) [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 1 = fromList (Z:.10:.1) [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 2 = fromList (Z:.10:.1) [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 3 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 4 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 5 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 6 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 7 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
bs2Answer bs | (B.head bs) P.== 8 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
bs2Answer bs | (B.head bs) P.== 9 = fromList (Z:.10:.1) [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

combineAnswers :: [Matrix Double] -> Matrix Double
combineAnswers (h : ls) = I.run (P.foldl (A.++) (use h) (P.map use ls))

mkSamps :: Int -> B.ByteString -> B.ByteString -> [(Matrix Double, Matrix Double)]
mkSamps mbs imgs answers = do
    let splitImgs = bsSplitEvery (mbs * 28 * 28) imgs
        imgMats = P.map (bs2Mat mbs) splitImgs
        answerMats = bsSplitEvery mbs answers
        answerMats' = chunksOf mbs (P.map bs2Answer answerMats)
        answerMats'' = P.map combineAnswers answerMats'
    P.zip imgMats answerMats''

getNeural :: StdGen -> String -> Reader Conf Network
getNeural g "SGD" = do
    cnf <- ask
    let lr1 = lr cnf
        errFn = getErrorFn (costF cnf)
        mbs = (miniBatchSize cnf)
        lsp = read (layers cnf) :: [LSpec]
        net = mkNetwork g ([((28*28), Relu)] : lsp) (SGDOptim (constant lr1)) errFn
    return net

runNeural' :: Network -> [(Matrix Double, Matrix Double)] -> Reader Conf ([Double], [Matrix Double], Vector Int, Vector Double)
runNeural' net samples = do
    cnf <- ask
    let mbs = miniBatchSize cnf
        (blinfo, block) = network2block net
        block' = I.run block
        fn = PTX.runN (trainMiniBatch mbs blinfo )
    return (runner fn block' samples) where

        runner :: ((Vector Int, Vector Double) -> (Matrix Double, Matrix Double) -> (Matrix Double, Matrix Double, Vector Int, Vector Double)) -> (Vector Int, Vector Double) -> [(Matrix Double, Matrix Double)] -> ([Double], [Matrix Double], Vector Int, Vector Double)
        runner fn bl [last] = do
            let (errs, bps, vi, vd) = fn bl last 
            ([P.sum (A.toList errs)], [bps], vi, vd) 
                
        runner fn bl (first : rest) = do
            let (err, bp, vi, vd) = fn bl first
                (err', bp', vi', vd') = runner fn (vi, vd) rest
                errL = P.sum (A.toList err)
            (errL : err', bp : bp', vi', vd')
        
getErrorFn :: String -> ErrorFn
getErrorFn "MSE" = (mseErrorFn, dmseErrorFn)

