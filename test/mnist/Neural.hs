module Neural where

import Conf
import Control.Monad.Reader
import Data.Array.Accelerate as A
import Data.Array.Accelerate.Interpreter as I
import Data.Array.Accelerate.LLVM.PTX as PTX
import qualified Data.ByteString as B
import Data.List.Split
import ML.ANN.Block
import ML.ANN.ErrorFn
import ML.ANN.Network
import ML.ANN.Types
import Prelude as P
import Samps
import System.Random

runNeural :: Int -> Int -> B.ByteString -> B.ByteString -> B.ByteString -> B.ByteString -> Reader Conf (String, String, String, String)
runNeural seed lineNo imgs answers testImgs testAnswers = do
    let gen = mkStdGen seed
    cnf <- ask
    neural <- getNeural gen ( optimizer cnf)
    let imgs' = B.drop 16 imgs
        answers' = B.drop 8 answers
        testImgs' = B.drop 16 imgs
        testAnswers' = B.drop 8 testAnswers
        samps = mkSamps (miniBatchSize cnf) imgs' answers'
        samps' = mkSamps 1 testImgs' testAnswers'
        (blinfo, _) = network2block neural
    (errs, _, vi, vd) <- runNeural' neural samps
    let (err' : errs') = P.map (show) errs
        retStr = P.foldl (\x -> \y -> x P.++ "\n" P.++ y) err' errs'
        retName = "results/" P.++ (show lineNo) P.++ ".txt"
        retName' = "results/" P.++ "test-" P.++ (show lineNo) P.++ ".txt"
        net' = block2network blinfo (use (vi, vd))
        testRes = testResults net' samps'
        (err'' : errs'') = P.map (show) testRes
        retStr' = P.foldl (\x -> \y -> x P.++ "\n" P.++ y)  err'' errs''
    (return (retName, retStr, retName', retStr'))

testResults :: Network -> [(Matrix Double, Matrix Double)] -> [Double]
testResults net samps = do
    let fn = PTX.runN (inferNetwork net) 
        (inps, outps) = P.unzip samps
        outps' = P.map (A.toList) outps
        results = P.map fn inps
        results' = P.map (A.toList) results
        err a b = P.sum (P.zipWith (\x -> \y -> (x - y) * (x - y)) a b)
        errs = P.map (\(x, y) -> err x y) (P.zip outps' results')
    errs

getNeural :: StdGen -> String -> Reader Conf Network
getNeural g "Adam" = do
    cnf <- ask
    let lr1 = lr cnf
        b1 = beta1 cnf
        b2 = beta2 cnf
        errFn = getErrorFn (costF cnf)
        mbs = miniBatchSize cnf
        lsp = read (layers cnf) :: [LSpec]
        net = mkNetwork g ([((28*28), Sigmoid)] : lsp) (AdamOptim (constant lr1) (constant b1) (constant b2)) errFn
    return net


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
